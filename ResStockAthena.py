"""
# ResStockAthena
- - - - - - - - -
A class to run AWS Athena queries to get various data from a ResStock run. All queries and aggregation that can be
common accross different ResStock projects should be implemented in this class. For queries that are project specific, a
new class can be created by inheriting ResStockAthena and adding in the project specific logic and queries.

:author: Rajendra.Adhikari@nrel.gov
"""

import os
import boto3
import botocore
import pythena
from typing import List, Tuple, Union
import time
import logging
from threading import Thread
from botocore.exceptions import ClientError
import pandas as pd
import datetime
from dateutil import parser
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExistsException(Exception):
    def __init__(self, message, existing_data=None):
        super(DataExistsException, self).__init__(message)
        self.existing_data = existing_data


class ResStockAthena:
    def __init__(self, workgroup: str,
                 db_name: str,
                 buildstock_type: str = None,
                 table_name: Union[str, Tuple[str, str]] = None,
                 region_name: str = 'us-west-2',
                 timestamp_column_name='time',
                 sample_weight_column="build_existing_model.sample_weight",
                 execution_history=None) -> None:
        """
        A class to run common Athena queries for ResStock runs and download results as pandas dataFrame
        Args:
            db_name: The athena database name
            buildstock_type: 'resstock' or 'comstock' runs
            table_name: If a single string is provided, say, 'mfm_run', then it must correspond to two tables in athena
                        named mfm_run_baseline and mfm_run_timeseries. Or, two strings can be provided as a tuple, (such
                        as 'mfm_run_2_baseline', 'mfm_run5_timeseries') and they must be a baseline table and a
                        timeseries table.
            region_name: The AWS region where the database exists. Defaults to 'us-west-2'.
            timestamp_column_name: The column name for the time column. Defaults to 'time'
            execution_history: A temporary files to record which execution is run by the user, to help stop them. Will
                               use .execution_history if not supplied.
        """
        self.workgroup = workgroup
        self.buildstock_type = buildstock_type
        self.py_thena = pythena.Athena(db_name, region_name)
        self.s3 = boto3.client('s3')
        self.aws_athena = boto3.client('athena', region_name=region_name)
        self.aws_glue = boto3.client('glue', region_name=region_name)
        self.db_name = db_name
        self.region_name = region_name
        self.timestamp_column_name = timestamp_column_name
        self.cache = {}  # To store small but frequently queried result, such as total number of timesteps
        if sample_weight_column:
            self.sample_weight_column = self.make_column_string(sample_weight_column)
        else:
            self.sample_weight_column = 1
        if isinstance(table_name, str):
            self.ts_table_name = f'{table_name}_timeseries'
            self.baseline_table_name = f'{table_name}_baseline'
        elif isinstance(table_name, tuple):
            self.baseline_table_name = f'{table_name[0]}'
            self.ts_table_name = f'{table_name[1]}'
        else:
            self.baseline_table_name = None
            self.ts_table_name = None

        if self.baseline_table_name and self.ts_table_name:
            self.ts_table = self.aws_glue.get_table(DatabaseName=self.db_name, Name=self.ts_table_name)['Table']
            self.baseline_table = self.aws_glue.get_table(DatabaseName=self.db_name,
                                                          Name=self.baseline_table_name)['Table']

        self.tables = {}
        self.join_list = {}

        self.batch_query_status_map = {}
        self.batch_query_id = 0

        if not execution_history:
            self.execution_history_file = '.execution_history'
        else:
            self.execution_history_file = execution_history

        if os.path.exists(self.execution_history_file):
            with open(self.execution_history_file, 'r') as f:
                existing_entries = f.readlines()

            valid_entries = []
            for entry in existing_entries:
                try:
                    entry_time, _ = entry.split(',')
                    if time.time() - float(entry_time) < 24 * 60 * 60:  # discard history if more than a day old
                        valid_entries += entry
                except (ValueError, TypeError):
                    pass

            with open(self.execution_history_file, 'w') as f:
                f.writelines(valid_entries)

    @property
    def execution_ids_history(self):
        exe_ids = []
        if os.path.exists(self.execution_history_file):
            with open(self.execution_history_file, 'r') as f:
                for line in f.readlines():
                    _, exe_id = line.split(',')
                    exe_ids.append(exe_id.strip())
        return exe_ids

    def delete_table(self, table_name):
        delete_table_query = f"""DROP TABLE {self.db_name}.{table_name};"""
        result, reason = self.execute_raw(delete_table_query)
        if result.upper() == "SUCCEEDED":
            return "SUCCEEDED"
        else:
            raise Exception(f"Deleting it failed. Reason: {reason}")

    def add_table(self, table_name, table_df, s3_bucket, s3_prefix, override=False):
        s3_location = s3_bucket + '/' + s3_prefix
        save = False
        s3_data = self.s3.list_objects(Bucket=s3_bucket, Prefix=f'{s3_prefix}/{table_name}/{table_name}.csv')
        if 'Contents' in s3_data and override == False:
            # existing_data = pd.read_csv(f's3://{s3_location}/{table_name}/{table_name}.csv')
            raise DataExistsException("Table already exists", f's3://{s3_location}/{table_name}/{table_name}.csv')
        else:
            print("Saving the csv to s3")
            table_df.to_csv(f's3://{s3_location}/{table_name}/{table_name}.csv', index=False)
            print("Saving Done.")

        column_formats = []
        for column_name, dtype in table_df.dtypes.items():
            if np.issubdtype(dtype, np.integer):
                type = "int"
            elif np.issubdtype(dtype, np.floating):
                type = "double"
            else:
                type = "string"
            column_formats.append(f"`{column_name}` {type}")

        column_formats = ",".join(column_formats)

        table_create_query = f"""
        CREATE EXTERNAL TABLE {self.db_name}.{table_name} ({column_formats}
        )
        ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
        WITH SERDEPROPERTIES (
          'skip.header.line.count' = '1',
          'field.delim' = ','
        ) LOCATION 's3://{s3_location}/{table_name}/'
        TBLPROPERTIES ('has_encrypted_data'='false');
        """
        print("Running create table query.")
        result, reason = self.execute_raw(table_create_query)
        if result.lower() == "failed" and 'alreadyexists' in reason.lower():
            if not override:
                existing_data = pd.read_csv(f's3://{s3_location}/{table_name}/{table_name}.csv')
                raise DataExistsException("Table already exists", existing_data)
            delete_table_query = f"""
            DROP TABLE {self.db_name}.{table_name};
            """
            result, reason = self.execute_raw(delete_table_query)
            if result.upper() == "SUCCEEDED":
                result, reason = self.execute_raw(table_create_query)
                if result.upper() == "SUCCEEDED":
                    return "SUCCEEDED"
                else:
                    raise Exception(f"There was an existing table named {table_name} which is now successfully deleted,"
                                    f"but new table failed to be created. Reason: {reason}")
            else:
                raise Exception(f"There was an existing table named {table_name}. Deleting it failed."
                                f" Reason: {reason}")
        elif result.upper() == "SUCCEEDED":
            return "SUCCEEDED"
        else:
            raise Exception(f"Failed to create the table. Reason: {reason}")

    def execute_raw(self, query, db=None, run_async=False):

        if not db:
            db = self.db_name

        response = self.aws_athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={
                'Database': db
            },
            WorkGroup=self.workgroup)

        query_execution_id = response['QueryExecutionId']
        if run_async:
            return query_execution_id
        start_time = time.time()
        query_stat = ""
        while time.time() - start_time < 30*60:  # 30 minute timeout
            query_stat = self.aws_athena.get_query_execution(QueryExecutionId=query_execution_id)
            if query_stat['QueryExecution']['Status']['State'].lower() not in ['pending', 'running', 'queued']:
                reason = query_stat['QueryExecution']['Status'].get('StateChangeReason', '')
                return query_stat['QueryExecution']['Status']['State'], reason
            time.sleep(1)

        raise TimeoutError(f"Query failed to complete within 30 mins. Last status: {query_stat}")

    def _save_execution_id(self, execution_id):
        with open(self.execution_history_file, 'a') as f:
            f.write(f'{time.time()},{execution_id}\n')

    def execute(self, query, db=None, run_async=False):
        """
        Executes a query
        Args:
            query: The SQL query to run in Athena
            db: Optionally specify the database on which to run the query. Defaults to the database supplied during
                initialization
            run_async: Whether to wait until the query completes (run_async=False) or return immediately
            (run_async=True).

        Returns:
            if run_async is False, returns the results dataframe.
            if run_async is  True, returns the query_execution_id. Use `get_query_result` to get the result after
            the query is completed. Use :get_status: to check the status of the query.
        """
        if not db:
            db = self.db_name

        self.py_thena._Athena__database = db  # override the DB in pythena
        res = self.py_thena.execute(query, save_results=True, run_async=run_async, workgroup=self.workgroup)
        self.py_thena._Athena__database = self.db_name  # restore the DB name

        if run_async:
            # in case of asynchronous run, you get the execution id only. Save it before returning
            self._save_execution_id(res)
            return res
        else:
            # in case of synchronous run, just return the dataFrame
            return res[0]

    def print_all_batch_query_status(self):
        for count in self.batch_query_status_map.keys():
            print(f'Query {count}: {self.get_batch_query_report(count)}\n')

    def stop_batch_query(self, batch_id):
        """
        Stops all the queries running under a batch query
        Args:
            batch_id: The batch_id of the batch_query. Returned by :py:sumbit_batch_query

        Returns:
            None
        """
        self.batch_query_status_map[batch_id]['to_submit_ids'].clear()
        for exec_id in self.batch_query_status_map[batch_id]['submitted_execution_ids']:
            self.stop_query(exec_id)

    def print_failed_query_errors(self, batch_id):
        stats = self.batch_query_status_map.get(batch_id, None)
        if stats:
            for i, exe_id in enumerate(stats['submitted_execution_ids']):
                completion_stat = self.get_query_status(exe_id)
                if completion_stat in ['FAILED']:
                    print(f"Query id: {exe_id}. \n Query string: {stats['submitted_queries'][i]}."
                          f"\nError: {self.get_query_error(exe_id)}\n")

    def get_ids_for_failed_queries(self, batch_id):
        failed_ids = []
        for i, exe_id in enumerate(self.batch_query_status_map[batch_id]['submitted_execution_ids']):
            completion_stat = self.get_query_status(exe_id)
            if completion_stat in ['FAILED', 'CANCELLED']:
                failed_ids.append(self.batch_query_status_map[batch_id]['submitted_ids'][i])
        return failed_ids

    def get_batch_query_report(self, batch_id: int):
        """
        Returns the status of the queries running under a batch query.
        Args:
            batch_id: The batch_id of the batch_query.

        Returns:
            A dictionary detailing status of the queries.
        """
        stats = self.batch_query_status_map.get(batch_id, None)
        if stats:
            success_count = 0
            fail_count = 0
            running_count = 0
            other = 0
            for exe_id in stats['submitted_execution_ids']:
                completion_stat = self.get_query_status(exe_id)
                if completion_stat == 'RUNNING':
                    running_count += 1
                elif completion_stat == 'SUCCEEDED':
                    success_count += 1
                elif completion_stat in ['FAILED', 'CANCELLED']:
                    fail_count += 1
                else:
                    other += 1

            result = {'Submitted': len(stats['submitted_ids']),
                      'Running': running_count,
                      'Pending': len(stats['to_submit_ids']) + other,
                      'Completed': success_count,
                      'Failed': fail_count
                      }

            return result
        else:
            return None

    def did_batch_query_complete(self, batch_id):
        """
        Checks if all the queries in a batch query has completed or not.
        Args:
            batch_id: The batch_id for the batch_query

        Returns:
            True or False
        """
        status = self.get_batch_query_report(batch_id)
        if status['Pending'] > 0 or status['Running'] > 0:
            return False
        else:
            return True

    def get_batch_query_result(self, batch_id, combine=True, no_block=False):
        """
        Concatenates and returns the results of all the queries of a batchquery
        Args:
            batch_id (int): The batch_id for the batch_query
            no_block (bool): Whether to wait until all queries have completed or return immediately. If you use
                            no_block = true and the batch hasn't completed, it will throw BatchStillRunning exception.
            combine: Whether to combine the individual query result into a single dataframe

        Returns:
            The concatenated dataframe of the results of all the queries in a batch query.

        """
        last_time = time.time()
        last_report = None
        while True:
            if self.did_batch_query_complete(batch_id):
                query_exe_ids = self.batch_query_status_map[batch_id]['submitted_execution_ids']
                if len(query_exe_ids) == 0:
                    raise ValueError("No query was submitted successfully")
                res_df_array = []
                for index, exe_id in enumerate(query_exe_ids):
                    df = self.get_query_result(exe_id)
                    df['query_id'] = index
                    res_df_array.append(df)
                if combine:
                    return pd.concat(res_df_array)
                else:
                    return res_df_array
            else:
                if no_block:
                    raise Exception('Batch query not completed yet.')
                else:
                    report = self.get_batch_query_report(batch_id)
                    if time.time() - last_time > 60 or last_report is None or report != last_report:
                        logger.info(report)
                        last_report = report
                        last_time = time.time()
                    time.sleep(20)

    def submit_batch_query(self, queries: List[str]):
        """
        Submit multiple related queries
        Args:
            queries: List of queries to submit. Setting `get_query_only` flag while making calls to aggregation
                    functions is easiest way to obtain queries. The queries in the list can either be a string in or a
                    tuple of (database_name, query_string). When just a string is used, the database defaults to the one
                    supplied during init.

        Returns:
            An integer representing the batch_query id. The id can be used with other batch_query functions.
        """
        queries = list(queries)
        to_submit_ids = list(range(len(queries)))
        id_list = list(to_submit_ids)  # make a copy
        submitted_ids = []
        submitted_execution_ids = []
        submitted_queries = []
        self.batch_query_id += 1
        batch_query_id = self.batch_query_id
        self.batch_query_status_map[batch_query_id] = {'to_submit_ids': to_submit_ids,
                                                       'all_ids': list(id_list),
                                                       'submitted_ids': submitted_ids,
                                                       'submitted_execution_ids': submitted_execution_ids,
                                                       'submitted_queries': submitted_queries
                                                       }

        def run_queries():
            while to_submit_ids:
                current_id = to_submit_ids[0]  # get the first one
                current_query = queries[0]
                if isinstance(current_query, tuple):
                    db, current_query = current_query
                else:
                    db = self.db_name
                try:
                    execution_id = self.execute(current_query, db=db, run_async=True)
                    logger.info(f"Submitted queries[{current_id}]")
                    to_submit_ids.pop(0)  # if query queued successfully, remove it from the list
                    queries.pop(0)
                    submitted_ids.append(current_id)
                    submitted_execution_ids.append(execution_id)
                    submitted_queries.append(current_query)
                except ClientError as e:
                    if e.response['Error']['Code'] == 'TooManyRequestsException':
                        time.sleep(60)  # wait for a minute before submitting another query
                    elif e.response['Error']['Code'] == 'InvalidRequestException':
                        logger.info(f"Queries[{current_id}] is Invalid: {e.response['Message']} \n {current_query}")
                        to_submit_ids.pop(0)  # query failed, so remove it from the list
                        queries.pop(0)
                        raise
                    else:
                        raise

        query_runner = Thread(target=run_queries)
        query_runner.start()
        return batch_query_id

    def get_query_result(self, query_id):
        return self.py_thena.get_result(query_execution_id=query_id, save_results=True)

    def get_query_status(self, query_id):
        return self.py_thena.get_query_status(query_id)

    def get_query_error(self, query_id):
        return self.py_thena.get_query_error(query_id)

    def get_all_running_queries(self):
        """
        Gives the list of all running queries (for this instance)

        Return:
            List of query execution ids of all the queries that are currently running in Athena.
        """
        exe_ids = self.aws_athena.list_query_executions(WorkGroup=self.workgroup)['QueryExecutionIds']
        running_ids = [i for i in exe_ids if i in self.execution_ids_history and
                       self.py_thena.get_query_status(i) == "RUNNING"]
        return running_ids

    def stop_all_queries(self):
        """
        Stops all queries that are running in Athena for this instance.
        Returns:
            Nothing

        """
        for count, stat in self.batch_query_status_map.items():
            stat['to_submit_ids'].clear()

        running_ids = self.get_all_running_queries()
        for i in running_ids:
            self.py_thena.cancel_query(query_execution_id=i)

        logger.info(f"Stopped {len(running_ids)} queries")

    def stop_query(self, execution_id):
        """
        Stops a running query.
        Args:
            execution_id: The execution id of the query being run.

        Returns:

        """
        return self.aws_athena.stop_query_execution(QueryExecutionId=execution_id)

    def get_cols(self, table='baseline', fuel_type=None):
        """
        Returns the columns of for a particular table.
        Args:
            table: Name of the table. One of 'baseline' or 'timeseries'
            fuel_type: Get only the columns for this fuel_type ('electricity', 'gas' etc)

        Returns:
            A list of column names as a list of strings.
        """
        if table in ['timeseries', 'ts']:
            cols = self.ts_table['StorageDescriptor']['Columns']
            cols = [c['Name'] for c in cols]
            if fuel_type:
                cols = [c for c in cols if c not in ['building_id', self.timestamp_column_name]]
                cols = [c for c in cols if fuel_type in c]
            return cols
        elif table == 'baseline':
            cols = self.baseline_table['StorageDescriptor']['Columns']
            cols = [c['Name'] for c in cols]
            if fuel_type:
                cols = [c for c in cols if 'simulation_output_report' in c]
                cols = [c for c in cols if fuel_type in c]
            return cols
        else:
            tbl = self.aws_glue.get_table(DatabaseName=self.db_name, Name=table)['Table']
            cols = tbl['StorageDescriptor']['Columns']
            return [c['Name'] for c in cols]

    @staticmethod
    def simple_label(label):
        if label.startswith('simulation_output_report'):
            return label.split('.')[1]
        else:
            return label

    @staticmethod
    def dress_literal(a):
        if isinstance(a, float) or isinstance(a, int):
            return f"{a}"
        elif isinstance(a, str):
            return f"'{a}'"
        elif isinstance(a, datetime.datetime):
            return f"timestamp '{str(a)}'"
        else:
            raise TypeError(f'Unsupported Type: {type(a)}')

    @staticmethod
    def make_column_string(c):
        if not c.startswith('"'):
            return f'"{c}"'
        else:
            return c

    @classmethod
    def _get_restrict_string(cls, restrict):
        query = ''
        C = cls.make_column_string
        if restrict:
            query += " where "
            condition_strs = []
            for column, vals in restrict:
                if isinstance(vals, list) or isinstance(vals, tuple):
                    vals = [cls.dress_literal(v) for v in vals]
                    condition_strs.append(f'''({C(column)} in ({', '.join(vals)}))''')
                elif isinstance(vals, str) or isinstance(vals, int):
                    vals = cls.dress_literal(vals)
                    condition_strs.append(f'''({C(column)} = {vals})''')

            query += " and ".join(condition_strs)
        return query

    def get_distinct_vals(self, column: str, table_name: str = None, get_query_only: bool = False):
        table_name = self.baseline_table_name if table_name is None else table_name
        C = self.make_column_string
        query = f"select distinct {C(column)} from {C(table_name)}"

        if get_query_only:
            return query

        r, query = self.execute(query, run_async=False)
        return r[column]

    def get_distinct_count(self, column: str, table_name: str = None, weight_column: str = None,
                           get_query_only: bool = False):
        C = ResStockAthena.make_column_string
        if table_name is None:
            table_name = self.baseline_table_name if table_name is None else table_name
            weight = self.sample_weight_column
        else:
            weight = C(weight_column) if weight_column else 1

        query = f"select {C(column)}, sum(1) as raw_count, sum({weight}) as weighted_count from {C(table_name)} " \
            f"group by 1 order by 1"

        if get_query_only:
            return query

        r = self.execute(query, run_async=False)
        return r

    def get_successful_simulation_count(self, restrict: List[Tuple[str, List]] = [], get_query_only: bool = False):
        """
        Returns the results_csv table for the resstock run
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas integer counting the number of successful simulation
        """
        C = ResStockAthena.make_column_string
        query = f'''select count(*) as count from {C(self.baseline_table_name)}'''
        restrict = list(restrict)
        restrict.insert(0, ('completed_status', 'Success'))
        query += self._get_restrict_string(restrict)
        if get_query_only:
            return query

        return self.execute(query)

    def get_results_csv(self, restrict: List[Tuple[str, Union[List, str, int]]] = [], get_query_only: bool = False):
        """
        Returns the results_csv table for the resstock run
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe that is a subset of the results csv, that belongs to provided list of utilities
        """
        C = ResStockAthena.make_column_string
        query = f'''select * from {C(self.baseline_table_name)} '''
        query += self._get_restrict_string(restrict)

        if get_query_only:
            return query

        return self.execute(query)

    def get_building_ids(self, restrict: List[Tuple[str, List]] = [], get_query_only: bool = False):
        """
        Returns the list of buildings based on the restrict list
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the building ids belonging to the provided list of locations.

        """
        C = ResStockAthena.make_column_string
        query = f'''select building_id from {C(self.baseline_table_name)} '''
        query += self._get_restrict_string(restrict)
        if get_query_only:
            return query
        res = self.execute(query)
        return res

    def aggregate_annual(self,
                         enduses: List[str] = None,
                         group_by: List[str] = None,
                         order_by: List[str] = None,
                         join_list: List[Tuple[str, str, str]] = [],
                         weights: List[str] = [],
                         restrict: List[Tuple[str, List]] = [],
                         custom_sample_weight: float = None,
                         run_async: bool = False,
                         get_query_only: bool = False):
        """
        Aggregates the baseline annual result on select enduses.
        Check the argument description below to learn about additional features and options.
        Args:
            enduses: The list of enduses to aggregate. Defaults to all electricity enduses

            group_by: The list of columns to group the aggregation by.

            order_by: The columns by which to sort the result.

            join_list: Additional table to join to baseline table to perform operation. All the inputs (`enduses`,
                       `group_by` etc) can use columns from these additional tables. It should be specified as a list of
                       tuples.
                       Example: `[(new_table_name, baseline_column_name, new_column_name), ...]`
                                where baseline_column_name and new_column_name are the columns on which the new_table
                                should be joined to baseline table.

            weights: The additional column to use as weight. The "build_existing_model.sample_weight" is already used.

            custom_sample_weight: If the sample weight is different from build_existing_model.sample_weight, use
                                  this parameter to provide a different weight

            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`

            run_async: Whether to run the query in the background. Returns immediately if running in background,
                       blocks otherwise.
            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging


        Returns:
                if get_query_only is True, returns the query_string, otherwise,
                    if run_async is True, it returns a query_execution_id.
                    if run_async is False, it returns the result_dataframe

        """

        C = ResStockAthena.make_column_string
        if enduses is None:
            enduses = self.get_cols(table='baseline', fuel_type='electricity')

        if custom_sample_weight:
            sample_weight = str(custom_sample_weight)
        else:
            sample_weight = self.sample_weight_column

        n_units_col = C("build_existing_model.units_represented")

        total_weight = f'{sample_weight}'
        for weight in weights:
            total_weight += f' * {C(weight)} '

        enduse_cols = ', '.join([f"sum({C(c)} * {total_weight} / {n_units_col}) as {C(self.simple_label(c))}"
                                 for c in enduses])
        if not group_by:
            query = f"select {enduse_cols} from {C(self.baseline_table_name)}"
        else:
            group_by_cols = ", ".join([f'{C(g)}' for g in group_by])
            grouping_metrics_cols = f" sum(1) as raw_count, sum({total_weight}) as scaled_unit_count"
            select_cols = group_by_cols + ", " + grouping_metrics_cols
            if enduse_cols:
                select_cols += ", " + enduse_cols

            query = f"select {select_cols} from {C(self.baseline_table_name)}"

        join_clause = ''
        for new_table_name, baseline_column_name, new_column_name in join_list:
            join_clause += f''' join {C(new_table_name)} on {C(baseline_column_name)} =\
                            {C(new_table_name)}.{C(new_column_name)}'''
        query += join_clause

        query += self._get_restrict_string(restrict)

        if group_by:
            query += " group by " + ", ".join([f'{C(g)}' for g in group_by])
        if order_by:
            query += " order by " + ", ".join([f'{C(o)}' for o in order_by])

        if get_query_only:
            return query

        t = time.time()
        res = self.execute(query, run_async=True)
        if run_async:
            return res
        else:
            time.sleep(1)
            while time.time() - t < 30 * 60:
                stat = self.py_thena.get_query_status(res)
                if stat.upper() == 'SUCCEEDED':
                    result = self.get_query_result(res)
                    return result
                elif stat.upper() == 'FAILED':
                    error = self.get_query_error(res)
                    raise Exception(error)
                else:
                    logger.info(f'Query status is {stat}')
                    time.sleep(30)

            raise Exception(f'Query failed {self.py_thena.get_query_status(res)}')

    def aggregate_timeseries(self,
                             enduses: List[str] = None,
                             group_by: List[str] = None,
                             order_by: List[str] = None,
                             join_list: List[Tuple[str, str, str]] = [],
                             weights: List[str] = [],
                             custom_sample_weight: float = None,
                             restrict: [Tuple[str, List]] = [],
                             run_async: bool = False,
                             get_query_only: bool = False,
                             correction_factors_table: str = None,
                             limit=None
                             ):
        """
        Aggregates the timeseries result on select enduses.
        Check the argument description below to learn about additional features and options.
        Args:
            enduses: The list of enduses to aggregate. Defaults to all electricity enduses

            group_by: The list of columns to group the aggregation by.

            order_by: The columns by which to sort the result.

            join_list: Additional table to join to baseline table to perform operation. All the inputs (`enduses`,
                       `group_by` etc) can use columns from these additional tables. It should be specified as a list of
                       tuples.
                       Example: `[(new_table_name, baseline_column_name, new_column_name), ...]`
                                where baseline_column_name and new_column_name are the columns on which the new_table
                                should be joined to baseline table.

            weights: The additional column to use as weight. The "build_existing_model.sample_weight" is already used.

            custom_sample_weight: If the sample weight is different from build_existing_model.sample_weight, use
                                  this parameter to provide a different weight

            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            limit: The maximum number of rows to query

            run_async: Whether to run the query in the background. Returns immediately if running in background,
                       blocks otherwise.
            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging
            correction_factors_table: A correction factor table used for scaling timeseries during aggregation. Use
                                      add_table method to add a correction_factor table to Athena if it doesn't exist.
                                      Column names matching with the baseline table as well as datetime columns such as
                                      'day_of_week', 'day_of_year', 'hour', 'minute' and 'month' will be used for
                                      joining the correction_factor table to timeseries table during aggregation.
                                      The column named 'factor_all' is applied to all the columns, whereas columns named
                                      'factor_<timeseries_enduse_column_name>' is used for only the specific enduse.


        Returns:
                if get_query_only is True, returns the query_string, otherwise,
                    if run_async is True, it returns a query_execution_id.
                    if run_async is False, it returns the result_dataframe

        Further notes on using correction_factors_table (CFT) argument:
        CFT is used for calibration purpose to adjust the timeseries by multiplying with various correction factors.
        There are four types of columns in CFT which have special meanings.
        1. Time columns (Valid column names: day_of_week, day_of_year, hour, minute and month)
           These columns are used to align the factors properly with the timeseries table. You can have more than one of
           these columns and they will be ANDed. For example, if you have day_of_week column and hour column, then the
           factor in each row is applied to a particular hour of the particular day_of_week.
        2. Baseline columns (Valid column names: any column in baseline table starting with build_existing_model.<...>)
           These columns are used to link the factors with particular building types. You can have zero or more
           of these columns and they will be ANDed (i.e. the building to which the factor will be applied has to match
           all of the condition)
        3. Factor columns (Valid column names: factor_all, factor_electricity_cooling_kwh, ... factor_<enduse_column>)
           These columns contain the correction factor. The factor_all column will be used to adjust all the enduses
           whereas factor_<enduse_column> is used to adjust only that particular enduse.
        4. simulation_weight_correction_factor: This column must be optional and corrects for failed simulation.
        The CFT can contain other columns besides the one listed above, but they will have no effect on the timeseries
        aggregation. Also, for CFT to work, the CFT must exist on athena. You can use add_table function to add CFT to
        athena if it doens't already exists.

        Tiny CFT example:
        month, simulation_weight_correction_factor, factor_all
        1, 1, 1.1
        12, 1, 0.95

        This will result in the January enduse timeseries (for all buildings everywhere) to be multiplied by 1.1 and
        December timeseries to be multiplied by 0.95. Timeseries for all other months will not be modified.

        """

        C = self.make_column_string
        if not enduses:
            enduses = self.get_cols(table='timeseries', fuel_type='electricity')
        else:
            enduses = enduses.copy()

        if custom_sample_weight:
            sample_weight = str(custom_sample_weight)
        else:
            sample_weight = self.sample_weight_column
        n_units_col = C("build_existing_model.units_represented")

        total_weight = f'{sample_weight}'
        for weight in weights:
            total_weight += f' * {C(weight)} '

        if correction_factors_table:
            correction_columns = self.get_cols(table=correction_factors_table)
            # c[7:] extracts enduse_name from 'factor_enduse_name' columns
            correction_enduses = [c[7:] for c in correction_columns if c.startswith('factor_') and c != 'factor_all']
            correction_factors_dict = {c: f'"factor_all" * "factor_{c}"' if c in correction_enduses else '"factor_all"'
                                       for c in enduses}

            if "simulation_weight_correction_factor" in correction_columns:
                total_weight += ' * COALESCE("simulation_weight_correction_factor", 1)'

            # COALESCE for the factors table is used to allow partial left join when sparse correction table is used
            enduse_cols = 'sum(COALESCE("factor_all", 1)) as correction_factor_all'

            # If the individual enduses have changed, we will need to adjust the total_site_electricity_kwh
            # TODO: Need to adjust total columns for other fuel types too if correction is applied to other fuel enduses
            if 'total_site_electricity_kwh' in enduses:
                enduses.remove('total_site_electricity_kwh')
                # we need to add (factor_enduse - 1) portion of all the corrected enduses to total_site_electricity_kwh
                enduse_cols += ', sum(("total_site_electricity_kwh"'

                enduse_corrected_fractions = [f'COALESCE("factor_{c}" - 1, 0) * {C(c)}' for c in correction_enduses
                                              if c.startswith('electricity')]
                if enduse_corrected_fractions:
                    enduse_cols += " + " + " + ".join(enduse_corrected_fractions)

                enduse_cols += f') * {total_weight} * COALESCE("factor_all", 1) / {n_units_col}) as' \
                    f' total_site_electricity_kwh'

            additional_enduses = [f"sum({C(c)} * {total_weight} * COALESCE({C(correction_factors_dict[c])}, 1) / "
                                  f"{n_units_col}) as {C(self.simple_label(c))}" for c in enduses
                                  if not c.startswith("schedules_")]
            if additional_enduses:
                enduse_cols += ', ' + ', '.join(additional_enduses)

        else:
            enduse_cols = ', '.join([f"sum({C(c)} * {total_weight} / {n_units_col}) as {C(self.simple_label(c))}"
                                     for c in enduses if not c.startswith("schedules_")])

        schedule_enduses = [f"sum({C(c)} * {total_weight})"
                            f" as {C(self.simple_label(c))}" for c in enduses if c.startswith("schedules_")]
        if schedule_enduses:
            enduse_cols += ', ' + ', '.join(schedule_enduses)

        if group_by is None:
            group_by = []

        group_by_cols = ", ".join([f'{C(g)}' for g in group_by])
        grouping_metrics_cols = "sum(1) as raw_count,"
        if self.timestamp_column_name not in group_by:
            # The aggregation is done across time so we should compensate unit count by dividing by the total number
            # of distinct timestamps per unit
            timesteps_per_unit = self._get_simulation_timesteps_count()
            grouping_metrics_cols += f" sum({total_weight})/{timesteps_per_unit} as scaled_unit_count"
        else:
            grouping_metrics_cols += f" sum({total_weight}) as scaled_unit_count"

        if group_by_cols:
            select_cols = group_by_cols + ", " + grouping_metrics_cols
        else:
            select_cols = grouping_metrics_cols
        if enduse_cols:
            select_cols += ", " + enduse_cols

        query = f"select {select_cols} from {C(self.ts_table_name)}"

        join_clause = f''' join {C(self.baseline_table_name)} on {C(self.ts_table_name)}."building_id" = '''\
                      f''' {C(self.baseline_table_name)}."building_id" '''
        for new_table_name, baseline_column_name, new_column_name in join_list:
            join_clause += f''' join {C(new_table_name)} on '''\
                           f''' {C(self.baseline_table_name)}.{C(baseline_column_name)} ='''\
                           f''' {C(new_table_name)}.{C(new_column_name)}'''

        if correction_factors_table:
            baseline_columns = self.get_cols(table='baseline')
            on_clauses = [f" {C(self.baseline_table_name)}.{C(c)} = {C(correction_factors_table)}.{C(c)} "
                          for c in baseline_columns if c in correction_columns]

            # https://prestodb.io/docs/0.217/functions/datetime.html#convenience-extraction-functions
            extraction_functions = ['day_of_week', 'day_of_year', 'day_of_month', 'hour', 'minute', 'month']
            on_clauses += [f" {c.lower()}({C(self.ts_table_name)}.{C(self.timestamp_column_name)}) = "
                           f" {C(correction_factors_table)}.{C(c)} "
                           for c in correction_columns if c.lower() in extraction_functions]
            if not on_clauses:
                raise ValueError("No column in the correction table matches any column in baseline table.")

            join_clause += f""" left join {C(correction_factors_table)} on {" and ".join(on_clauses)}"""

        query += join_clause

        query += self._get_restrict_string(restrict)

        if group_by:
            query += " group by " + ", ".join([f'''{C(g)}''' for g in group_by])
        if order_by:
            query += " order by " + ", ".join([f'''{C(o)}''' for o in order_by])

        if limit is not None:
            query += f" limit {limit}"

        if get_query_only:
            return query

        t = time.time()
        res = self.execute(query, run_async=True)
        if run_async:
            return res
        else:
            time.sleep(1)
            while time.time() - t < 30 * 60:
                stat = self.py_thena.get_query_status(res)
                if stat.upper() == 'SUCCEEDED':
                    result = self.get_query_result(res)
                    return result
                elif stat.upper() == 'FAILED':
                    error = self.get_query_error(res)
                    raise Exception(error)
                else:
                    logger.info(f"Query status is {stat}")
                    time.sleep(30)

            raise Exception(f'Query failed {self.py_thena.get_query_status(res)}')

    def _get_simulation_timesteps_count(self):
        C = self.make_column_string
        if "simulation_timesteps_count" in self.cache:
            if self.db_name + "/" + self.ts_table_name in self.cache["simulation_timesteps_count"]:
                return self.cache['simulation_timesteps_count'][self.db_name + "/" + self.ts_table_name]
        else:
            self.cache['simulation_timesteps_count'] = {}
        # find the simulation time interval
        sim_timesteps_count = self.execute(f'SELECT "building_id", sum(1) as count'
                                           f' from {C(self.ts_table_name)} group by 1')
        bld0_step_count = sim_timesteps_count['count'].iloc[0]
        if not sum(sim_timesteps_count['count'] == bld0_step_count) == len(sim_timesteps_count):
            logger.warning("Not all building has the same number of timestamps. This can cause wrong scaled_units_count"
                           " and other problems.")

        self.cache['simulation_timesteps_count'][self.db_name + "/" + self.ts_table_name] = bld0_step_count
        return bld0_step_count

    def _get_simulation_info(self):
        C = self.make_column_string
        # find the simulation time interval
        two_times = self.execute(f"SELECT distinct({C(self.timestamp_column_name)}) as time"
                                 f" from {C(self.ts_table_name)} limit 2")
        time1 = parser.parse(two_times['time'].iloc[0])
        time2 = parser.parse(two_times['time'].iloc[1])
        sim_year = time1.year
        sim_interval_seconds = (time2 - time1).total_seconds()
        return sim_year, sim_interval_seconds

    def get_building_average_kws_at(self,
                                    at_hour,
                                    at_days,
                                    enduses=None,
                                    custom_sample_weight=None,
                                    get_query_only=False):
        """
        Aggregates the timeseries result on select enduses, for the given days and hours.
        If all of the hour(s) fall exactly on the simulation timestamps, the aggregation is done by averaging the kW at
        those time stamps. If any of the hour(s) fall in between timestamps, then the following process is followed:
            i. The average kWs is calculated for timestamps specified by the hour, or just after it. Call it upper_kw
            ii. The average kWs is calculated for timestamps specified by the hour, or just before it. Call it lower_kw
            iii. Return the interpolation between upper_kw and lower_kw based on the average location of the hour(s)
                 between the upper and lower timestamps.

        Check the argument description below to learn about additional features and options.
        Args:
            at_hour: the hour(s) at which the average kWs of buildings need to be calculated at. It can either be a
                     single number if the hour is same for all days, or a list of numbers if the kW needs to be
                     calculated for different hours for different days.

            at_days: The list of days (of year) for which the average kW is to be calculated for.

            enduses: The list of enduses for which to calculate the average kWs

            custom_sample_weight: If you want to override the build_existing_model.sample_weight, provide a custom
                                  sample weight.
            get_query_only: Skips submitting the query to Athena and just returns the query strings. Useful for batch
                            submitting multiple queries or debugging.

        Returns:
                If get_query_only is True, returns two queries that gets the KW at two timestamps that are to immediate
                    left and right of the the supplied hour.
                If get_query_only is False, returns the average KW of each building at the given hour(s) across the
                supplied days.

        """

        C = self.make_column_string

        if not enduses:
            enduses = self.get_cols(table='timeseries', fuel_type='electricity')

        if isinstance(at_hour, list):
            if len(at_hour) != len(at_days) or len(at_hour) == 0:
                raise ValueError("The length of at_hour list should be the same as length of at_days list and"
                                 " not be empty")
        elif isinstance(at_hour, float) or isinstance(at_hour, int):
            at_hour = [at_hour] * len(at_days)
        else:
            raise ValueError("At hour should be a list or a number")

        sim_year, sim_interval_seconds = self._get_simulation_info()
        kw_factor = 3600.0 / sim_interval_seconds

        if custom_sample_weight:
            sample_weight = str(custom_sample_weight)
        else:
            sample_weight = self.sample_weight_column

        total_weight = f'{sample_weight}'
        n_units_col = C("build_existing_model.units_represented")
        enduse_cols = ', '.join([f"avg({C(c)} * {total_weight} * {kw_factor:.3f} / {n_units_col}) as"
                                 f" {C(self.simple_label(c))}" for c in enduses])
        grouping_metrics_cols = f" sum(1) as raw_count, sum({total_weight}) as scaled_unit_count"

        def get_upper_timestamps(day, hour):
            new_dt = datetime.datetime(year=sim_year, month=1, day=1)

            if round(hour * 3600 % sim_interval_seconds, 2) == 0:
                # if the hour falls exactly on the simulation timestamp, use the same timestamp
                # for both lower and upper
                add = 0
            else:
                add = 1

            upper_dt = new_dt + datetime.timedelta(days=day, seconds=sim_interval_seconds *
                                                   (int(hour * 3600 / sim_interval_seconds) + add))
            if upper_dt.year > sim_year:
                upper_dt = new_dt + datetime.timedelta(days=day, seconds=sim_interval_seconds *
                                                       (int(hour * 3600 / sim_interval_seconds)))
            return upper_dt

        def get_lower_timestamps(day, hour):
            new_dt = datetime.datetime(year=sim_year, month=1, day=1)
            lower_dt = new_dt + datetime.timedelta(days=day, seconds=sim_interval_seconds * int(hour * 3600 /
                                                                                                sim_interval_seconds))
            return lower_dt

        # check if the supplied hours fall exactly on the simulation timestamps
        exact_times = np.all([round(h * 3600 % sim_interval_seconds, 2) == 0 for h in at_hour])
        lower_timestamps = [get_lower_timestamps(d - 1, h) for d, h in zip(at_days, at_hour)]
        upper_timestamps = [get_upper_timestamps(d - 1, h) for d, h in zip(at_days, at_hour)]

        query_str = f'''select {C(self.ts_table_name)}."building_id" as building_id, '''
        query_str += grouping_metrics_cols
        query_str += f''', {enduse_cols} from {C(self.ts_table_name)}'''

        join_clause = f''' join {C(self.baseline_table_name)} on {C(self.ts_table_name)}."building_id" =\
                                 {C(self.baseline_table_name)}."building_id" '''
        query_str += join_clause

        lower_val_query = query_str + self._get_restrict_string([(self.timestamp_column_name, lower_timestamps)])
        upper_val_query = query_str + self._get_restrict_string([(self.timestamp_column_name, upper_timestamps)])
        lower_val_query += " group by 1 order by 1"
        upper_val_query += " group by 1 order by 1"

        if exact_times:
            # only one query is sufficient if the hours fall in exact timestamps
            queries = [lower_val_query]
        else:
            queries = [lower_val_query, upper_val_query]

        if get_query_only:
            return queries

        batch_id = self.submit_batch_query(queries)
        if exact_times:
            vals, = self.get_batch_query_result(batch_id, combine=False)
            return vals.drop(columns=['query_id'])
        else:
            lower_vals, upper_vals = self.get_batch_query_result(batch_id, combine=False)
            avg_upper_weight = np.mean([min_of_hour / sim_interval_seconds for hour in at_hour if
                                        (min_of_hour := hour * 3600 % sim_interval_seconds)])
            avg_lower_weight = 1 - avg_upper_weight
            # modify the lower vals to make it weighted average of upper and lower vals
            lower_vals[enduses] = lower_vals[enduses] * avg_lower_weight + upper_vals[enduses] * avg_upper_weight
            return lower_vals.drop(columns=['query_id'])
