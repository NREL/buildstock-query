"""
# ResStockAthena
- - - - - - - - -
A class to run AWS Athena queries to get various data from a ResStock run. All queries and aggregation that can be
common accross different ResStock projects should be implemented in this class. For queries that are project specific, a
new class can be created by inheriting ResStockAthena and adding in the project specific logic and queries.

:author: Rajendra.Adhikari@nrel.gov
"""

import re
from pyathena.connection import Connection
from pyathena.sqlalchemy_athena import AthenaDialect
import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
import dask.dataframe as dd
from pyathena.pandas.async_cursor import AsyncPandasCursor
from pyathena.pandas.cursor import PandasCursor
import os
import boto3
from typing import List, Tuple, Union
import time
import logging
from threading import Thread
from botocore.exceptions import ClientError
import pandas as pd
import datetime
import numpy as np
from collections import OrderedDict
import types
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomCompiler(AthenaDialect().statement_compiler):
    def render_literal_value(self, value, type_):
        if isinstance(value, (datetime.datetime)):
            return "timestamp '%s'" % str(value).replace("'", "''")
        return super(CustomCompiler, self).render_literal_value(value, type_)


class DataExistsException(Exception):
    def __init__(self, message, existing_data=None):
        super(DataExistsException, self).__init__(message)
        self.existing_data = existing_data


class ResStockAthena:
    def __init__(self, workgroup: str,
                 db_name: str,
                 buildstock_type: str = None,
                 table_name: Union[str, Tuple[str, str]] = None,
                 timestamp_column_name: str = 'time',
                 building_id_column_name: str = 'building_id',
                 sample_weight: str = "build_existing_model.sample_weight",
                 region_name: str = 'us-west-2',
                 execution_history=None
                 ) -> None:
        """
        A class to run common Athena queries for ResStock runs and download results as pandas dataFrame
        Args:
            db_name: The athena database name
            buildstock_type: 'resstock' or 'comstock' runs
            table_name: If a single string is provided, say, 'mfm_run', then it must correspond to two tables in athena
                        named mfm_run_baseline and mfm_run_timeseries. Or, two strings can be provided as a tuple, (such
                        as 'mfm_run_2_baseline', 'mfm_run5_timeseries') and they must be a baseline table and a
                        timeseries table.
            timestamp_column_name: The column name for the time column. Defaults to 'time'
            sample_weight: The column name to be used to get the sample weight. Defaults to
                           build_existing_model.sample_weight. Pass floats/integer to use constant sample weight.
            region_name: The AWS region where the database exists. Defaults to 'us-west-2'.
            execution_history: A temporary files to record which execution is run by the user, to help stop them. Will
                    use .execution_history if not supplied.
        """
        self.workgroup = workgroup
        self.buildstock_type = buildstock_type
        self.s3 = boto3.client('s3')
        self.aws_athena = boto3.client('athena', region_name=region_name)
        self.aws_glue = boto3.client('glue', region_name=region_name)
        self.conn = Connection(work_group=workgroup, region_name=region_name,
                               cursor_class=PandasCursor, schema_name=db_name)
        self.async_conn = Connection(work_group=workgroup, region_name=region_name,
                                     cursor_class=AsyncPandasCursor, schema_name=db_name, )
        self.db_name = db_name
        self.region_name = region_name

        self.tables = OrderedDict()
        self.join_list = {}

        self.batch_query_status_map = {}
        self.batch_query_id = 0

        self.timestamp_column_name = timestamp_column_name
        self.building_id_column_name = building_id_column_name
        self.sample_weight = sample_weight
        self.table_name = table_name
        self._initialize_tables()
        self._initialize_book_keeping(execution_history)

    def _initialize_tables(self):
        self.bs_table, self.ts_table = self._get_ts_bs_tables(self.table_name)

        self.timestamp_column = self.ts_table.c[self.timestamp_column_name]
        self.ts_bldgid_column = self.ts_table.c[self.building_id_column_name]
        self.bs_bldgid_column = self.bs_table.c[self.building_id_column_name]
        self.sample_wt = self._get_sample_weight(self.sample_weight)

    def _get_sample_weight(self, sample_weight):
        if not sample_weight:
            return sa.Integer(1)
        elif isinstance(sample_weight, str):
            return self.get_col(sample_weight)
        elif isinstance(sample_weight, int):
            return sa.literal(sample_weight)
        elif isinstance(sample_weight, float):
            return sa.literal(sample_weight)

    def get_tbl(self, table_name):
        return self.tables.setdefault(table_name, sa.Table(table_name, self.meta, autoload_with=self.engine))

    def get_col(self, column_name):
        matches = []
        if isinstance(column_name, sa.Column):
            return column_name  # already a col

        for name, table in self.tables.items():
            if column_name in table.columns:
                matches.append(table)
        if not matches:
            raise ValueError(f"Column {column_name} not found in any tables {[t.name for t in self.tables.values()]}")
        if len(matches) > 1:
            logger.warning(
                f"Column {column_name} found in multiple tables {[t.name for t in matches]}. Using {matches[0].name}")
        return matches[0].c[column_name]

    def _get_ts_bs_tables(self, table_name):
        self.engine = self._create_athena_engine(region_name=self.region_name, database=self.db_name,
                                                 workgroup=self.workgroup)
        self.meta = sa.MetaData(bind=self.engine)
        if isinstance(table_name, str):
            baseline_table_name = self.get_tbl(f'{table_name}_baseline')
            ts_table_name = self.get_tbl(f'{table_name}_timeseries')
        elif isinstance(table_name, tuple):
            baseline_table_name = self.get_tbl(f'{table_name[0]}')
            ts_table_name = self.get_tbl(f'{table_name[1]}')
        else:
            baseline_table_name = None
            ts_table_name = None
        return baseline_table_name, ts_table_name

    def _initialize_book_keeping(self, execution_history):
        if not execution_history:
            self.execution_history_file = '.execution_history'
        else:
            self.execution_history_file = execution_history

        self.execution_cost = {'GB': 0, 'Dollars': 0}  # Tracks the cost of current session. Only used for Athena query
        self.seen_execution_ids = set()  # set to prevent double counting same execution id
        self.cache = {}  # To store small but frequently queried result, such as total number of timesteps

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

    def _create_athena_engine(self, region_name: str, database: str, workgroup: str) -> sa.engine.Engine:
        connect_args = {"cursor_class": PandasCursor, "work_group": workgroup}
        engine = sa.create_engine(
            f"awsathena+rest://:@athena.{region_name}.amazonaws.com:443/{database}", connect_args=connect_args
        )
        return engine

    def delete_table(self, table_name):
        """
        Function to delete athena table.
        :param table_name: Athena table name
        :return:
        """
        delete_table_query = f"""DROP TABLE {self.db_name}.{table_name};"""
        result, reason = self.execute_raw(delete_table_query)
        if result.upper() == "SUCCEEDED":
            return "SUCCEEDED"
        else:
            raise Exception(f"Deleting it failed. Reason: {reason}")

    def add_table(self, table_name, table_df, s3_bucket, s3_prefix, override=False):
        """
        Function to add a table in s3.
        :param table_name: The name of the table
        :param table_df: The pandas dataframe to use as table data
        :param s3_bucket: s3 bucket name
        :param s3_prefix: s3 prefix to save the table to.
        :param override: Whether to override eixsting table.
        :return:
        """
        s3_location = s3_bucket + '/' + s3_prefix
        s3_data = self.s3.list_objects(Bucket=s3_bucket, Prefix=f'{s3_prefix}/{table_name}')

        if 'Contents' in s3_data and override is False:
            raise DataExistsException("Table already exists", f's3://{s3_location}/{table_name}/{table_name}.csv')
        else:
            if 'Contents' in s3_data:
                existing_objects = [{'Key': el['Key']} for el in s3_data['Contents']]
                print(f"The following existing objects is being delete and replaced: {existing_objects}")
                print(f"Saving s3://{s3_location}/{table_name}/{table_name}.parquet)")
                self.s3.delete_objects(Bucket=s3_bucket, Delete={"Objects": existing_objects})
            print(f"Saving factors to s3 in s3://{s3_location}/{table_name}/{table_name}.parquet")
            table_df.to_parquet(f's3://{s3_location}/{table_name}/{table_name}.parquet', index=False)
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
        CREATE EXTERNAL TABLE {self.db_name}.{table_name} ({column_formats})
        STORED AS PARQUET
        LOCATION 's3://{s3_location}/{table_name}/'
        TBLPROPERTIES ('has_encrypted_data'='false');
        """

        print("Running create table query.")
        result, reason = self.execute_raw(table_create_query)
        if result.lower() == "failed" and 'alreadyexists' in reason.lower():
            if not override:
                existing_data = pd.read_csv(f's3://{s3_location}/{table_name}/{table_name}.csv')
                raise DataExistsException("Table already exists", existing_data)
            print(f"There was existing table {table_name} in Athena which was deleted and recreated.")
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
        """
        Directly executes the supplied query in Athena.
        :param query:
        :param db:
        :param run_async:
        :return:
        """
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

    def log_execution_cost(self, execution_id):
        if not execution_id.startswith('A'):
            # Can't log cost for Spark query
            return
        res = self.aws_athena.get_query_execution(QueryExecutionId=execution_id[1:])
        scanned_GB = res['QueryExecution']['Statistics']['DataScannedInBytes'] / 1e9
        cost = scanned_GB * 5 / 1e3  # 5$ per TB scanned
        if execution_id not in self.seen_execution_ids:
            self.execution_cost['Dollars'] += cost
            self.execution_cost['GB'] += scanned_GB
            self.seen_execution_ids.add(execution_id)

        logger.info(f"{execution_id} cost {scanned_GB:.1f}GB (${cost:.1f}). Session total:"
                    f" {self.execution_cost['GB']:.1f} GB (${self.execution_cost['Dollars']:.1f})")

    def _compile(self, query):
        compiled_query = CustomCompiler(AthenaDialect(), query).process(query, literal_binds=True)
        return compiled_query

    def execute(self, query, run_async=False):
        """
        Executes a query
        Args:
            query: The SQL query to run in Athena
            run_async: Whether to wait until the query completes (run_async=False) or return immediately
            (run_async=True).

        Returns:
            if run_async is False, returns the results dataframe.
            if run_async is  True, returns the query_execution_id, futures
        """
        if not isinstance(query, str):
            query = self._compile(query)

        if run_async:
            # in case of asynchronous run, you get the execution id abd futures object
            exe_id, result_future = self.async_conn.cursor().execute(query)
            result_future.as_pandas = types.MethodType(lambda x: x.result().as_pandas(), result_future)
            self._save_execution_id(exe_id)
            return exe_id, result_future
        else:
            # in case of synchronous run, just return the dataFrame
            df = self.conn.cursor().execute(query).as_pandas()
            return df

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

    def get_failed_queries(self, batch_id):
        stats = self.batch_query_status_map.get(batch_id, None)
        failed_query_ids, failed_queries = [], []
        if stats:
            for i, exe_id in enumerate(stats['submitted_execution_ids']):
                completion_stat = self.get_query_status(exe_id)
                if completion_stat in ['FAILED', 'CANCELLED']:
                    failed_query_ids.append(exe_id)
                    failed_queries.append(stats['submitted_queries'][i])
        return failed_query_ids, failed_queries

    def print_failed_query_errors(self, batch_id):
        failed_ids, failed_queries = self.get_failed_queries(batch_id)
        for exe_id, query in zip(failed_ids, failed_queries):
            print(f"Query id: {exe_id}. \n Query string: {query}. Query Ended with: {self.get_query_status(exe_id)}"
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
                    # for example: QUEUED
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

    def wait_for_batch_query(self, batch_id):
        while True:
            last_time = time.time()
            last_report = None
            report = self.get_batch_query_report(batch_id)
            if time.time() - last_time > 60 or last_report is None or report != last_report:
                logger.info(report)
                last_report = report
                last_time = time.time()
            if report['Pending'] == 0 and report['Running'] == 0:
                break
            time.sleep(20)

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
        if no_block and self.did_batch_query_complete(batch_id) is False:
            raise Exception('Batch query not completed yet.')

        self.wait_for_batch_query(batch_id)
        logger.info("Batch query completed. ")
        report = self.get_batch_query_report(batch_id)
        query_exe_ids = self.batch_query_status_map[batch_id]['submitted_execution_ids']
        query_futures = self.batch_query_status_map[batch_id]['queries_futures']
        if report['Failed'] > 0:
            logger.warning(f"{report['Failed']} queries failed. Redoing them")
            failed_ids, failed_queries = self.get_failed_queries(batch_id)
            new_batch_id = self.submit_batch_query(failed_queries)
            new_exe_ids = self.batch_query_status_map[new_batch_id]['submitted_execution_ids']

            self.wait_for_batch_query(new_batch_id)
            new_exe_ids_map = {entry[0]: entry[1] for entry in zip(failed_ids, new_exe_ids)}

            new_report = self.get_batch_query_report(new_batch_id)
            if new_report['Failed'] > 0:
                self.print_failed_query_errors(new_batch_id)
                raise Exception("Queries failed again. Sorry!")
            logger.info("The queries succeeded this time. Gathering all the results.")
            # replace the old failed exe_ids with new successful exe_ids
            for indx, old_exe_id in enumerate(query_exe_ids):
                query_exe_ids[indx] = new_exe_ids_map.get(old_exe_id, old_exe_id)

        if len(query_exe_ids) == 0:
            raise ValueError("No query was submitted successfully")
        res_df_array = []
        for index, exe_id in enumerate(query_exe_ids):
            df = query_futures[index].result().as_pandas()
            df['query_id'] = index
            logger.info(f"Got result from Query [{index}] ({exe_id})")
            res_df_array.append(df)
        if combine:
            logger.info("Concatenating the results.")
            return pd.concat(res_df_array)
        else:
            return res_df_array

    def submit_batch_query(self, queries: List[str]):
        """
        Submit multiple related queries
        Args:
            queries: List of queries to submit. Setting `get_query_only` flag while making calls to aggregation
                    functions is easiest way to obtain queries.
        Returns:
            An integer representing the batch_query id. The id can be used with other batch_query functions.
        """
        queries = list(queries)
        to_submit_ids = list(range(len(queries)))
        id_list = list(to_submit_ids)  # make a copy
        submitted_ids = []
        submitted_execution_ids = []
        submitted_queries = []
        queries_futures = []
        self.batch_query_id += 1
        batch_query_id = self.batch_query_id
        self.batch_query_status_map[batch_query_id] = {'to_submit_ids': to_submit_ids,
                                                       'all_ids': list(id_list),
                                                       'submitted_ids': submitted_ids,
                                                       'submitted_execution_ids': submitted_execution_ids,
                                                       'submitted_queries': submitted_queries,
                                                       'queries_futures': queries_futures
                                                       }

        def run_queries():
            while to_submit_ids:
                current_id = to_submit_ids[0]  # get the first one
                current_query = queries[0]
                try:
                    execution_id, future = self.execute(current_query, run_async=True)
                    logger.info(f"Submitted queries[{current_id}] ({execution_id})")
                    to_submit_ids.pop(0)  # if query queued successfully, remove it from the list
                    queries.pop(0)
                    submitted_ids.append(current_id)
                    submitted_execution_ids.append(execution_id)
                    submitted_queries.append(current_query)
                    queries_futures.append(future)
                except ClientError as e:
                    if e.response['Error']['Code'] == 'TooManyRequestsException':
                        logger.info("Athena complained about too many requests. Waiting for a minute.")
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
        return self.get_athena_query_result(execution_id=query_id)

    def get_athena_query_result(self, execution_id, timeout_minutes=60):
        t = time.time()
        while time.time() - t < timeout_minutes * 60:
            stat = self.get_query_status(execution_id)
            if stat.upper() == 'SUCCEEDED':
                result = self.get_result_from_s3(execution_id)
                self.log_execution_cost(execution_id)
                return result
            elif stat.upper() == 'FAILED':
                error = self.get_query_error(execution_id)
                raise Exception(error)
            else:
                logger.info(f"Query status is {stat}")
                time.sleep(30)

        raise Exception(f'Query timed-out. {self.get_query_status(execution_id)}')

    def get_result_from_s3(self, query_execution_id):
        query_status = self.get_query_status(query_execution_id)
        if query_status == 'SUCCEEDED':
            path = self.get_query_output_location(query_execution_id)
            df = dd.read_csv(path).compute()[0]
            return df
        # If failed, return error message
        elif query_status == 'FAILED':
            raise Exception(self.get_query_error(query_execution_id))
        elif query_status in ['RUNNING', 'QUEUED', 'PENDING']:
            raise Exception(f"Query still {query_status}")
        else:
            raise Exception(f"Query has unkown status {query_status}")

    def get_query_output_location(self, query_id):
        stat = self.aws_athena.get_query_execution(QueryExecutionId=query_id)
        output_path = stat['QueryExecution']['ResultConfiguration']['OutputLocation']
        return output_path

    def get_query_status(self, query_id):
        stat = self.aws_athena.get_query_execution(QueryExecutionId=query_id)
        return stat['QueryExecution']['Status']['State']

    def get_query_error(self, query_id):
        stat = self.aws_athena.get_query_execution(QueryExecutionId=query_id)
        return stat['QueryExecution']['Status']['StateChangeReason']

    def get_all_running_queries(self):
        """
        Gives the list of all running queries (for this instance)

        Return:
            List of query execution ids of all the queries that are currently running in Athena.
        """
        exe_ids = self.aws_athena.list_query_executions(WorkGroup=self.workgroup)['QueryExecutionIds']
        exe_ids = [exe_id for exe_id in exe_ids]

        running_ids = [i for i in exe_ids if i in self.execution_ids_history and
                       self.get_query_status(i) == "RUNNING"]
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
            self.stop_query(execution_id=i)

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
            cols = self.ts_table.columns
            if fuel_type:
                cols = [c for c in cols if c.name not in [self.ts_bldgid_column.name, self.timestamp_column.name]]
                cols = [c for c in cols if fuel_type in c.name]
            return cols
        elif table in ['baseline', 'bs']:
            cols = self.bs_table.columns
            if fuel_type:
                cols = [c for c in cols if 'simulation_output_report' in c.name]
                cols = [c for c in cols if fuel_type in c.name]
            return cols
        else:
            tbl = self.get_tbl(table)
            return tbl.columns

    def get_distinct_vals(self, column: str, table_name: str = None, get_query_only: bool = False):
        table_name = self.bs_table.name if table_name is None else table_name
        tbl = self.get_tbl(table_name)
        query = sa.select(tbl.c[column]).distinct()
        if get_query_only:
            return self._compile(query)

        r = self.execute(query, run_async=False)
        return r[column]

    def get_distinct_count(self, column: str, table_name: str = None, weight_column: str = None,
                           get_query_only: bool = False):
        if table_name is None:
            tbl = self.bs_table
        else:
            tbl = self.get_tbl(table_name)

        query = sa.select([tbl.c[column], safunc.sum(1).label("raw_count"),
                           safunc.sum(self.sample_wt).label("weighted_count")])
        query = query.group_by(tbl.c[column]).order_by(tbl.c[column])
        if get_query_only:
            return self._compile(query)

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
        query = sa.select(safunc.count().label("count"))

        restrict = list(restrict)
        restrict.insert(0, ('completed_status', ['Success']))
        query = self._add_restrict(query, restrict)
        if get_query_only:
            return self._compile(query)

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
        query = sa.select(['*']).select_from(self.bs_table)
        query = self._add_restrict(query, restrict)

        if get_query_only:
            return self._compile(query)

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
        query = sa.select(self.bs_bldgid_column)
        query = self._add_restrict(query, restrict)
        if get_query_only:
            return self._compile(query)
        res = self.execute(query)
        return res

    @staticmethod
    def simple_label(label):
        if label.startswith('report_simulation_output'):
            return label.split('.')[1]
        else:
            return label

    def _add_restrict(self, query, restrict):
        if not restrict:
            return query

        where_clauses = []
        for col, criteria in restrict:
            if isinstance(criteria, list) or isinstance(criteria, tuple):
                if len(criteria) > 1:
                    where_clauses.append(self.get_col(col).in_(criteria))
                    continue
                else:
                    criteria = criteria[0]
            where_clauses.append(self.get_col(col) == criteria)
        query = query.where(*where_clauses)
        return query

    def _add_join(self, query, join_list):
        for new_table_name, baseline_column_name, new_column_name in join_list:
            new_tbl = self.get_tbl(new_table_name)
            query = query.join(new_tbl, self.bs_table.c[baseline_column_name]
                               == new_tbl.c[new_column_name])
        return query

    def _add_group_by(self, query, group_by):
        if group_by:
            group_by_cols = [self.get_col(g[0]) if isinstance(
                g, tuple) else self.get_col(g) for g in group_by]
            query = query.group_by(*group_by_cols)
        return query

    def _add_order_by(self, query, order_by):
        if order_by:
            order_by_cols = [self.get_col(g[0]) if isinstance(
                g, tuple) else self.get_col(g) for g in order_by]
            query = query.order_by(*order_by_cols)
        return query

    def _get_enduse_cols(self, enduses, table='baseline'):
        tbl = self.bs_table if table == 'baseline' else self.ts_table
        if not enduses:
            enduse_cols = self.get_cols(table=table, fuel_type='electricity')
        else:
            enduse_cols = [tbl.c[e] for e in enduses if not e.startswith('schedule_')]
        # enduse_cols = [sa.column(e.name) for e in enduse_cols]
        return enduse_cols

    def _get_schedule_cols(self, enduses):
        if not enduses:
            sch_cols = [c for c in self.ts_table.columns if c.name.startswith('schedule_')]
        else:
            sch_cols = [c for c in enduses if c.name.startswith('schedule_')]
        return sch_cols

    def _get_weight(self, weights):
        total_weight = self.sample_wt
        for weight_col in weights:
            if isinstance(weight_col, tuple):
                tbl = self.get_tbl(weight_col[1])
                total_weight *= tbl.c[weight_col[0]]
            else:
                total_weight *= self.get_col(weight_col)
        return total_weight

    def aggregate_annual(self,
                         enduses: List[str] = None,
                         group_by: List[str] = None,
                         order_by: List[str] = None,
                         join_list: List[Tuple[str, str, str]] = [],
                         weights: List[Tuple] = [],
                         restrict: List[Tuple[str, List]] = [],
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

            weights: The additional columns to use as weight. The "build_existing_model.sample_weight" is already used.
                     It is specified as either list of string or list of tuples. When only string is used, the string
                     is the column name, when tuple is passed, the second element is the table name.

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

        [self.get_tbl(jl[0]) for jl in join_list]  # ingress all tables in join list
        enduses = self._get_enduse_cols(enduses)
        total_weight = self._get_weight(weights)
        enduse_selection = [safunc.sum(enduse * total_weight).label(self.simple_label(enduse.name))
                            for enduse in enduses]
        grouping_metrics_selction = [safunc.sum(1).label("raw_count"),
                                     safunc.sum(total_weight).label("scaled_unit_count")]

        if not group_by:
            query = sa.select(grouping_metrics_selction + enduse_selection)
        else:
            group_by_selection = [self.get_col(g[0]).label(g[1]) if isinstance(
                g, tuple) else self.get_col(g) for g in group_by]
            query = sa.select(group_by_selection + grouping_metrics_selction + enduse_selection)
        # jj = self.bs_table.join(self.ts_table, self.ts_table.c['building_id']==self.bs_table.c['building_id'])
        # self._compile(query.select_from(jj))
        # query = query.select_from(self.bs_table)
        query = self._add_join(query, join_list)
        query = self._add_restrict(query, restrict)
        query = self._add_group_by(query, group_by)
        query = self._add_order_by(query, order_by)

        if get_query_only:
            return self._compile(query)

        return self.execute(query, run_async=run_async)

    def _clean_group_by(self, group_by):
        """

        :param group_by: The group_by list
        :return: cleaned version of group_by

        Sometimes, it is necessary to include the table name in the group_by column. For example, a group_by could be
        ['time', '"res_national_53_2018_baseline"."build_existing_model.state"']. This is necessary if the another table
        (such as correction factors table) that has the same column ("build_existing_model.state") as the baseline
        table. However, the query result will not include the table name in columns, so it is necessary to transform
        the group_by to a cleaner version (['time', 'build_existing_model.state']).

        Othertimes, quotes are used in group_by columns, such as ['"time"'], but the query result will not contain the
        quote so it is necessary to remove the quote.

        Some other time, a group_by column is specified as a tuple of column and a as name. For example, group_by can
        contain [('month(time)', 'MOY')], in this case, we want to convert it into just 'MOY' since that is what will be
        present in the returned query.

        """
        new_group_by = []
        for col in group_by:
            if isinstance(col, tuple):
                new_group_by.append(col[1])
                continue

            match = re.search(r'"[\w\.]*"\."([\w\.]*)"', col)
            if not match:
                match = re.search(r'"([\w\.]*)"', col)

            if match:
                new_group_by.append(match.group(1))
            else:
                new_group_by.append(col)
        return new_group_by

    def aggregate_timeseries_light(self,
                                   enduses: List[str] = [],
                                   group_by: List[str] = [],
                                   order_by: List[str] = [],
                                   join_list: List[Tuple[str, str, str]] = [],
                                   weights: List[str] = [],
                                   restrict: List[Tuple[str, List]] = [],
                                   run_async: bool = False,
                                   get_query_only: bool = False,
                                   limit=None
                                   ):
        """
        Lighter version of aggregate_timeseries where each enduse is submitted as a separate query to be light on
        Athena. For information on the input parameters, check the documentation on aggregate_timeseries.
        """

        if run_async:
            raise ValueError("Async run is not available for aggregate_timeseries_light since it needs to combine"
                             "the result after the query finishes.")

        enduses = self._get_enduse_cols(enduses, table='timeseries')
        print(enduses)
        batch_queries_to_submit = []
        for indx, enduse in enumerate(enduses):
            query = self.aggregate_timeseries(enduses=[enduse.name],
                                              group_by=group_by,
                                              order_by=order_by,
                                              join_list=join_list,
                                              weights=weights,
                                              restrict=restrict,
                                              get_query_only=True,
                                              limit=limit)
            batch_queries_to_submit.append(query)

        if get_query_only:
            logger.warning("Not recommended to use get_query_only and split_enduses used together."
                           " The results from the queries cannot be directly combined to get the desired result."
                           " There are further processing done in the function. The queries should be used for"
                           " information or debugging purpose only. Use get_query_only=False to get proper result.")
            return batch_queries_to_submit

        batch_query_id = self.submit_batch_query(batch_queries_to_submit)

        result_dfs = self.get_batch_query_result(batch_id=batch_query_id, combine=False)
        logger.info("Joining the individual enduses result into a single DataFrame")
        group_by = self._clean_group_by(group_by)
        for res in result_dfs:
            res.set_index(group_by, inplace=True)
        self.result_dfs = result_dfs
        joined_enduses_df = result_dfs[0].drop(columns=['query_id'])
        for enduse, res in list(zip(enduses, result_dfs))[1:]:
            joined_enduses_df = joined_enduses_df.join(res[[enduse.name]])

        logger.info("Joining Completed.")
        return joined_enduses_df.reset_index()

    def aggregate_timeseries(self,
                             enduses: List[str] = [],
                             group_by: List[str] = [],
                             order_by: List[str] = [],
                             join_list: List[Tuple[str, str, str]] = [],
                             weights: List[str] = [],
                             restrict: List[Tuple[str, List]] = [],
                             run_async: bool = False,
                             split_enduses: bool = False,
                             get_query_only: bool = False,
                             limit: int = None
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

            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            limit: The maximum number of rows to query

            run_async: Whether to run the query in the background. Returns immediately if running in background,
                       blocks otherwise.
            split_enduses: Whether to query for each enduses in a separate query to reduce Athena load for query. Useful
                           when Athena runs into "Query exhausted resources ..." errors.
            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging


        Returns:
                if get_query_only is True, returns the query_string, otherwise,
                    if run_async is True, it returns a query_execution_id.
                    if run_async is False, it returns the result_dataframe

        """

        if split_enduses:
            return self.aggregate_timeseries_light(enduses=enduses, group_by=group_by, order_by=order_by,
                                                   join_list=join_list, weights=weights, restrict=restrict,
                                                   run_async=run_async, get_query_only=get_query_only,
                                                   limit=limit)
        [self.get_tbl(jl[0]) for jl in join_list]  # ingress all tables in join list
        enduses = self._get_enduse_cols(enduses, table='timeseries')
        total_weight = self._get_weight(weights)
        schedule_cols = self._get_schedule_cols(enduses)

        enduse_selection = [safunc.sum(enduse * total_weight).label(self.simple_label(enduse.name))
                            for enduse in enduses + schedule_cols]
        group_by_selection = [self.get_col(g[0]).label(g[1]) if isinstance(
            g, tuple) else self.get_col(g) for g in group_by]

        if self.timestamp_column.name not in group_by:
            logger.info("Aggregation done accross timestamps. Result no longer a timeseries.")
            # The aggregation is done across time so we should compensate unit count by dividing by the total number
            # of distinct timestamps per unit
            timesteps_per_unit = self._get_simulation_timesteps_count()
            grouping_metrics_selection = [safunc.sum(1).label(
                "raw_count"), safunc.sum(total_weight / timesteps_per_unit).label("scaled_unit_count")]
        else:
            grouping_metrics_selection = [safunc.sum(1).label(
                "raw_count"), safunc.sum(total_weight).label("scaled_unit_count")]

        query = sa.select(group_by_selection + grouping_metrics_selection + enduse_selection)
        query = query.join(self.bs_table, self.bs_bldgid_column == self.ts_bldgid_column)
        if join_list:
            query = self._add_join(query, join_list)

        query = self._add_restrict(query, restrict)
        query = self._add_group_by(query, group_by)
        query = self._add_order_by(query, order_by)
        query = query.limit(limit) if limit else query

        if get_query_only:
            return self._compile(query)

        return self.execute(query, run_async=run_async)

    def _get_simulation_timesteps_count(self):
        if "simulation_timesteps_count" in self.cache:
            if self.db_name + "/" + self.ts_table.name in self.cache["simulation_timesteps_count"]:
                return self.cache['simulation_timesteps_count'][self.db_name + "/" + self.ts_table.name]
        else:
            self.cache['simulation_timesteps_count'] = {}

        # find the simulation time interval
        query = sa.select([self.ts_bldgid_column, safunc.sum(1).label('count')])
        query = query.group_by(self.ts_bldgid_column)
        sim_timesteps_count = self.execute(query)
        bld0_step_count = sim_timesteps_count['count'].iloc[0]

        if not sum(sim_timesteps_count['count'] == bld0_step_count) == len(sim_timesteps_count):
            logger.warning("Not all buildings have the same number of timestamps. This can cause wrong \
                            scaled_units_count and other problems.")

        self.cache['simulation_timesteps_count'][self.db_name + "/" + self.ts_table.name] = bld0_step_count
        return bld0_step_count

    def _get_simulation_info(self, get_query_only=False):
        # find the simulation time interval
        query0 = sa.select([self.ts_bldgid_column]).limit(1)  # get a building id
        bldg_df = self.execute(query0)
        bldg_id = bldg_df.iloc[0].values[0]
        query1 = sa.select([self.timestamp_column.distinct().label('time')]).where(self.ts_bldgid_column == bldg_id)
        query1 = query1.order_by(self.timestamp_column).limit(2)
        if get_query_only:
            return self._compile(query1)

        two_times = self.execute(query1)
        time1 = two_times[self.timestamp_column_name].iloc[0]
        time2 = two_times[self.timestamp_column_name].iloc[1]
        sim_year = time1.year
        sim_interval_seconds = (time2 - time1).total_seconds()
        return sim_year, sim_interval_seconds

    def get_building_average_kws_at(self,
                                    at_hour,
                                    at_days,
                                    enduses=None,
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

            get_query_only: Skips submitting the query to Athena and just returns the query strings. Useful for batch
                            submitting multiple queries or debugging.

        Returns:
                If get_query_only is True, returns two queries that gets the KW at two timestamps that are to immediate
                    left and right of the the supplied hour.
                If get_query_only is False, returns the average KW of each building at the given hour(s) across the
                supplied days.

        """
        if isinstance(at_hour, list):
            if len(at_hour) != len(at_days) or len(at_hour) == 0:
                raise ValueError("The length of at_hour list should be the same as length of at_days list and"
                                 " not be empty")
        elif isinstance(at_hour, float) or isinstance(at_hour, int):
            at_hour = [at_hour] * len(at_days)
        else:
            raise ValueError("At hour should be a list or a number")

        enduse_cols = self._get_enduse_cols(enduses, table='timeseries')
        total_weight = self._get_weight([])
        schedule_cols = self._get_schedule_cols(enduse_cols)

        sim_year, sim_interval_seconds = self._get_simulation_info()
        kw_factor = 3600.0 / sim_interval_seconds

        enduse_selection = [safunc.avg(enduse * total_weight * kw_factor).label(self.simple_label(enduse.name))
                            for enduse in enduse_cols + schedule_cols]
        grouping_metrics_selection = [safunc.sum(1).label("raw_count"),
                                      safunc.sum(total_weight).label("scaled_unit_count")]

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

        query = sa.select([self.ts_bldgid_column] + grouping_metrics_selection + enduse_selection)
        query = query.join(self.bs_table, self.bs_bldgid_column == self.ts_bldgid_column)
        query = self._add_group_by(query, [self.ts_bldgid_column])
        query = self._add_order_by(query, [self.ts_bldgid_column])

        lower_val_query = self._add_restrict(query, [(self.timestamp_column_name, lower_timestamps)])
        upper_val_query = self._add_restrict(query, [(self.timestamp_column_name, upper_timestamps)])

        if exact_times:
            # only one query is sufficient if the hours fall in exact timestamps
            queries = [lower_val_query]
        else:
            queries = [lower_val_query, upper_val_query]

        if get_query_only:
            return [self._compile(q) for q in queries]

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
