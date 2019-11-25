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
import pythena
from typing import List, Tuple, Union
import time
import logging
from threading import Thread
from botocore.exceptions import ClientError
import pandas as pd
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResStockAthena:
    def __init__(self, db_name: str,
                 table_name: Union[str, Tuple[str, str]],
                 region_name: str = 'us-west-2',
                 timestamp_column_name='time',
                 execution_history=None) -> None:
        """
        A class to run common Athena queries for ResStock runs and download results as pandas dataFrame
        Args:
            db_name: The athena database name
            table_name: If a single string is provided, say, 'mfm_run', then it must correspond to two tables in athena
                        named mfm_run_baseline and mfm_run_timeseries. Or, two strings can be provided as a tuple, (such
                        as 'mfm_run_2_baseline', 'mfm_run5_timeseries') and they must be a baseline table and a
                        timeseries table.
            region_name: The AWS region where the database exists. Defaults to 'us-west-2'.
            timestamp_column_name: The column name for the time column. Defaults to 'time'
            execution_history: A temporary files to record which execution is run by the user, to help stop them. Will
                               use .execution_history if not supplied.
        """
        self.py_thena = pythena.Athena(db_name, region_name)
        self.aws_athena = boto3.client('athena', region_name=region_name)
        self.aws_glue = boto3.client('glue', region_name=region_name)
        self.db_name = db_name
        self.region_name = region_name
        self.timestamp_column_name = timestamp_column_name

        if isinstance(table_name, str):
            self.ts_table_name = f'{table_name}_timeseries'
            self.baseline_table_name = f'{table_name}_baseline'
        else:
            self.baseline_table_name = f'{table_name[0]}'
            self.ts_table_name = f'{table_name[1]}'

        self.ts_table = self.aws_glue.get_table(DatabaseName=self.db_name, Name=self.ts_table_name)['Table']
        self.baseline_table = self.aws_glue.get_table(DatabaseName=self.db_name, Name=self.baseline_table_name)['Table']
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

    def _save_execution_id(self, execution_id):
        with open(self.execution_history_file, 'a') as f:
            f.write(f'{time.time()},{execution_id}\n')

    def execute(self, query, run_async=False):
        """
        Executes a query
        Args:
            query: The SQL query to run in Athena
            run_async: Whether to wait until the query completes (run_async=False) or return immediately
            (run_async=True).

        Returns:
            if run_async is False, returns the results dataframe.
            if run_async is  True, returns the query_execution_id. Use `get_query_result` to get the result after
            the query is completed. Use :get_status: to check the status of the query.
        """
        res = self.py_thena.execute(query, save_results=True, run_async=run_async)
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
            for exe_id in stats['submitted_execution_ids']:
                completion_stat = self.get_query_status(exe_id)
                if completion_stat == 'RUNNING':
                    running_count += 1
                elif completion_stat == 'SUCCEEDED':
                    success_count += 1
                elif completion_stat in ['FAILED', 'CANCELLED']:
                    fail_count += 1

            result = {'Submitted': len(stats['submitted_ids']),
                      'Running': running_count,
                      'Pending': len(stats['to_submit_ids']),
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

    def get_batch_query_result(self, batch_id, no_block=False):
        """
        Concatenates and returns the results of all the queries of a batchquery
        Args:
            batch_id (int): The batch_id for the batch_query
            no_block (bool): Whether to wait until all queries have completed or return immediately. If you use
                            no_block = true and the batch hasn't completed, it will throw BatchStillRunning exception.

        Returns:
            The concatenated dataframe of the results of all the queries in a batch query.

        """
        last_time = time.time()
        last_report = None
        while True:
            if self.did_batch_query_complete(batch_id):
                query_exe_ids = self.batch_query_status_map[batch_id]['submitted_execution_ids']
                res_df_array = []
                for index, exe_id in enumerate(query_exe_ids):
                    res_df_array.append(self.get_query_result(exe_id))
                return pd.concat(res_df_array)
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
                current_id = to_submit_ids[-1]  # get the last one
                current_query = queries[-1]
                try:
                    execution_id = self.execute(current_query, run_async=True)
                    logger.info(f"Submitted queries[{current_id}]")
                    to_submit_ids.pop()  # if query queued successfully, remove it from the list
                    queries.pop()
                    submitted_ids.append(current_id)
                    submitted_execution_ids.append(execution_id)
                    submitted_queries.append(current_query)
                except ClientError as e:
                    if e.response['Error']['Code'] == 'TooManyRequestsException':
                        time.sleep(60)  # wait for a minute before submitting another query
                    else:
                        raise e

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
        exe_ids = self.aws_athena.list_query_executions()['QueryExecutionIds']
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
            if not fuel_type:
                cols = [c for c in cols if 'simulation_output_report' in c]
                cols = [c for c in cols if fuel_type in c]
            return cols

    @staticmethod
    def simple_label(label):
        if label.startswith('simulation_output_report'):
            return label.split('.')[1]
        else:
            return label

    @staticmethod
    def dress_literal(l):
        if isinstance(l, float) or isinstance(l, int):
            return f"{l}"
        elif isinstance(l, str):
            return f"'{l}'"
        else:
            raise TypeError(f'Unsupported Type: {type(l)}')

    @staticmethod
    def make_column_string(c):
        if not c.startswith('"'):
            return f'"{c}"'
        else:
            return c

    def get_distinct_vals(self, column: str, table_name: str = None, get_query_only: bool = False):
        table_name = self.baseline_table_name if table_name is None else table_name
        C = ResStockAthena.make_column_string
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
            weight = C("build_existing_model.sample_weight")
        else:
            weight = C(weight_column) if weight_column else 1

        query = f"select {C(column)}, sum(1) as raw_count, sum({weight}) as weighted_count from {C(table_name)} " \
            f"group by 1 order by 1"

        if get_query_only:
            return query

        r = self.execute(query, run_async=False)
        return r

    def aggregate_annual(self,
                         enduses: List[str] = None,
                         group_by: List[str] = None,
                         order_by: List[str] = None,
                         join_list: List[Tuple[str, str, str]] = [],
                         weights: List[str] = [],
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

            weights: The additional column to use as weight. The "build_existing_model.sample_weight" is already used.

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

        sample_weight = C("build_existing_model.sample_weight")
        n_units_col = C("build_existing_model.units_represented")

        total_weight = f'{sample_weight}'
        for weight in weights:
            total_weight += f' * {C(weight)} '

        enduse_cols = ', '.join([f"sum({C(c)} * {total_weight} / {n_units_col}) as {self.simple_label(c)}"
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

        if restrict:
            query += f" where "
            condition_strs = []
            for column, vals in restrict:
                vals = [self.dress_literal(v) for v in vals]
                if len(vals) == 1:
                    condition_strs.append(f'''({C(column)} = {vals[0]})''')
                else:
                    condition_strs.append(f'''({C(column)} in ({', '.join(vals)}))''')

            query += " and ".join(condition_strs)

        if group_by:
            query += f" group by " + ", ".join([f'{C(g)}' for g in group_by])
        if order_by:
            query += f" order by " + ", ".join([f'{C(o)}' for o in order_by])

        if get_query_only:
            return query

        t = time.time()
        res = self.execute(query, run_async=True)
        if run_async:
            return res
        else:
            time.sleep(1)
            while time.time() - t < 30 * 60:
                if self.py_thena.get_query_status(res).lower() != 'running':
                    result = self.get_query_result(res)
                    return result
                else:
                    time.sleep(10)

            raise Exception(f'Query failed {self.py_thena.get_query_status(res)}')

    def aggregate_timeseries(self,
                             enduses: List[str] = None,
                             group_by: List[str] = None,
                             order_by: List[str] = None,
                             join_list: List[Tuple[str, str, str]] = [],
                             weights: List[str] = [],
                             restrict: [Tuple[str, List]] = [],
                             run_async: bool = False,
                             get_query_only: bool = False):
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

            run_async: Whether to run the query in the background. Returns immediately if running in background,
                       blocks otherwise.
            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging

        Returns:
                if get_query_only is True, returns the query_string, otherwise,
                    if run_async is True, it returns a query_execution_id.
                    if run_async is False, it returns the result_dataframe

        """

        C = self.make_column_string
        if not enduses:
            enduses = self.get_cols(table='timeseries', fuel_type='electricity')

        sample_weight = C("build_existing_model.sample_weight")
        n_units_col = C("build_existing_model.units_represented")

        total_weight = f'{sample_weight}'
        for weight in weights:
            total_weight += f' * {C(weight)} '

        enduse_cols = ', '.join([f"sum({C(c)} * {total_weight} / {n_units_col}) as {self.simple_label(c)}"
                                 for c in enduses])

        if not group_by:
            query = f'select {enduse_cols} from {C(self.ts_table_name)}'
        else:
            group_by_cols = ", ".join([f'{C(g)}' for g in group_by])
            grouping_metrics_cols = f" sum(1) as raw_count, sum({total_weight}) as scaled_unit_count"
            select_cols = group_by_cols + ", " + grouping_metrics_cols
            if enduse_cols:
                select_cols += ", " + enduse_cols
            query = f"select {select_cols} from {C(self.ts_table_name)}"
            join_clause = f''' join {C(self.baseline_table_name)} on {C(self.ts_table_name)}."building_id" =\
                             {C(self.baseline_table_name)}."building_id" '''
            for new_table_name, baseline_column_name, new_column_name in join_list:
                join_clause += f''' join {C(new_table_name)} on\
                {C(self.baseline_table_name)}.{C(baseline_column_name)} = {C(new_table_name)}.{C(new_column_name)}'''
            query += join_clause

        if restrict:
            query += f" where "
            condition_strs = []
            for column, vals in restrict:
                vals = [self.dress_literal(v) for v in vals]
                if len(vals) == 1:
                    condition_strs.append(f'''({C(column)} = {vals[0]})''')
                else:
                    condition_strs.append(f'''({C(column)} in ({', '.join(vals)}))''')

            query += " and ".join(condition_strs)

        if group_by:
            query += f" group by " + ", ".join([f'''{C(g)}''' for g in group_by])
        if order_by:
            query += f" order by " + ", ".join([f'''{C(o)}''' for o in order_by])

        if get_query_only:
            return query

        t = time.time()
        res = self.execute(query, run_async=True)
        if run_async:
            return res
        else:
            time.sleep(1)
            while time.time() - t < 30 * 60:
                if self.py_thena.get_query_status(res).lower() != 'running':
                    result = self.get_query_result(res)
                    return result
                else:
                    time.sleep(10)

            raise Exception(f'Query failed {self.py_thena.get_query_status(res)}')
