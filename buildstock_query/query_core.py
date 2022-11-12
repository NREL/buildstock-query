"""
# ResStockAthena
- - - - - - - - -
A class to run AWS Athena queries to get various data from a ResStock run. All queries and aggregation that can be
common accross different ResStock projects should be implemented in this class. For queries that are project specific, a
new class can be created by inheriting ResStockAthena and adding in the project specific logic and queries.

:author: Rajendra.Adhikari@nrel.gov
"""


import boto3
import contextlib
import pathlib
from pyathena.connection import Connection
from pyathena.error import OperationalError
from pyathena.sqlalchemy_athena import AthenaDialect
import sqlalchemy as sa
import dask.dataframe as dd
from pyathena.pandas.async_cursor import AsyncPandasCursor
from pyathena.pandas.cursor import PandasCursor
import os
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
from buildstock_query.utils import FutureDf, DataExistsException, CustomCompiler
from buildstock_query.utils import save_pickle, load_pickle
from concurrent import futures


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FUELS = ['electricity', 'natural_gas', 'propane', 'fuel_oil', 'coal', 'wood_cord', 'wood_pellets']


class QueryException(Exception):
    pass


class QueryCore:
    def __init__(self, workgroup: str,
                 db_name: str,
                 buildstock_type: str = None,
                 table_name: Union[str, Tuple[str, str]] = None,
                 timestamp_column_name: str = 'time',
                 building_id_column_name: str = 'building_id',
                 sample_weight: str = "build_existing_model.sample_weight",
                 region_name: str = 'us-west-2',
                 execution_history=None,
                 skip_reports=False
                 ) -> None:
        """
        A class to run common Athena queries for BuildStock runs and download results as pandas dataFrame
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
        logger.info(f"Loading {table_name} ...")
        self.workgroup = workgroup
        self.buildstock_type = buildstock_type
        self._query_cache: dict[str, pd.DataFrame] = {}  # {"query": query_result_df} to cache queries
        self._session_queries: set[str] = set()  # Set of all queries that is run in current session.

        self._aws_s3 = boto3.client('s3')
        self._aws_athena = boto3.client('athena', region_name=region_name)
        self._aws_glue = boto3.client('glue', region_name=region_name)

        self._conn = Connection(work_group=workgroup, region_name=region_name,
                                cursor_class=PandasCursor, schema_name=db_name)
        self._async_conn = Connection(work_group=workgroup, region_name=region_name,
                                      cursor_class=AsyncPandasCursor, schema_name=db_name, )

        self.db_name = db_name
        self.region_name = region_name

        self._tables: dict = OrderedDict()  # Internal record of tables

        self._batch_query_status_map: dict = {}
        self._batch_query_id = 0

        self.timestamp_column_name = timestamp_column_name
        self.building_id_column_name = building_id_column_name
        self.sample_weight = sample_weight
        self.table_name = table_name
        self._initialize_tables()
        self._initialize_book_keeping(execution_history)

        with contextlib.suppress(FileNotFoundError):
            self.load_cache()

    def load_cache(self, path: str = None):
        """Read and update query cache from pickle file.

        Args:
            path (str, optional): The path to the pickle file. If not provided, reads from current directory.
        """
        path = path or f"{self.table_name}_query_cache.pkl"
        before_count = len(self._query_cache)
        saved_cache = load_pickle(path)
        logger.info(f"{len(saved_cache)} queries cache read from {path}.")
        self._query_cache.update(saved_cache)
        after_count = len(self._query_cache)
        if diff := after_count - before_count:
            logger.info(f"{diff} queries cache is updated.")
        else:
            logger.info("Cache already upto date.")

    def save_cache(self, path: str = None, trim_excess: bool = False):
        """Saves queries cache to a pickle file. It is good idea to run this afer making queries so that on the next
        session these queries won't have to be run on Athena and can be directly loaded from the file.

        Args:
            path (str, optional): The path to the pickle file. If not provided, the file will be saved on the current
            directory.
            trim_excess (bool, optional): If true, any queries in the cache that is not run in current session will be
            remved before saving it to file. This is useful if the cache has accumulated a bunch of stray queries over
            several sessions that are no longer used. Defaults to False.
        """
        path = path or f"{self.table_name}_query_cache.pkl"
        if trim_excess:
            if excess_queries := [key for key in self._query_cache if key not in self._session_queries]:
                for query in excess_queries:
                    del self._query_cache[query]
                logger.info(f"{len(excess_queries)} excess queries removed from cache.")
        save_pickle(path, self._query_cache)
        logger.info(f"{len(self._query_cache)} queries cache saved to {path}")

    def _initialize_tables(self):
        self.bs_table, self.ts_table, self.up_table = self._get_tables(self.table_name)

        self.bs_bldgid_column = self.bs_table.c[self.building_id_column_name]
        if self.ts_table is not None:
            self.timestamp_column = self.ts_table.c[self.timestamp_column_name]
            self.ts_bldgid_column = self.ts_table.c[self.building_id_column_name]
        if self.up_table is not None:
            self.up_bldgid_column = self.up_table.c[self.building_id_column_name]
        self.sample_wt = self._get_sample_weight(self.sample_weight)

    def _get_sample_weight(self, sample_weight):
        if not sample_weight:
            return sa.literal(1)
        elif isinstance(sample_weight, str):
            try:
                return self.get_column(sample_weight)
            except ValueError:
                logger.error("Sample weight column not found. Using weight of 1.")
                return sa.literal(1)
        elif isinstance(sample_weight, (int, float)):
            return sa.literal(sample_weight)

    def get_table(self, table_name, missing_ok=False):

        if isinstance(table_name, sa.schema.Table):
            return table_name  # already a table

        try:
            return self._tables.setdefault(table_name, sa.Table(table_name, self.meta, autoload_with=self.engine))
        except sa.exc.NoSuchTableError:
            if missing_ok:
                logger.warning(f"No {table_name} table is present.")
                return None
            else:
                raise

    def get_column(self, column_name: sa.Column | sa.sql.elements.Label | str, table_name=None):
        if isinstance(column_name, (sa.Column, sa.sql.elements.Label)):
            return column_name  # already a col
        if table_name:
            valid_tables = [self.get_table(table_name)]
        else:
            valid_tables = [table for _, table in self._tables.items() if column_name in table.columns]

        if not valid_tables:
            raise ValueError(f"Column {column_name} not found in any tables {[t.name for t in self._tables.values()]}")
        if len(valid_tables) > 1:
            logger.warning(
                f"Column {column_name} found in multiple tables {[t.name for t in valid_tables]}."
                f"Using {valid_tables[0].name}")
        return valid_tables[0].c[column_name]

    def _get_tables(self, table_name: str | tuple):
        self.engine = self._create_athena_engine(region_name=self.region_name, database=self.db_name,
                                                 workgroup=self.workgroup)
        self.meta = sa.MetaData(bind=self.engine)
        if isinstance(table_name, str):
            baseline_table = self.get_table(f'{table_name}_baseline')
            ts_table = self.get_table(f'{table_name}_timeseries', missing_ok=True)
            upgrade_table = self.get_table(f'{table_name}_upgrades', missing_ok=True)
        elif isinstance(table_name, tuple):
            baseline_table = self.get_table(f'{table_name[0]}')
            ts_table = self.get_table(f'{table_name[1]}', missing_ok=True)
            upgrade_table = self.get_table(f'{table_name[2]}', missing_ok=True)
        else:
            baseline_table = None
            ts_table = None
            upgrade_table = None
        return baseline_table, ts_table, upgrade_table

    def _initialize_book_keeping(self, execution_history):
        self.execution_history_file = execution_history or '.execution_history'
        self.execution_cost = {'GB': 0, 'Dollars': 0}  # Tracks the cost of current session. Only used for Athena query
        self.seen_execution_ids = set()  # set to prevent double counting same execution id

        if os.path.exists(self.execution_history_file):
            with open(self.execution_history_file, 'r') as f:
                existing_entries = f.readlines()
            valid_entries = []
            for entry in existing_entries:
                with contextlib.suppress(ValueError, TypeError):
                    entry_time, _ = entry.split(',')
                    if time.time() - float(entry_time) < 24 * 60 * 60:  # discard history if more than a day old
                        valid_entries += entry
            with open(self.execution_history_file, 'w') as f:
                f.writelines(valid_entries)

    @property
    def execution_ids_history(self):
        exe_ids = []
        if os.path.exists(self.execution_history_file):
            with open(self.execution_history_file, 'r') as f:
                for line in f:
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
            raise QueryException(f"Deleting it failed. Reason: {reason}")

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
        s3_data = self._aws_s3.list_objects(Bucket=s3_bucket, Prefix=f'{s3_prefix}/{table_name}')

        if 'Contents' in s3_data and override is False:
            raise DataExistsException("Table already exists", f's3://{s3_location}/{table_name}/{table_name}.csv')
        if 'Contents' in s3_data:
            existing_objects = [{'Key': el['Key']} for el in s3_data['Contents']]
            print(f"The following existing objects is being delete and replaced: {existing_objects}")
            print(f"Saving s3://{s3_location}/{table_name}/{table_name}.parquet)")
            self._aws_s3.delete_objects(Bucket=s3_bucket, Delete={"Objects": existing_objects})
        print(f"Saving factors to s3 in s3://{s3_location}/{table_name}/{table_name}.parquet")
        table_df.to_parquet(f's3://{s3_location}/{table_name}/{table_name}.parquet', index=False)
        print("Saving Done.")

        column_formats = []
        for column_name, dtype in table_df.dtypes.items():
            if np.issubdtype(dtype, np.integer):
                col_type = "int"
            elif np.issubdtype(dtype, np.floating):
                col_type = "double"
            else:
                col_type = "string"
            column_formats.append(f"`{column_name}` {col_type}")

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
            if result.upper() != "SUCCEEDED":
                raise QueryException(f"There was an existing table named {table_name}. Deleting it failed."
                                     f" Reason: {reason}")
            result, reason = self.execute_raw(table_create_query)
            if result.upper() == "SUCCEEDED":
                return "SUCCEEDED"
            else:
                raise QueryException(f"There was an existing table named {table_name} which is now successfully "
                                     f"deleted but new table failed to be created. Reason: {reason}")
        elif result.upper() == "SUCCEEDED":
            return "SUCCEEDED"
        else:
            raise QueryException(f"Failed to create the table. Reason: {reason}")

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

        response = self._aws_athena.start_query_execution(
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
            query_stat = self._aws_athena.get_query_execution(QueryExecutionId=query_execution_id)
            if query_stat['QueryExecution']['Status']['State'].lower() not in ['pending', 'running', 'queued']:
                reason = query_stat['QueryExecution']['Status'].get('StateChangeReason', '')
                return query_stat['QueryExecution']['Status']['State'], reason
            time.sleep(1)

        raise TimeoutError(f"Query failed to complete within 30 mins. Last status: {query_stat}")

    def _save_execution_id(self, execution_id):
        with open(self.execution_history_file, 'a') as f:
            f.write(f'{time.time()},{execution_id}\n')

    def _log_execution_cost(self, execution_id):
        if not execution_id.startswith('A'):
            # Can't log cost for Spark query
            return
        res = self._aws_athena.get_query_execution(QueryExecutionId=execution_id[1:])
        scanned_GB = res['QueryExecution']['Statistics']['DataScannedInBytes'] / 1e9
        cost = scanned_GB * 5 / 1e3  # 5$ per TB scanned
        if execution_id not in self.seen_execution_ids:
            self.execution_cost['Dollars'] += cost
            self.execution_cost['GB'] += scanned_GB
            self.seen_execution_ids.add(execution_id)

        logger.info(f"{execution_id} cost {scanned_GB:.1f}GB (${cost:.1f}). Session total:"
                    f" {self.execution_cost['GB']:.1f} GB (${self.execution_cost['Dollars']:.1f})")

    def _compile(self, query) -> str:
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

        self._session_queries.add(query)
        if run_async:
            if query in self._query_cache:
                return "CACHED", FutureDf(self._query_cache[query].copy())
            # in case of asynchronous run, you get the execution id and futures object
            exe_id, result_future = self._async_conn.cursor().execute(query, na_values=[''])

            def get_pandas(future):
                res = future.result()
                if res.state != 'SUCCEEDED':
                    raise OperationalError(f"{res.state}: {res.state_change_reason}")
                else:
                    return res.as_pandas()

            result_future.as_pandas = types.MethodType(get_pandas, result_future)
            result_future.add_done_callback(lambda f: self._query_cache.update({query: f.as_pandas()}))
            self._save_execution_id(exe_id)
            return exe_id, result_future
        else:
            if query not in self._query_cache:
                self._query_cache[query] = self._conn.cursor().execute(query).as_pandas()
            return self._query_cache[query].copy()

    def print_all_batch_query_status(self):
        for count in self._batch_query_status_map.keys():
            print(f'Query {count}: {self.get_batch_query_report(count)}\n')

    def stop_batch_query(self, batch_id):
        """
        Stops all the queries running under a batch query
        Args:
            batch_id: The batch_id of the batch_query. Returned by :py:sumbit_batch_query

        Returns:
            None
        """
        self._batch_query_status_map[batch_id]['to_submit_ids'].clear()
        for exec_id in self._batch_query_status_map[batch_id]['submitted_execution_ids']:
            self.stop_query(exec_id)

    def get_failed_queries(self, batch_id):
        stats = self._batch_query_status_map.get(batch_id, None)
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
        for i, exe_id in enumerate(self._batch_query_status_map[batch_id]['submitted_execution_ids']):
            completion_stat = self.get_query_status(exe_id)
            if completion_stat in ['FAILED', 'CANCELLED']:
                failed_ids.append(self._batch_query_status_map[batch_id]['submitted_ids'][i])
        return failed_ids

    def get_batch_query_report(self, batch_id: int):
        """
        Returns the status of the queries running under a batch query.
        Args:
            batch_id: The batch_id of the batch_query.

        Returns:
            A dictionary detailing status of the queries.
        """
        if not (stats := self._batch_query_status_map.get(batch_id, None)):
            raise ValueError(f"{batch_id=} not found.")
        success_count = 0
        fail_count = 0
        running_count = 0
        other = 0
        for exe_id in stats['submitted_execution_ids']:
            if exe_id == 'CACHED':
                completion_stat = "SUCCEEDED"
            else:
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
            raise QueryException('Batch query not completed yet.')

        self.wait_for_batch_query(batch_id)
        logger.info("Batch query completed. ")
        report = self.get_batch_query_report(batch_id)
        query_exe_ids = self._batch_query_status_map[batch_id]['submitted_execution_ids']
        query_futures = self._batch_query_status_map[batch_id]['queries_futures']
        if report['Failed'] > 0:
            logger.warning(f"{report['Failed']} queries failed. Redoing them")
            failed_ids, failed_queries = self.get_failed_queries(batch_id)
            new_batch_id = self.submit_batch_query(failed_queries)
            new_exe_ids = self._batch_query_status_map[new_batch_id]['submitted_execution_ids']

            self.wait_for_batch_query(new_batch_id)
            new_exe_ids_map = {entry[0]: entry[1] for entry in zip(failed_ids, new_exe_ids)}

            new_report = self.get_batch_query_report(new_batch_id)
            if new_report['Failed'] > 0:
                self.print_failed_query_errors(new_batch_id)
                raise QueryException("Queries failed again. Sorry!")
            logger.info("The queries succeeded this time. Gathering all the results.")
            # replace the old failed exe_ids with new successful exe_ids
            for indx, old_exe_id in enumerate(query_exe_ids):
                query_exe_ids[indx] = new_exe_ids_map.get(old_exe_id, old_exe_id)

        if len(query_exe_ids) == 0:
            raise ValueError("No query was submitted successfully")
        res_df_array = []
        for index, exe_id in enumerate(query_exe_ids):
            df = query_futures[index].as_pandas()
            if combine:
                if len(df) == 0:
                    df = pd.DataFrame({'query_id': [index]})
                else:
                    df['query_id'] = index
            logger.info(f"Got result from Query [{index}] ({exe_id})")
            res_df_array.append(df)
        if not combine:
            return res_df_array
        logger.info("Concatenating the results.")
        # return res_df_array
        return pd.concat(res_df_array)

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
        submitted_ids: list[int] = []
        submitted_execution_ids: list[str] = []
        submitted_queries: list[str] = []
        queries_futures: list[futures.Future] = []
        self._batch_query_id += 1
        batch_query_id = self._batch_query_id
        self._batch_query_status_map[batch_query_id] = {'to_submit_ids': to_submit_ids,
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
                self._log_execution_cost(execution_id)
                return result
            elif stat.upper() == 'FAILED':
                error = self.get_query_error(execution_id)
                raise QueryException(error)
            else:
                logger.info(f"Query status is {stat}")
                time.sleep(30)

        raise QueryException(f'Query timed-out. {self.get_query_status(execution_id)}')

    def get_result_from_s3(self, query_execution_id):
        query_status = self.get_query_status(query_execution_id)
        if query_status == 'SUCCEEDED':
            path = self.get_query_output_location(query_execution_id)
            df = dd.read_csv(path).compute()[0]
            return df
        # If failed, return error message
        elif query_status == 'FAILED':
            raise QueryException(self.get_query_error(query_execution_id))
        elif query_status in ['RUNNING', 'QUEUED', 'PENDING']:
            raise QueryException(f"Query still {query_status}")
        else:
            raise QueryException(f"Query has unkown status {query_status}")

    def get_query_output_location(self, query_id):
        stat = self._aws_athena.get_query_execution(QueryExecutionId=query_id)
        output_path = stat['QueryExecution']['ResultConfiguration']['OutputLocation']
        return output_path

    def get_query_status(self, query_id):
        stat = self._aws_athena.get_query_execution(QueryExecutionId=query_id)
        return stat['QueryExecution']['Status']['State']

    def get_query_error(self, query_id):
        stat = self._aws_athena.get_query_execution(QueryExecutionId=query_id)
        return stat['QueryExecution']['Status']['StateChangeReason']

    def get_all_running_queries(self):
        """
        Gives the list of all running queries (for this instance)

        Return:
            List of query execution ids of all the queries that are currently running in Athena.
        """
        exe_ids = self._aws_athena.list_query_executions(WorkGroup=self.workgroup)['QueryExecutionIds']
        exe_ids = list(exe_ids)

        running_ids = [i for i in exe_ids if i in self.execution_ids_history and
                       self.get_query_status(i) == "RUNNING"]
        return running_ids

    def stop_all_queries(self):
        """
        Stops all queries that are running in Athena for this instance.
        Returns:
            Nothing

        """
        for count, stat in self._batch_query_status_map.items():
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
        return self._aws_athena.stop_query_execution(QueryExecutionId=execution_id)

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
            tbl = self.get_table(table)
            return tbl.columns

    @staticmethod
    def _simple_label(label):
        if '.' in label:
            return ''.join(label.split('.')[1:])
        else:
            return label

    def _add_restrict(self, query, restrict):
        if not restrict:
            return query
        where_clauses = []
        for col, criteria in restrict:
            if isinstance(criteria, (list, tuple)):
                if len(criteria) > 1:
                    where_clauses.append(self.get_column(col).in_(criteria))
                    continue
                else:
                    criteria = criteria[0]
            where_clauses.append(self.get_column(col) == criteria)
        query = query.where(*where_clauses)
        return query

    def _get_name(self, col):
        if isinstance(col, tuple):
            return col[1]
        if isinstance(col, str):
            return col
        if isinstance(col, (sa.Column, sa.sql.elements.Label)):
            return col.name
        raise ValueError(f"Can't get name for {col} of type {type(col)}")

    def _add_join(self, query, join_list):
        for new_table_name, baseline_column_name, new_column_name in join_list:
            new_tbl = self.get_table(new_table_name)
            query = query.join(new_tbl, self.bs_table.c[baseline_column_name]
                               == new_tbl.c[new_column_name])
        return query

    def _add_group_by(self, query, group_by_selection):
        if group_by_selection:
            selected_cols = list(query.selected_columns)
            a = [sa.text(str(selected_cols.index(g) + 1)) for g in group_by_selection]
            query = query.group_by(*a)
        return query

    def _add_order_by(self, query, order_by_selection):
        if order_by_selection:
            selected_cols = list(query.selected_columns)
            a = [sa.text(str(selected_cols.index(g) + 1)) for g in order_by_selection]
            query = query.order_by(*a)
        return query

    def _get_weight(self, weights):
        total_weight = self.sample_wt
        for weight_col in weights:
            if isinstance(weight_col, tuple):
                tbl = self.get_table(weight_col[1])
                total_weight *= tbl.c[weight_col[0]]
            else:
                total_weight *= self.get_column(weight_col)
        return total_weight

    def delete_everything(self):
        info = self._aws_glue.get_table(DatabaseName=self.db_name, Name=self.bs_table.name)
        self.pth = pathlib.Path(info['Table']['StorageDescriptor']['Location']).parent
        tables_to_delete = [self.bs_table.name]
        if self.ts_table is not None:
            tables_to_delete.append(self.ts_table.name)
        if self.up_table is not None:
            tables_to_delete.append(self.up_table.name)
        print(f"Will delete the following tables {tables_to_delete} and the {self.pth} folder")
        while True:
            curtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            confirm = input(f"Enter {curtime} to confirm.")
            if confirm == "":
                print("Abandoned the idea.")
                break
            if confirm != curtime:
                print(f"Please pass {curtime} as confirmation to confirm you want to delete everything.")
                continue
            self._aws_glue.batch_delete_table(DatabaseName=self.db_name, TablesToDelete=tables_to_delete)
            print("Deleted the table from athena, now will delete the data in s3")
            s3 = boto3.resource('s3')
            bucket = s3.Bucket(self.pth.parts[1])
            prefix = str(pathlib.Path(*self.pth.parts[2:]))
            total_files = [file.key for file in bucket.objects.filter(Prefix=prefix)]
            print(f"There are {len(total_files)} files to be deleted. Deleting them now")
            bucket.objects.filter(Prefix=prefix).delete()
            print("Delete from s3 completed")
            break