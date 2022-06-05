"""
# ResStockAthena
- - - - - - - - -
A class to run AWS Athena queries to get various data from a ResStock run. All queries and aggregation that can be
common accross different ResStock projects should be implemented in this class. For queries that are project specific, a
new class can be created by inheriting ResStockAthena and adding in the project specific logic and queries.

:author: Rajendra.Adhikari@nrel.gov
"""


import re
import boto3
import contextlib
import pathlib
from collections import Counter
from pyathena.connection import Connection
from pyathena.error import OperationalError
from pyathena.sqlalchemy_athena import AthenaDialect
import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
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
from eulpda.smart_query.upgrades_analyzer import UpgradesAnalyzer
from eulpda.smart_query.utils import FutureDf, DataExistsException, CustomCompiler, print_r, print_g
from eulpda.smart_query.utils import save_pickle, load_pickle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResStockAthena:
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
        logger.info(f"Loading {table_name} ...")
        self.workgroup = workgroup
        self.buildstock_type = buildstock_type
        self._query_cache = {}

        self._aws_s3 = boto3.client('s3')
        self._aws_athena = boto3.client('athena', region_name=region_name)
        self._aws_glue = boto3.client('glue', region_name=region_name)

        self._conn = Connection(work_group=workgroup, region_name=region_name,
                                cursor_class=PandasCursor, schema_name=db_name)
        self._async_conn = Connection(work_group=workgroup, region_name=region_name,
                                      cursor_class=AsyncPandasCursor, schema_name=db_name, )

        self.db_name = db_name
        self.region_name = region_name

        self._tables = OrderedDict()  # Internal record of tables

        self._batch_query_status_map = {}
        self._batch_query_id = 0

        self.timestamp_column_name = timestamp_column_name
        self.building_id_column_name = building_id_column_name
        self.sample_weight = sample_weight
        self.table_name = table_name
        self._initialize_tables()
        self._initialize_book_keeping(execution_history)

        with contextlib.suppress(FileNotFoundError):
            self.load_cache()

        if not skip_reports:
            logger.info("Getting Success counts...")
            print(self.get_success_report())
            if self.ts_table is not None:
                self.check_integrity()
        self.save_cache()

    def load_cache(self, path=None):
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

    def save_cache(self, path=None):
        path = path or f"{self.table_name}_query_cache.pkl"
        save_pickle(path, self._query_cache)
        logger.info(f"{len(self._query_cache)} queries cache saved to {path}")

    def _get_bs_success_report(self, get_query_only=False):
        bs_query = sa.select([self.bs_table.c['completed_status'], safunc.count().label("count")])
        bs_query = bs_query.group_by(sa.text('1'))
        if get_query_only:
            return self._compile(bs_query)
        df = self.execute(bs_query)
        df.insert(0, 'upgrade', 0)
        return self._process_report(df)

    def _get_change_report(self, get_query_only=False):
        """Returns counts of buildings to which upgrade didn't do any changes on energy consumption

        Args:
            get_query_only (bool, optional): _description_. Defaults to False.
        """
        queries = []
        chng_types = ["no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng", "null", "any"]
        for ch_type in chng_types:
            up_query = sa.select([self.up_table.c['upgrade'], safunc.count().label("change")])
            up_query = up_query.join(self.bs_table, self.bs_bldgid_column == self.up_bldgid_column)
            conditions = self._get_change_conditions(change_type=ch_type)
            up_query = up_query.where(sa.and_(self.bs_table.c['completed_status'] == 'Success',
                                              self.up_table.c['completed_status'] == 'Success',
                                              conditions))
            up_query = up_query.group_by(sa.text('1'))
            up_query = up_query.order_by(sa.text('1'))
            queries.append(self._compile(up_query))
        if get_query_only:
            return queries
        change_df = None
        for chng_type, query in zip(chng_types, queries):
            df = self.execute(query)
            df.rename(columns={"change": chng_type}, inplace=True)
            df['upgrade'] = df['upgrade'].map(int)
            df = df.set_index('upgrade').sort_index()
            change_df = change_df.join(df, how='outer') if change_df is not None else df
        return change_df.fillna(0)

    def print_change_details(self, upgrade, yml_file, change_type='no-chng'):
        ua = self.get_upgrades_analyzer(yml_file)
        bad_bids = self.get_buildings_by_change(upgrade, change_type=change_type)
        good_bids = self.get_buildings_by_change(upgrade, change_type='ok-chng')
        ua.print_unique_characteristic(upgrade, change_type, good_bids, bad_bids)

    def _get_upgrade_buildings(self, upgrade, trim_missing_bs=True, get_query_only=False):
        up_query = sa.select([self.up_bldgid_column])
        if trim_missing_bs:
            up_query = up_query.join(self.bs_table, self.bs_bldgid_column == self.up_bldgid_column)
            up_query = up_query.where(sa.and_(self.bs_table.c['completed_status'] == 'Success',
                                              self.up_table.c['completed_status'] == 'Success',
                                              self.up_table.c['upgrade'] == str(upgrade),
                                              ))
        else:
            up_query = up_query.where(sa.and_(self.up_table.c['upgrade'] == str(upgrade),
                                              self.up_table.c['completed_status'] == 'Success'))
        if get_query_only:
            return self._compile(up_query)
        df = self.execute(up_query)
        return df[self.bs_bldgid_column.name].values

    def _get_change_conditions(self, change_type):
        threshold = 1e-3
        fuel_cols = [col.name for col in self.up_table.columns if col.name.startswith('report_simulation_output') and
                     col.name.endswith(('total_m_btu'))]  # Look at all fuel type totals
        unmet_hours_cols = ['report_simulation_output.unmet_hours_cooling_hr',
                            'report_simulation_output.unmet_hours_heating_hr']
        all_cols = fuel_cols + unmet_hours_cols
        null_chng_conditions = sa.and_(*[sa.or_(self.up_table.c[col] == sa.null(),
                                                self.bs_table.c[col] == sa.null()
                                                ) for col in fuel_cols])

        no_chng_conditions = sa.and_(*[safunc.coalesce(safunc.abs(self.up_table.c[col] -
                                                                  self.bs_table.c[col]), 0) < threshold
                                       for col in fuel_cols])
        good_chng_conditions = sa.or_(*[self.bs_table.c[col] - self.up_table.c[col] >= threshold for col in fuel_cols])
        opp_chng_conditions = sa.and_(*[safunc.coalesce(self.bs_table.c[col] - self.up_table.c[col], -1) < threshold
                                        for col in fuel_cols], sa.not_(no_chng_conditions))
        true_good_chng_conditions = sa.or_(*[self.bs_table.c[col] - self.up_table.c[col] >= threshold
                                             for col in all_cols])
        true_opp_chng_conditions = sa.and_(*[safunc.coalesce(self.bs_table.c[col] - self.up_table.c[col], -1) <
                                             threshold for col in all_cols], sa.not_(no_chng_conditions))
        if change_type == 'no-chng':
            conditions = no_chng_conditions
        elif change_type == 'bad-chng':
            conditions = opp_chng_conditions
        elif change_type == 'true-bad-chng':
            conditions = true_opp_chng_conditions
        elif change_type == 'ok-chng':
            conditions = good_chng_conditions
        elif change_type == 'true-ok-chng':
            conditions = true_good_chng_conditions
        elif change_type == 'null':
            conditions = null_chng_conditions
        elif change_type == 'any':
            conditions = sa.true
        else:
            raise ValueError(f"Invalid {change_type=}")
        return conditions

    def get_buildings_by_change(self, upgrade, change_type='no-chng', get_query_only=False):
        up_query = sa.select([self.bs_bldgid_column, self.bs_table.c['completed_status'],
                              self.up_table.c['completed_status']])
        up_query = up_query.join(self.up_table, self.bs_bldgid_column == self.up_bldgid_column)

        conditions = self._get_change_conditions(change_type)
        up_query = up_query.where(sa.and_(self.bs_table.c['completed_status'] == 'Success',
                                          self.up_table.c['completed_status'] == 'Success',
                                          self.up_table.c['upgrade'] == str(upgrade),
                                          conditions))
        if get_query_only:
            return self._compile(up_query)
        df = self.execute(up_query)
        return df[self.bs_bldgid_column.name].values

    def _get_up_success_report(self, trim_missing_bs=True, get_query_only=False):
        """Get success report for upgrades

        Args:
            trim_missing_bs (bool, optional): Ignore buildings that have no successful runs in the baseline.
                Defaults to True.
            get_query_only (bool, optional): Returns query only without the result. Defaults to False.

        Returns:
            Union[str, pd.DataFrame]: If get_query_only then returns the query string. Otherwise returns the dataframe.
        """
        up_query = sa.select([self.up_table.c['upgrade'], self.up_table.c['completed_status'],
                              safunc.count().label("count")])
        if trim_missing_bs:
            up_query = up_query.join(self.bs_table, self.bs_bldgid_column == self.up_bldgid_column)
            up_query = up_query.where(self.bs_table.c['completed_status'] == 'Success')

        up_query = up_query.group_by(sa.text('1'), sa.text('2'))
        up_query = up_query.order_by(sa.text('1'), sa.text('2'))
        if get_query_only:
            return self._compile(up_query)
        df = self.execute(up_query)
        return self._process_report(df)

    def _process_report(self, df):
        df['upgrade'] = df['upgrade'].map(int)
        pf = df.pivot(index=['upgrade'], columns=['completed_status'], values=['count'])
        pf.columns = [c[1] for c in pf.columns]
        pf['Sum'] = pf.sum(axis=1)
        for col in ['Fail', 'Invalid']:
            if col not in pf.columns:
                pf.insert(1, col, 0)
        return pf

    def _get_full_options_report(self, trim_missing_bs=True, get_query_only=False):
        opt_name_cols = [c for c in self.up_table.columns if c.name.startswith("upgrade_costs.option_")
                         and c.name.endswith("name")]
        query = sa.select([self.up_table.c['upgrade']] + opt_name_cols + [safunc.count().label("Success")])
        if trim_missing_bs:
            query = query.join(self.bs_table, self.bs_bldgid_column == self.up_bldgid_column)
            query = query.where(self.bs_table.c['completed_status'] == 'Success')

        grouping_texts = [sa.text(str(i+1)) for i in range(1+len(opt_name_cols))]
        query = query.group_by(*grouping_texts)
        query = query.order_by(*grouping_texts)
        if get_query_only:
            return self._compile(query)
        df = self.execute(query)
        simple_names = [f"option{i+1}" for i in range(len(opt_name_cols))]
        df.columns = ['upgrade'] + simple_names + ['Success']
        df['upgrade'] = df['upgrade'].map(int)
        applied_rows = df[simple_names].any(axis=1)  # select only rows with at least one option applied
        return df[applied_rows]

    def get_options_report(self, trim_missing_bs=True):
        full_report = self._get_full_options_report(trim_missing_bs=trim_missing_bs)
        option_cols = [c for c in full_report.columns if c.startswith("option")]
        total_counts = Counter()
        for option in option_cols:
            counts = Counter(full_report.groupby(['upgrade', option])['Success'].sum().to_dict())
            total_counts += counts
        option_df = pd.DataFrame.from_dict({'Success': total_counts}, orient='columns')
        option_df = option_df.reset_index()
        option_df.columns = ['upgrade', 'option', 'Success']
        upgrade_df = self.get_success_report(trim_missing_bs=trim_missing_bs).reset_index()
        upgrade_df = upgrade_df[upgrade_df['upgrade'] != 0]
        upgrade_df = upgrade_df[['upgrade', 'Success', 'Fail', 'Unapplicaple']]
        upgrade_df.insert(1, 'option', 'All')
        full_df = pd.concat([option_df, upgrade_df])
        full_df = full_df.sort_values(['upgrade', 'option'])
        return full_df

    def get_buildstock_df(self):
        results_df = self.get_results_csv()
        results_df = results_df[results_df["completed_status"] == "Success"]
        buildstock_cols = [c for c in results_df.columns if c.startswith("build_existing_model.")]
        buildstock_df = results_df[buildstock_cols]
        buildstock_cols = [''.join(c.split(".")[1:]).replace("_", " ") for c in buildstock_df.columns
                           if c.startswith("build_existing_model.")]
        buildstock_df.columns = buildstock_cols
        return buildstock_df

    def get_upgrades_analyzer(self, yaml_file):
        buildstock_df = self.get_buildstock_df()
        ua = UpgradesAnalyzer(buildstock=buildstock_df, yaml_file=yaml_file)
        return ua

    def check_options_integrity(self, yaml_file):
        ua_df = self.get_upgrades_analyzer(yaml_file).get_report()
        ua_df = ua_df[['upgrade', 'option', 'applicable_to']]
        opt_report_df = self.get_options_report().fillna(0)
        opt_report_dict = opt_report_df.set_index(['upgrade', 'option']).to_dict()
        serious = False
        for indx, row in ua_df.iterrows():
            applied_to = opt_report_dict['Success'].get((row.upgrade, row.option), 0)
            upgrade_failures = opt_report_dict['Fail'].get((row.upgrade, 'All'), 0)
            if applied_to != row.applicable_to:
                diff = row.applicable_to - applied_to
                if row.option == 'All' and diff == upgrade_failures:
                    print_g(
                        f"Upgrade {row.upgrade} was was supposed to be applied to "
                        f"{row.applicable_to} samples, but applied to {applied_to} samples. This difference of {diff}"
                        f" exactly matches with {upgrade_failures} failures in Upgrade {row.upgrade}. It's all good."
                    )
                    continue
                elif row.option == 'All':
                    serious = True
                    print_r(
                        f"SERIOUS ISSUE: Upgrade {row.upgrade} was was supposed to be applied to "
                        f"{row.applicable_to} samples, but applied to {applied_to} samples. This difference of {diff}"
                        f" doesn't match with {upgrade_failures} failures in Upgrade {row.upgrade}"
                    )

                print(f"Upgrade {row.upgrade}, option {row.option} was supposed to be applied to {row.applicable_to} "
                      f"samples. But it was applied to {applied_to} samples.")
                if 0 < diff <= upgrade_failures:
                    print(f"The difference of {diff} is likely because Upgrade {row.upgrade} caused {upgrade_failures} "
                          f"failures.")
                else:
                    print_r(f"SERIOUS ISSUE: Upgrade {row.upgrade} caused only {upgrade_failures} failures, so a "
                            f"difference of {diff} indicates problem in simulation.")
                    serious = True
        if not serious:
            print_g("Integrity check passed.")
            return True
        else:
            print_r("Integrity check failed. Please check the serious issues above.")
            return False

    def get_success_report(self, trim_missing_bs='auto', get_query_only=False):

        baseline_result = self._get_bs_success_report(get_query_only)

        if self.up_table is None:
            return baseline_result

        if trim_missing_bs == 'auto':
            if 'Success' not in baseline_result:
                logger.warning("None of the simulation was successful in baseline. The counts for upgrade will be"
                               " returned without requiring corresponding successful baseline run.")
                trim_missing_bs = False
            else:
                trim_missing_bs = True
        upgrade_result = self._get_up_success_report(trim_missing_bs, get_query_only).fillna(0)
        change_result = self._get_change_report(get_query_only).fillna(0)
        if get_query_only:
            return baseline_result, upgrade_result, change_result
        if 'Success' in upgrade_result.columns:
            pa = round(100 * (upgrade_result['Fail'] + upgrade_result['Success']) /
                       upgrade_result['Sum'], 1)
            upgrade_result['Applied %'] = pa

        pf = pd.concat([baseline_result, upgrade_result])
        pf = pf.rename(columns={'Invalid': 'Unapplicaple'})
        pf = pf.join(change_result).fillna(0)
        pf['no-chng %'] = round(100 * pf['no-chng'] / pf['Success'], 1)
        pf['bad-chng %'] = round(100 * pf['bad-chng'] / pf['Success'], 1)
        pf['ok-chng %'] = round(100 * pf['ok-chng'] / pf['Success'], 1)
        pf['true-ok-chng %'] = round(100 * pf['true-ok-chng'] / pf['Success'], 1)
        pf['true-bad-chng %'] = round(100 * pf['true-bad-chng'] / pf['Success'], 1)
        return pf

    def _get_ts_report(self, get_query_only=False):
        ts_query = sa.select([self.ts_table.c['upgrade'],
                              safunc.count(self.ts_bldgid_column.distinct()).label("count")])
        ts_query = ts_query.group_by(sa.text('1'))
        ts_query = ts_query.order_by(sa.text('1'))
        if get_query_only:
            return self._compile(ts_query)
        df = self.execute(ts_query)
        df['upgrade'] = df['upgrade'].map(int)
        df = df.set_index('upgrade')
        df = df.rename(columns={'count': 'Success'})
        return df

    def _get_rows_per_building(self, get_query_only=False):
        select_cols = []
        if self.up_table is not None:
            select_cols.append(self.ts_table.c['upgrade'])
        select_cols.append(self.ts_bldgid_column)
        select_cols.append(safunc.count().label("row_count"))
        ts_query = sa.select(select_cols)
        if self.up_table is not None:
            ts_query = ts_query.group_by(sa.text('1'), sa.text('2'))
        else:
            ts_query = ts_query.group_by(sa.text('1'))

        if get_query_only:
            return self._compile(ts_query)
        df = self.execute(ts_query)
        if (df['row_count'] == df['row_count'][0]).all():  # verify all buildings got same number of rows
            return df['row_count'][0]
        else:
            raise ValueError("Not all buildings have same number of rows.")

    def check_integrity(self):
        logger.info("Checking integrity with ts_tables ...")
        raw_ts_report = self._get_ts_report()
        raw_success_report = self.get_success_report(trim_missing_bs=False)
        bs_dict = raw_success_report['Success'].to_dict()
        ts_dict = raw_ts_report.to_dict()['Success']
        check_pass = True
        for upgrade, count in ts_dict.items():
            if count != bs_dict.get(upgrade, 0):
                print_r(f"Upgrade {upgrade} has {count} samples in timeseries table, but {bs_dict.get(upgrade, 0)}"
                        " samples in baseline/upgrade table.")
                check_pass = False
        if check_pass:
            print_g("Annual and timeseries tables are verified to have the same number of buildings.")
        try:
            rowcount = self._get_rows_per_building()
            print_g(f"All buildings are verified to have the same number of ({rowcount}) timeseries rows.")
        except ValueError:
            check_pass = False
            print_r("Different buildings have different number of timeseries rows.")
        return check_pass

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

    def _get_gcol(self, column_name):  # gcol => group by col
        if isinstance(column_name, (sa.Column, sa.sql.elements.Label)):
            return column_name  # already a col

        if isinstance(column_name, tuple):
            try:
                return self.get_column(column_name[0]).label(column_name[1])
            except ValueError:
                new_name = f"build_existing_model.{column_name[0]}"
                return self.get_column(new_name).label(column_name[1])
        elif isinstance(column_name, str):
            try:
                return self.get_column(column_name).label(self._simple_label(column_name))
            except ValueError:
                if not column_name.startswith("build_existing_model."):
                    new_name = f"build_existing_model.{column_name}"
                    return self.get_column(new_name).label(column_name)
                raise ValueError(f"Invalid column name {column_name}")
        else:
            raise ValueError(f"Invalid column name type {column_name}: {type(column_name)}")

    def get_column(self, column_name, table_name=None):
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

    def _get_tables(self, table_name):
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
                raise Exception(f"There was an existing table named {table_name}. Deleting it failed."
                                f" Reason: {reason}")
            result, reason = self.execute_raw(table_create_query)
            if result.upper() == "SUCCEEDED":
                return "SUCCEEDED"
            else:
                raise Exception(f"There was an existing table named {table_name} which is now successfully deleted,"
                                f"but new table failed to be created. Reason: {reason}")
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
            raise Exception('Batch query not completed yet.')

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
                raise Exception("Queries failed again. Sorry!")
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
        submitted_ids = []
        submitted_execution_ids = []
        submitted_queries = []
        queries_futures = []
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

    def get_distinct_vals(self, column: str, table_name: str = None, get_query_only: bool = False):
        table_name = self.bs_table.name if table_name is None else table_name
        tbl = self.get_table(table_name)
        query = sa.select(tbl.c[column]).distinct()
        if get_query_only:
            return self._compile(query)

        r = self.execute(query, run_async=False)
        return r[column]

    def get_distinct_count(self, column: str, table_name: str = None, weight_column: str = None,
                           get_query_only: bool = False):
        tbl = self.bs_table if table_name is None else self.get_table(table_name)
        query = sa.select([tbl.c[column], safunc.sum(1).label("sample_count"),
                           safunc.sum(self.sample_wt).label("weighted_count")])
        query = query.group_by(tbl.c[column]).order_by(tbl.c[column])
        if get_query_only:
            return self._compile(query)

        r = self.execute(query, run_async=False)
        return r

    def get_successful_simulation_count(self, restrict: List[Tuple[str, List]] = None, get_query_only: bool = False):
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

        restrict = list(restrict) if restrict else []
        restrict.insert(0, ('completed_status', ['Success']))
        query = self._add_restrict(query, restrict)
        if get_query_only:
            return self._compile(query)

        return self.execute(query)

    def get_results_csv(self, restrict: List[Tuple[str, Union[List, str, int]]] = None, get_query_only: bool = False):
        """
        Returns the results_csv table for the resstock run
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe that is a subset of the results csv, that belongs to provided list of utilities
        """
        restrict = list(restrict) if restrict else []
        query = sa.select(['*']).select_from(self.bs_table)
        query = self._add_restrict(query, restrict)
        compiled_query = self._compile(query)
        if get_query_only:
            return compiled_query
        if compiled_query in self._query_cache:
            return self._query_cache[compiled_query].copy().set_index(self.bs_bldgid_column.name)
        logger.info("Making results_csv query ...")
        return self.execute(query).set_index(self.bs_bldgid_column.name)

    def get_upgrades_csv(self, upgrade=None, restrict: List[Tuple[str, Union[List, str, int]]] = None,
                         get_query_only: bool = False):
        """
        Returns the results_csv table for the resstock run
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe that is a subset of the results csv, that belongs to provided list of utilities
        """
        restrict = list(restrict) if restrict else []
        query = sa.select(['*']).select_from(self.up_table)
        if upgrade:
            query = query.where(self.up_table.c['upgrade'] == str(upgrade))

        query = self._add_restrict(query, restrict)
        compiled_query = self._compile(query)
        if get_query_only:
            return compiled_query
        if compiled_query in self._query_cache:
            return self._query_cache[compiled_query].copy().set_index(self.bs_bldgid_column.name)
        logger.info("Making results_csv query for upgrade ...")
        return self.execute(query).set_index(self.bs_bldgid_column.name)

    def get_building_ids(self, restrict: List[Tuple[str, List]] = None, get_query_only: bool = False):
        """
        Returns the list of buildings based on the restrict list
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the building ids belonging to the provided list of locations.

        """
        restrict = list(restrict) if restrict else []
        query = sa.select(self.bs_bldgid_column)
        query = self._add_restrict(query, restrict)
        if get_query_only:
            return self._compile(query)
        res = self.execute(query)
        return res

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

    def _add_group_by(self, query, group_by):
        if group_by:
            slected_cols = [c.name for c in query.selected_columns]
            a = [sa.text(str(slected_cols.index(self._get_name(g)) + 1)) for g in group_by]
            query = query.group_by(*a)
        return query

    def _add_order_by(self, query, order_by):
        if order_by:
            slected_cols = [c.name for c in query.selected_columns]
            a = [sa.text(str(slected_cols.index(self._get_name(g)) + 1)) for g in order_by]
            query = query.order_by(*a)
        return query

    def _get_enduse_cols(self, enduses, table='baseline'):
        tbls_dict = {'baseline': self.bs_table,
                     'upgrade': self.up_table,
                     'timeseries': self.ts_table}
        tbl = tbls_dict[table]
        try:
            enduse_cols = [tbl.c[e] for e in enduses]
        except KeyError as e:
            if table in ['baseline', 'upgrade']:
                enduse_cols = [tbl.c[f"report_simulation_output.{e}"] for e in enduses]
            else:
                raise ValueError(f"Invalid enduse column names for {table} table") from e
        return enduse_cols

    def _get_weight(self, weights):
        total_weight = self.sample_wt
        for weight_col in weights:
            if isinstance(weight_col, tuple):
                tbl = self.get_table(weight_col[1])
                total_weight *= tbl.c[weight_col[0]]
            else:
                total_weight *= self.get_column(weight_col)
        return total_weight

    def get_groupby_cols(self) -> List[str]:
        cols = {y.removeprefix("build_existing_model.") for y in self.bs_table.c.keys()
                if y.startswith("build_existing_model.")}
        return list(cols)

    def validate_group_by(self, group_by):
        valid_groupby_cols = self.get_groupby_cols()
        group_by_cols = [g[0] if isinstance(g, tuple) else g for g in group_by]
        if not set(group_by_cols).issubset(valid_groupby_cols):
            invalid_cols = ", ".join(f'"{x}"' for x in set(group_by).difference(valid_groupby_cols))
            raise ValueError(f"The following are not valid columns in the database: {invalid_cols}")
        return group_by
        # TODO: intelligently select groupby columns order by cardinality (most to least groups) for
        # performance

    def get_available_upgrades(self) -> dict:
        """Get the available upgrade scenarios and their identifier numbers
        :return: Upgrade scenario names
        :rtype: dict
        """
        return list(self.get_success_report().query("Success>0").index)

    def validate_upgrade(self, upgrade_id):
        available_upgrades = self.get_available_upgrades()
        if upgrade_id not in set(available_upgrades):
            raise ValueError(f"`upgrade_id` = {upgrade_id} is not a valid upgrade."
                             "It doesn't exist or have no successful run")
        return str(upgrade_id)

    def _process_groupby_cols(self, group_by, annual_only=False):
        if not group_by:
            return []
        if annual_only:
            new_group_by = []
            for entry in group_by:
                if isinstance(entry, str) and not entry.startswith("build_existing_model."):
                    new_group_by.append("build_existing_model." + entry)
                elif isinstance(entry, tuple) and not entry[0].startswith("build_existing_model."):
                    new_group_by.append(("build_existing_model." + entry[0], entry[1]))
                else:
                    new_group_by.append(entry)
            group_by = new_group_by
        return [self._get_gcol(entry) for entry in group_by]

    def _split_restrict(self, restrict):
        # Some cols like "state" might be available in both ts and bs table
        bs_restrict = []  # restrict to apply to baseline table
        ts_restrict = []  # restrict to apply to timeseries table
        for col, restrict_vals in restrict:
            if col in self.ts_table.columns:  # prioritize ts table
                ts_restrict.append([self.ts_table.c[col], restrict_vals])
            else:
                bs_restrict.append([self._get_gcol(col), restrict_vals])
        return bs_restrict, ts_restrict

    def aggregate_annual(self,
                         enduses: List[str] = None,
                         group_by: List[str] = None,
                         sort: bool = False,
                         upgrade_id: str = None,
                         join_list: List[Tuple[str, str, str]] = None,
                         weights: List[Tuple] = None,
                         restrict: List[Tuple[str, List]] = None,
                         get_quartiles: bool = False,
                         run_async: bool = False,
                         get_query_only: bool = False):
        """
        Aggregates the baseline annual result on select enduses.
        Check the argument description below to learn about additional features and options.
        Args:
            enduses: The list of enduses to aggregate. Defaults to all electricity enduses

            group_by: The list of columns to group the aggregation by.

            sort: Whether to sort the results by group_by colummns

            upgrade_id: The upgrade to query for. Only valid with runs with upgrade.

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
            get_quartiles: If true, return the following quartiles in addition to the sum for each enduses:
                           [0, 0.02, .25, .5, .75, .98, 1]. The 0% quartile is the minimum and the 100% quartile
                           is the maximum.
            run_async: Whether to run the query in the background. Returns immediately if running in background,
                       blocks otherwise.
            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging


        Returns:
                if get_query_only is True, returns the query_string, otherwise,
                    if run_async is True, it returns a query_execution_id.
                    if run_async is False, it returns the result_dataframe

        """
        join_list = list(join_list) if join_list else []
        weights = list(weights) if weights else []
        restrict = list(restrict) if restrict else []

        [self.get_table(jl[0]) for jl in join_list]  # ingress all tables in join list
        if upgrade_id in {None, 0, '0'}:
            enduses = self._get_enduse_cols(enduses, table='baseline')
        else:
            upgrade_id = self.validate_upgrade(upgrade_id)
            enduses = self._get_enduse_cols(enduses, table='upgrade')

        total_weight = self._get_weight(weights)
        enduse_selection = [safunc.sum(enduse * total_weight).label(self._simple_label(enduse.name))
                            for enduse in enduses]
        if get_quartiles:
            enduse_selection += [sa.func.approx_percentile(enduse, [0, 0.02, 0.25, 0.5, 0.75, 0.98, 1]).
                                 label(self._simple_label(enduse.name)+"__quartiles") for enduse in enduses]
        grouping_metrics_selction = [safunc.sum(1).label("sample_count"),
                                     safunc.sum(total_weight).label("units_count")]

        if not group_by:
            query = sa.select(grouping_metrics_selction + enduse_selection)
            group_by_selection = []
        else:
            group_by_selection = self._process_groupby_cols(group_by, annual_only=True)
            query = sa.select(group_by_selection + grouping_metrics_selction + enduse_selection)
        # jj = self.bs_table.join(self.ts_table, self.ts_table.c['building_id']==self.bs_table.c['building_id'])
        # self._compile(query.select_from(jj))
        if upgrade_id not in [None, 0, '0']:
            tbljoin = self.bs_table.join(
                self.up_table, sa.and_(self.bs_table.c[self.building_id_column_name] ==
                                       self.up_table.c[self.building_id_column_name],
                                       self.up_table.c["upgrade"] == str(upgrade_id),
                                       self.up_table.c["completed_status"] == "Success"))
            query = query.select_from(tbljoin)

        restrict = [[self.bs_table.c['completed_status'], ('Success',)]] + restrict
        query = self._add_join(query, join_list)
        query = self._add_restrict(query, restrict)
        query = self._add_group_by(query, group_by_selection)
        query = self._add_order_by(query, group_by_selection if sort else [])

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

            if match := re.search(r'"[\w\.]*"\."([\w\.]*)"', col) or re.search(r'"([\w\.]*)"', col):
                new_group_by.append(match.group(1))
            else:
                new_group_by.append(col)
        return new_group_by

    def aggregate_timeseries_light(self,
                                   enduses: List[str] = None,
                                   group_by: List[str] = None,
                                   sort: bool = False,
                                   join_list: List[Tuple[str, str, str]] = None,
                                   weights: List[str] = None,
                                   restrict: List[Tuple[str, List]] = None,
                                   run_async: bool = False,
                                   get_query_only: bool = False,
                                   limit=None
                                   ):
        """
        Lighter version of aggregate_timeseries where each enduse is submitted as a separate query to be light on
        Athena. For information on the input parameters, check the documentation on aggregate_timeseries.
        """
        enduses = list(enduses) if enduses else []
        group_by = list(group_by) if group_by else []
        join_list = list(join_list) if join_list else []
        weights = list(weights) if weights else []
        restrict = list(restrict) if restrict else []

        if run_async:
            raise ValueError("Async run is not available for aggregate_timeseries_light since it needs to combine"
                             "the result after the query finishes.")

        enduses = self._get_enduse_cols(enduses, table='timeseries')
        print(enduses)
        batch_queries_to_submit = []
        for enduse in enduses:
            query = self.aggregate_timeseries(enduses=[enduse.name],
                                              group_by=group_by,
                                              sort=sort,
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
                             enduses: List[str] = None,
                             group_by: List[str] = None,
                             sort: bool = False,
                             join_list: List[Tuple[str, str, str]] = None,
                             weights: List[str] = None,
                             restrict: List[Tuple[str, List]] = None,
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
        enduses = list(enduses) if enduses else []
        group_by = list(group_by) if group_by else []
        join_list = list(join_list) if join_list else []
        weights = list(weights) if weights else []
        restrict = list(restrict) if restrict else []

        if split_enduses:
            return self.aggregate_timeseries_light(enduses=enduses, group_by=group_by, sort=sort,
                                                   join_list=join_list, weights=weights, restrict=restrict,
                                                   run_async=run_async, get_query_only=get_query_only,
                                                   limit=limit)
        [self.get_table(jl[0]) for jl in join_list]  # ingress all tables in join list
        enduses = self._get_enduse_cols(enduses, table='timeseries')
        total_weight = self._get_weight(weights)

        enduse_selection = [safunc.sum(enduse * total_weight).label(self._simple_label(enduse.name))
                            for enduse in enduses]
        group_by_selection = [self.get_column(g[0]).label(g[1]) if isinstance(
            g, tuple) else self.get_column(g) for g in group_by]

        if self.timestamp_column.name not in group_by:
            logger.info("Aggregation done accross timestamps. Result no longer a timeseries.")
            # The aggregation is done across time so we should compensate unit count by dividing by the total number
            # of distinct timestamps per unit
            rows_per_building = self._get_rows_per_building()
            grouping_metrics_selection = [(safunc.sum(1) / rows_per_building).label(
                "sample_count"), safunc.sum(total_weight / rows_per_building).label("units_count")]
        else:
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(total_weight).label("units_count")]

        query = sa.select(group_by_selection + grouping_metrics_selection + enduse_selection)
        query = query.join(self.bs_table, self.bs_bldgid_column == self.ts_bldgid_column)
        if join_list:
            query = self._add_join(query, join_list)

        query = self._add_restrict(query, restrict)
        query = self._add_group_by(query, group_by)
        query = self._add_order_by(query, group_by if sort else [])
        query = query.limit(limit) if limit else query

        if get_query_only:
            return self._compile(query)

        return self.execute(query, run_async=run_async)

    def _get_simulation_timesteps_count(self):
        # find the simulation time interval
        query = sa.select([self.ts_bldgid_column, safunc.sum(1).label('count')])
        query = query.group_by(self.ts_bldgid_column)
        sim_timesteps_count = self.execute(query)
        bld0_step_count = sim_timesteps_count['count'].iloc[0]
        n_buildings_with_same_count = sum(sim_timesteps_count['count'] == bld0_step_count)
        if n_buildings_with_same_count != len(sim_timesteps_count):
            logger.warning("Not all buildings have the same number of timestamps. This can cause wrong"
                           "scaled_units_count and other problems.")

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

        sim_year, sim_interval_seconds = self._get_simulation_info()
        kw_factor = 3600.0 / sim_interval_seconds

        enduse_selection = [safunc.avg(enduse * total_weight * kw_factor).label(self._simple_label(enduse.name))
                            for enduse in enduse_cols]
        grouping_metrics_selection = [safunc.sum(1).label("sample_count"),
                                      safunc.sum(total_weight).label("units_count")]

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
            return vals
        else:
            lower_vals, upper_vals = self.get_batch_query_result(batch_id, combine=False)
            avg_upper_weight = np.mean([min_of_hour / sim_interval_seconds for hour in at_hour if
                                        (min_of_hour := hour * 3600 % sim_interval_seconds)])
            avg_lower_weight = 1 - avg_upper_weight
            # modify the lower vals to make it weighted average of upper and lower vals
            lower_vals[enduses] = lower_vals[enduses] * avg_lower_weight + upper_vals[enduses] * avg_upper_weight
            return lower_vals
