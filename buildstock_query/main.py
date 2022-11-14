"""
# ResStockAthena
- - - - - - - - -
A class to run AWS Athena queries to get various data from a ResStock run. All queries and aggregation that can be
common accross different ResStock projects should be implemented in this class. For queries that are project specific, a
new class can be created by inheriting ResStockAthena and adding in the project specific logic and queries.

:author: Rajendra.Adhikari@nrel.gov
"""


import contextlib
import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
from typing import List, Tuple, Union
import logging
import re
from buildstock_query.tools.upgrades_analyzer import UpgradesAnalyzer
from buildstock_query.query_core import QueryCore
from buildstock_query.report_query import BuildStockReport
from buildstock_query.aggregate_query import BuildStockAggregate
from buildstock_query.savings_query import BuildStockSavings
from buildstock_query.utility_query import BuildStockUtility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FUELS = ['electricity', 'natural_gas', 'propane', 'fuel_oil', 'coal', 'wood_cord', 'wood_pellets']


class BuildStockQuery(QueryCore):
    def __init__(self, skip_reports: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.report = BuildStockReport(self)
        self.agg = BuildStockAggregate(self)
        self.savings = BuildStockSavings(self)
        self.utility = BuildStockUtility(self)

        with contextlib.suppress(FileNotFoundError):
            self.load_cache()

        if not skip_reports:
            logger.info("Getting Success counts...")
            print(self.report.get_success_report())
            if self.ts_table is not None:
                self.report.check_ts_bs_integrity()
        self.save_cache()

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

    def _get_rows_per_building(self, get_query_only=False):
        select_cols = []
        if self.up_table is not None:
            select_cols.append(self.ts_table.c['upgrade'])
        select_cols.extend((self.ts_bldgid_column, safunc.count().label("row_count")))
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

    def get_distinct_vals(self, column: str, table_name: str | None = None, get_query_only: bool = False):
        table_name = self.bs_table.name if table_name is None else table_name
        tbl = self.get_table(table_name)
        query = sa.select(tbl.c[column]).distinct()
        if get_query_only:
            return self._compile(query)

        r = self.execute(query, run_async=False)
        return r[column]

    def get_distinct_count(self, column: str, table_name: str | None = None, weight_column: str | None = None,
                           get_query_only: bool = False):
        tbl = self.bs_table if table_name is None else self.get_table(table_name)
        query = sa.select([tbl.c[column], safunc.sum(1).label("sample_count"),
                           safunc.sum(self.sample_wt).label("weighted_count")])
        query = query.group_by(tbl.c[column]).order_by(tbl.c[column])
        if get_query_only:
            return self._compile(query)

        r = self.execute(query, run_async=False)
        return r

    def get_results_csv(self,
                        restrict: List[Tuple[str, Union[List, str, int]]] | None = None,
                        get_query_only: bool = False):
        """
        Returns the results_csv table for the BuildStock run
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
        self._session_queries.add(compiled_query)
        if compiled_query in self._query_cache:
            return self._query_cache[compiled_query].copy().set_index(self.bs_bldgid_column.name)
        logger.info("Making results_csv query ...")
        return self.execute(query).set_index(self.bs_bldgid_column.name)

    def get_upgrades_csv(self, upgrade=None,
                         restrict: List[Tuple[str, Union[List, str, int]]] | None = None,
                         get_query_only: bool = False, copy=True):
        """
        Returns the results_csv table for the BuildStock run
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
        self._session_queries.add(compiled_query)
        if compiled_query in self._query_cache:
            return self._query_cache[compiled_query].copy().set_index(self.bs_bldgid_column.name)
        logger.info("Making results_csv query for upgrade ...")
        return self.execute(query).set_index(self.bs_bldgid_column.name)

    def get_building_ids(self, restrict: List[Tuple[str, List]] | None = None, get_query_only: bool = False):
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

    def _get_gcol(self, column):  # gcol => group by col
        if isinstance(column, sa.Column):
            return column.label(self._simple_label(column.name))  # already a col

        if isinstance(column, sa.sql.elements.Label):
            return column

        if isinstance(column, tuple):
            try:
                return self.get_column(column[0]).label(column[1])
            except ValueError:
                new_name = f"build_existing_model.{column[0]}"
                return self.get_column(new_name).label(column[1])
        elif isinstance(column, str):
            try:
                return self.get_column(column).label(self._simple_label(column))
            except ValueError as e:
                if not column.startswith("build_existing_model."):
                    new_name = f"build_existing_model.{column}"
                    return self.get_column(new_name).label(column)
                raise ValueError(f"Invalid column name {column}") from e
        else:
            raise ValueError(f"Invalid column name type {column}: {type(column)}")

    def _get_enduse_cols(self, enduses, table='baseline') -> list[sa.Column]:
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

    def get_available_upgrades(self) -> list:
        """Get the available upgrade scenarios and their identifier numbers
        :return: Upgrade scenario names
        :rtype: dict
        """
        return list(self.report.get_success_report().query("Success>0").index)

    def validate_upgrade(self, upgrade_id):
        upgrade_id = 0 if upgrade_id in (None, '0') else upgrade_id
        available_upgrades = self.get_available_upgrades() or [0]
        if upgrade_id not in set(available_upgrades):
            raise ValueError(f"`upgrade_id` = {upgrade_id} is not a valid upgrade."
                             "It doesn't exist or have no successful run")
        return str(upgrade_id)

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

    def _split_group_by(self, processed_group_by):
        # Some cols like "state" might be available in both ts and bs table
        ts_group_by = []  # restrict to apply to baseline table
        bs_group_by = []  # restrict to apply to timeseries table
        for g in processed_group_by:
            if g.name in self.ts_table.columns:
                ts_group_by.append(g)
            else:
                bs_group_by.append(g)
        return bs_group_by, ts_group_by

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

    def _process_groupby_cols(self, group_by, annual_only=False):
        if not group_by:
            return []
        if annual_only:
            new_group_by = []
            for entry in group_by:
                if isinstance(entry, str) and not entry.startswith("build_existing_model."):
                    new_group_by.append(f"build_existing_model.{entry}")
                elif isinstance(entry, tuple) and not entry[0].startswith("build_existing_model."):
                    new_group_by.append((f"build_existing_model.{entry[0]}", entry[1]))
                else:
                    new_group_by.append(entry)
            group_by = new_group_by
        return [self._get_gcol(entry) for entry in group_by]

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

    def get_buildings_by_locations(self, location_col, locations: List[str], get_query_only: bool = False):
        """
        Returns the list of buildings belonging to given list of locations.
        Args:
            location_col: The column used for "build_existing_model.county" etc
            locations: list of `build_existing_model.location' strings
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the building ids belonging to the provided list of locations.

        """
        query = sa.select([self.bs_bldgid_column])
        query = query.where(self.get_column(location_col).in_(locations))
        query = self._add_order_by(query, [self.bs_bldgid_column])
        if get_query_only:
            return self._compile(query)
        res = self.execute(query)
        return res
