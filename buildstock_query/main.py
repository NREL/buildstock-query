import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
from typing import List, Union, Sequence
import logging
import re
from buildstock_query.tools import UpgradesAnalyzer
from buildstock_query.query_core import QueryCore
import pandas as pd
from pydantic import validate_arguments, Field
from typing import Optional, Literal
from typing_extensions import assert_never
import typing
from datetime import datetime
from buildstock_query.schema.run_params import BSQParams
from buildstock_query.schema.utilities import DBColType, SALabel, AnyColType
from buildstock_query.schema.utilities import MappedColumn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FUELS = ['electricity', 'natural_gas', 'propane', 'fuel_oil', 'coal', 'wood_cord', 'wood_pellets']


class BuildStockQuery(QueryCore):

    @validate_arguments(config=dict(smart_union=True))
    def __init__(self,
                 workgroup: str,
                 db_name: str,
                 table_name: Union[str, tuple[str, Optional[str], Optional[str]]],
                 buildstock_type: Literal['resstock', 'comstock'] = 'resstock',
                 timestamp_column_name: str = 'time',
                 building_id_column_name: str = 'building_id',
                 sample_weight: Union[int, float, str] = "build_existing_model.sample_weight",
                 region_name: str = 'us-west-2',
                 execution_history: Optional[str] = None,
                 skip_reports: bool = False,
                 ) -> None:
        """A class to run Athena queries for BuildStock runs and download results as pandas DataFrame.

        Args:
            workgroup (str): The workgroup for athena. The cost will be charged based on workgroup.
            db_name (str): The athena database name
            buildstock_type (str, optional): 'resstock' or 'comstock' runs. Defaults to 'resstock'
            table_name (str or Union[str, tuple[str, Optional[str], Optional[str]]]): If a single string is provided,
            say, 'mfm_run', then it must correspond to tables in athena named mfm_run_baseline and optionally
            mfm_run_timeseries and mf_run_upgrades. Or, tuple of three elements can be privided for the table names
            for baseline, timeseries and upgrade. Timeseries and upgrade can be None if no such table exist.
            timestamp_column_name (str, optional): The column name for the time column. Defaults to 'time'
            building_id_column_name (str, optional): The column name for building_id. Defaults to 'building_id'
            sample_weight (str, optional): The column name to be used to get the sample weight. Pass floats/integer to
                use constant sample weight.. Defaults to "build_existing_model.sample_weight".
            region_name (str, optional): the AWS region where the database exists. Defaults to 'us-west-2'.
            execution_history (str, optional): A temporary files to record which execution is run by the user,
                to help stop them. Will use .execution_history if not supplied.
            skip_reports (bool, optional): If true, skips report printing during initialization. If False (default),
                prints report from `buildstock_query.report_query.BuildStockReport.get_success_report`.
        """
        self.params = BSQParams(
            workgroup=workgroup,
            db_name=db_name,
            buildstock_type=buildstock_type,
            table_name=table_name,
            timestamp_column_name=timestamp_column_name,
            building_id_column_name=building_id_column_name,
            sample_weight=sample_weight,
            region_name=region_name,
            execution_history=execution_history
        )
        self.run_params = self.params.get_run_params()
        from buildstock_query.report_query import BuildStockReport
        from buildstock_query.aggregate_query import BuildStockAggregate
        from buildstock_query.savings_query import BuildStockSavings
        from buildstock_query.utility_query import BuildStockUtility

        super().__init__(params=self.run_params)
        #: `buildstock_query.report_query.BuildStockReport` object to perform report queries
        self.report: BuildStockReport = BuildStockReport(self)
        #: `buildstock_query.aggregate_query.BuildStockAggregate` object to perform aggregate queries
        self.agg: BuildStockAggregate = BuildStockAggregate(self)
        #: `buildstock_query.savings_query.BuildStockSavings` object to perform savings queries
        self.savings = BuildStockSavings(self)
        #: `buildstock_query.utility_query.BuildStockUtility` object to perform utility queries
        self.utility = BuildStockUtility(self)

        if not skip_reports:
            logger.info("Getting Success counts...")
            print(self.report.get_success_report())
            if self.ts_table is not None:
                self.report.check_ts_bs_integrity()
            self.save_cache()

    def get_buildstock_df(self) -> pd.DataFrame:
        """Returns the building characteristics data by quering Athena tables using the same format as that produced
        by the sampler and written as buildstock.csv. It only includes buildings with successful simulation.
        Returns:
            pd.DataFrame: The buildstock.csv dataframe.
        """
        results_df = self.get_results_csv()
        results_df = results_df[results_df["completed_status"] == "Success"]
        buildstock_cols = [c for c in results_df.columns if c.startswith("build_existing_model.")]
        buildstock_df = results_df[buildstock_cols]
        buildstock_cols = [''.join(c.split(".")[1:]).replace("_", " ") for c in buildstock_df.columns
                           if c.startswith("build_existing_model.")]
        buildstock_df.columns = buildstock_cols
        return buildstock_df

    @validate_arguments
    def get_upgrades_analyzer(self, yaml_file: str, opt_sat_file: str) -> UpgradesAnalyzer:
        """
            Returns the UpgradesAnalyzer object with buildstock.csv downloaded from athena (see get_buildstock_df help)

        Args:
            yaml_file (str): The path to the buildstock configuration file.
            opt_sat_file (str): The path to the opt_saturation.csv file for the housing characteristics.

        Returns:
            UpgradesAnalyzer: returns UpgradesAnalyzer object. See UpgradesAnalyzer.
        """

        buildstock_df = self.get_buildstock_df()
        ua = UpgradesAnalyzer(buildstock=buildstock_df, yaml_file=yaml_file, opt_sat_file=opt_sat_file)
        return ua

    @typing.overload
    def _get_rows_per_building(self, get_query_only: Literal[False] = False) -> int:
        ...

    @typing.overload
    def _get_rows_per_building(self, get_query_only: Literal[True]) -> str:
        ...

    @validate_arguments
    def _get_rows_per_building(self, get_query_only: bool = False) -> Union[int, str]:
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

    @validate_arguments(config=dict(smart_union=True))
    def get_distinct_vals(self, column: str, table_name: Optional[str],
                          get_query_only: bool = False) -> Union[str, pd.Series]:
        """
            Find distinct vals.
        Args:
            column (str): The column in the table for which distinct vals is needed.
            table_name (str, optional): The table in athena. Defaults to baseline table.
            get_query_only (bool, optional): If true, only returns the SQL query. Defaults to False.

        Returns:
            pd.Series: The distinct vals.
        """
        table_name = self.bs_table.name if table_name is None else table_name
        tbl = self.get_table(table_name)
        query = sa.select(tbl.c[column]).distinct()
        if get_query_only:
            return self._compile(query)

        r = self.execute(query, run_async=False)
        return r[column]

    @validate_arguments(config=dict(smart_union=True))
    def get_distinct_count(self, column: str, table_name: Optional[str] = None,
                           get_query_only: bool = False) -> Union[pd.DataFrame, str]:
        """
            Find distinct counts.
        Args:
            column (str): The column in the table for which distinct counts is needed.
            table_name (str, optional): The table in athena. Defaults to baseline table.
            get_query_only (bool, optional): If true, only returns the SQL query. Defaults to False.

        Returns:
            pd.Series: The distinct counts.
        """
        tbl = self.bs_table if table_name is None else self.get_table(table_name)
        query = sa.select([tbl.c[column], safunc.sum(1).label("sample_count"),
                           safunc.sum(self.sample_wt).label("weighted_count")])
        query = query.group_by(tbl.c[column]).order_by(tbl.c[column])
        if get_query_only:
            return self._compile(query)

        r = self.execute(query, run_async=False)
        return r

    @typing.overload
    def get_results_csv(self, *,
                        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
                            default_factory=list),
                        get_query_only: Literal[False] = False) -> pd.DataFrame:
        ...

    @typing.overload
    def get_results_csv(self, *,
                        get_query_only: Literal[True],
                        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
                            default_factory=list),
                        ) -> str:
        ...

    @typing.overload
    def get_results_csv(self, *,
                        get_query_only: bool,
                        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
                            default_factory=list),
                        ) -> Union[str, pd.DataFrame]:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True, smart_union=True))
    def get_results_csv(self,
                        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
                            default_factory=list),
                        get_query_only: bool = False) -> Union[pd.DataFrame, str]:
        """
        Returns the results_csv table for the BuildStock run
        Args:
            restrict (List[Tuple[str, Union[List, str, int]]], optional): The list of where condition to restrict the
                results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only (bool): If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe that is a subset of the results csv, that belongs to provided list of utilities
        """
        restrict = list(restrict) if restrict else []
        query = sa.select(['*']).select_from(self.bs_table)
        query = self._add_restrict(query, restrict, bs_only=True)
        compiled_query = self._compile(query)
        if get_query_only:
            return compiled_query
        self._session_queries.add(compiled_query)
        if compiled_query in self._query_cache:
            return self._query_cache[compiled_query].copy().set_index(self.bs_bldgid_column.name)
        logger.info("Making results_csv query ...")
        result = self.execute(query)
        return result.set_index(self.bs_bldgid_column.name)

    @typing.overload
    def get_upgrades_csv(self, *, get_query_only: Literal[False] = False, upgrade_id: Union[int, str] = '0',
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
        default_factory=list)
    ) -> pd.DataFrame:
        ...

    @typing.overload
    def get_upgrades_csv(self, *, get_query_only: Literal[True], upgrade_id: Union[int, str] = '0',
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
        default_factory=list)
    ) -> str:
        ...

    @typing.overload
    def get_upgrades_csv(self, *, get_query_only: bool, upgrade_id: Union[int, str] = '0',
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
        default_factory=list)
    ) -> Union[pd.DataFrame, str]:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True, smart_union=True))
    def get_upgrades_csv(self, *, upgrade_id: Union[str, int] = '0',
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
            default_factory=list),
            get_query_only: bool = False) -> Union[pd.DataFrame, str]:
        """
        Returns the results_csv table for the BuildStock run for an upgrade.
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe that is a subset of the results csv, that belongs to provided list of utilities
        """
        restrict = list(restrict) if restrict else []
        query = sa.select(['*']).select_from(self.up_table)
        if upgrade_id:
            if self.up_table is None:
                raise ValueError("This run has no upgrades")
            query = query.where(self.up_table.c['upgrade'] == str(upgrade_id))

        query = self._add_restrict(query, restrict, bs_only=True)
        compiled_query = self._compile(query)
        if get_query_only:
            return compiled_query
        self._session_queries.add(compiled_query)
        if compiled_query in self._query_cache:
            return self._query_cache[compiled_query].copy().set_index(self.bs_bldgid_column.name)
        logger.info("Making results_csv query for upgrade ...")
        return self.execute(query).set_index(self.bs_bldgid_column.name)

    @typing.overload
    def get_building_ids(self, *,
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
                             default_factory=list),
                         get_query_only: Literal[False] = False
                         ) -> pd.DataFrame:
        ...

    @typing.overload
    def get_building_ids(self, *,
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
                             default_factory=list),
                         get_query_only: Literal[True]
                         ) -> str:
        ...

    @typing.overload
    def get_building_ids(self, *,
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
                             default_factory=list),
                         get_query_only: bool
                         ) -> Union[pd.DataFrame, str]:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True, smart_union=True))
    def get_building_ids(self,
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(
                             default_factory=list),
                         get_query_only: bool = False
                         ) -> Union[str, pd.DataFrame]:
        """
        Returns the list of buildings based on the restrict list
        Args:
            restrict (List[Tuple[str, List]], optional): The list of where condition to restrict the results to. It
                    should be specified as a list of tuple.
                    Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only (bool): If set to true, returns the query string instead of the result. Default is False.

        Returns:
            Pandas dataframe consisting of the building ids belonging to the provided list of locations.

        """
        restrict = list(restrict) if restrict else []
        query = sa.select(self.bs_bldgid_column)
        query = self._add_restrict(query, restrict, bs_only=True)
        if get_query_only:
            return self._compile(query)
        return self.execute(query)

    @typing.overload
    def _get_simulation_info(self, get_query_only: Literal[False] = False) -> tuple[int, int, int]:
        ...

    @typing.overload
    def _get_simulation_info(self, get_query_only: Literal[True]) -> str:
        ...

    @validate_arguments(config=dict(smart_union=True))
    def _get_simulation_info(self, get_query_only: bool = False) -> Union[str, tuple[int, int, int]]:
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
        reference_time = datetime(year=sim_year, month=1, day=1)
        sim_interval_seconds = (time2 - time1).total_seconds()
        start_offset_seconds = int((time1 - reference_time).total_seconds())
        return sim_year, sim_interval_seconds, start_offset_seconds

    def get_special_column(self,
                           column_type: Literal['month', 'day', 'hour', 'is_weekend', 'day_of_week']) -> DBColType:
        _, _, start_offset = self._get_simulation_info()
        if start_offset > 0:
            # If timestamps are not period begining we should make them so we get proper values of special columns.
            time_col = sa.func.date_add('second', -start_offset, self.timestamp_column)
        else:
            time_col = self.timestamp_column

        if column_type == 'month':
            return sa.func.month(time_col).label('month')
        elif column_type == 'day':
            return sa.func.day(time_col).label('day')
        elif column_type == 'hour':
            return sa.func.hour(time_col).label('hour')
        elif column_type == 'day_of_week':
            return sa.func.day_of_week(time_col).label('day_of_week')
        elif column_type == 'is_weekend':
            return sa.cast(sa.func.day_of_week(time_col).in_([6, 7]), sa.Integer).label('is_weekend')
        else:
            assert_never(column_type)
            raise ValueError(f"Unknown special column type: {column_type}")

    def _get_gcol(self, column) -> DBColType:  # gcol => group by col
        """Get a DB column for the purpose of grouping. If the provided column doesn't exist as is,
        tries to get the column by prepending build_existing_model."""

        if isinstance(column, sa.Column):
            return column.label(self._simple_label(column.name))  # already a col

        if isinstance(column, SALabel):
            return column

        if isinstance(column, MappedColumn):
            return sa.literal(column).label(self._simple_label(column.name))

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

    def _get_enduse_cols(self, enduses: Sequence[AnyColType],
                         table='baseline') -> Sequence[DBColType]:
        tbls_dict = {'baseline': self.bs_table,
                     'upgrade': self.up_table,
                     'timeseries': self.ts_table}
        tbl = tbls_dict[table]
        enduse_cols: list[DBColType] = []
        for enduse in enduses:
            if isinstance(enduse, (sa.Column, SALabel)):
                enduse_cols.append(enduse)
            elif isinstance(enduse, str):
                try:
                    enduse_cols.append(tbl.c[enduse])
                except KeyError as err:
                    if table in ['baseline', 'upgrade']:
                        enduse_cols.append(tbl.c[f"report_simulation_output.{enduse}"])
                    else:
                        raise ValueError(f"Invalid enduse column names for {table} table") from err
            elif isinstance(enduse, MappedColumn):
                enduse_cols.append(sa.literal(enduse).label(enduse.name))
            else:
                assert_never(enduse)
        return enduse_cols

    def get_groupby_cols(self) -> List[str]:
        """Find list of building characteristics that can be used for grouping.

        Returns:
            List[str]: List of building characteristics.
        """
        cols = {y.removeprefix("build_existing_model.") for y in self.bs_table.c.keys()
                if y.startswith("build_existing_model.")}
        return list(cols)

    def _validate_group_by(self, group_by: Sequence[Union[str, tuple[str, str]]]):
        valid_groupby_cols = self.get_groupby_cols()
        group_by_cols = [g[0] if isinstance(g, tuple) else g for g in group_by]
        if not set(group_by_cols).issubset(valid_groupby_cols):
            invalid_cols = ", ".join(f'"{x}"' for x in set(group_by).difference(valid_groupby_cols))
            raise ValueError(f"The following are not valid columns in the database: {invalid_cols}")
        return group_by
        # TODO: intelligently select groupby columns order by cardinality (most to least groups) for
        # performance

    def get_available_upgrades(self) -> Sequence[str]:
        """Get the available upgrade scenarios and their identifier numbers.
        Returns:
            list: List of upgrades
        """
        return list([str(u) for u in self.report.get_success_report().query("Success>0").index])

    def _validate_upgrade(self, upgrade_id: Union[int, str]) -> str:
        upgrade_id = '0' if upgrade_id in (None, '0') else str(upgrade_id)
        available_upgrades = self.get_available_upgrades() or ['0']
        if upgrade_id not in set(available_upgrades):
            raise ValueError(f"`upgrade_id` = {upgrade_id} is not a valid upgrade."
                             "It doesn't exist or have no successful run")
        return str(upgrade_id)

    def _split_restrict(self, restrict):
        # Some cols like "state" might be available in both ts and bs table
        bs_restrict = []  # restrict to apply to baseline table
        ts_restrict = []  # restrict to apply to timeseries table
        for col, restrict_vals in restrict:
            if self.ts_table is not None and col in self.ts_table.columns:  # prioritize ts table
                ts_restrict.append([self.ts_table.c[col], restrict_vals])
            else:
                bs_restrict.append([self._get_gcol(col), restrict_vals])
        return bs_restrict, ts_restrict

    def _split_group_by(self, processed_group_by):
        # Some cols like "state" might be available in both ts and bs table
        ts_group_by = []  # restrict to apply to baseline table
        bs_group_by = []  # restrict to apply to timeseries table
        for g in processed_group_by:
            if self.ts_table is not None and g.name in self.ts_table.columns:
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

    @typing.overload
    def get_buildings_by_locations(self, location_col: str, locations: List[str],
                                   get_query_only: Literal[False] = False) -> pd.DataFrame:
        ...

    @typing.overload
    def get_buildings_by_locations(self, location_col: str, locations: List[str],
                                   get_query_only: Literal[True]) -> str:
        ...

    @typing.overload
    def get_buildings_by_locations(self, location_col: str, locations: List[str],
                                   get_query_only: bool) -> Union[str, pd.DataFrame]:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True, smart_union=True))
    def get_buildings_by_locations(self, location_col: str, locations: List[str],
                                   get_query_only: bool = False) -> Union[str, pd.DataFrame]:
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
