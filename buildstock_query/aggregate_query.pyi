from typing import Literal, Callable
from collections.abc import Sequence
import pandas as pd
import typing
from buildstock_query.schema.query_params import TSQuery, BaseQuery, Query
from buildstock_query import main
from buildstock_query.schema.utilities import AnyColType, AnyTableType
from pydantic import Field

class BuildStockAggregate:
    def __init__(self, buildstock_query: main.BuildStockQuery) -> None: ...
    @typing.overload
    def aggregate_annual(
        self,
        *,
        enduses: Sequence[AnyColType],
        get_query_only: Literal[True],
        group_by: Sequence[AnyColType | tuple[str, str]] = [],
        sort: bool = False,
        upgrade_id: int | str = "0",
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
        weights: Sequence[str | tuple] = [],
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        get_quartiles: bool = False,
        get_nonzero_count: bool = False,
        agg_func: str | Callable | None = "sum",
    ) -> str: ...
    @typing.overload
    def aggregate_annual(
        self,
        *,
        enduses: Sequence[AnyColType],
        get_query_only: Literal[False] = False,
        group_by: Sequence[AnyColType | tuple[str, str]] = [],
        sort: bool = False,
        upgrade_id: int | str = "0",
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
        weights: Sequence[str | tuple] = [],
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        get_quartiles: bool = False,
        get_nonzero_count: bool = False,
        agg_func: str | Callable | None = "sum",
    ) -> pd.DataFrame: ...
    @typing.overload
    def aggregate_annual(
        self,
        *,
        enduses: Sequence[AnyColType],
        get_query_only: bool,
        group_by: Sequence[AnyColType | tuple[str, str]] = [],
        sort: bool = False,
        upgrade_id: int | str = "0",
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
        weights: Sequence[str | tuple] = [],
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        get_quartiles: bool = False,
        get_nonzero_count: bool = False,
        agg_func: str | Callable | None = "sum",
    ) -> pd.DataFrame | str:
        """
        Aggregates the baseline annual result on select enduses.
        Check the argument description below to learn about additional features and options.
        Args:
            enduses: The list of enduses to aggregate. Defaults to all electricity enduses

            group_by: The list of columns to group the aggregation by.

            sort: Whether to sort the results by group_by columns

            upgrade_id: The upgrade to query for. Only valid with runs with upgrade. If not provided, use the baseline

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
            avoid: Just like restrict, but the opposite. It will only include rows that do not match (any of) the
                   conditions.
            get_quartiles: If true, return the following quartiles in addition to the sum for each enduses:
                        [0, 0.02, .25, .5, .75, .98, 1]. The 0% quartile is the minimum and the 100% quartile
                        is the maximum.
            get_nonzero_count: If true, return the number of non-zero rows for each enduses. Useful, for example, for
                        finding the number of natural gas customers by using natural gas total fuel use as the enduse.

            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging

            agg_func: The aggregation function to use. Defaults to 'sum'.
                      See other options in https://prestodb.io/docs/current/functions/aggregate.html

        Returns:
                if get_query_only is True, returns the query_string, otherwise returns the dataframe
        """

    @typing.overload
    def aggregate_annual(self, *, params: BaseQuery) -> str | pd.DataFrame: ...
    @typing.overload
    def aggregate_timeseries(
        self,
        *,
        enduses: Sequence[AnyColType],
        get_query_only: Literal[True],
        group_by: Sequence[AnyColType | tuple[str, str]] = [],
        upgrade_id: int | str = "0",
        sort: bool = False,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
        weights: Sequence[str | tuple] = [],
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        split_enduses: bool = False,
        collapse_ts: bool = False,
        timestamp_grouping_func: str | None = None,
        limit: int | None = None,
        agg_func: str | Callable | None = "sum",
    ) -> str: ...
    @typing.overload
    def aggregate_timeseries(
        self,
        *,
        enduses: Sequence[AnyColType],
        group_by: Sequence[AnyColType | tuple[str, str]] = [],
        upgrade_id: int | str = "0",
        sort: bool = False,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
        weights: Sequence[str | tuple] = [],
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        split_enduses: bool = False,
        collapse_ts: bool = False,
        timestamp_grouping_func: str | None = None,
        get_query_only: Literal[False] = False,
        limit: int | None = None,
        agg_func: str | Callable | None = "sum",
    ) -> pd.DataFrame: ...
    @typing.overload
    def aggregate_timeseries(
        self,
        *,
        enduses: Sequence[AnyColType],
        get_query_only: bool,
        group_by: Sequence[AnyColType | tuple[str, str]] = [],
        upgrade_id: int | str = "0",
        sort: bool = False,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
        weights: Sequence[str | tuple] = [],
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = [],
        split_enduses: bool = False,
        collapse_ts: bool = False,
        timestamp_grouping_func: str | None = None,
        limit: int | None = None,
        agg_func: str | Callable | None = "sum",
    ) -> str | pd.DataFrame:
        """
        Aggregates the timeseries result on select enduses.
        Check the argument description below to learn about additional features and options.
        Args:
        enduses: The list of enduses to aggregate. Defaults to all electricity enduses

        group_by: The list of columns to group the aggregation by.

        upgrade_id: The upgrade to query for. Only valid with runs with upgrade. If not provided, use the baseline

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

        split_enduses: Whether to query for each enduses in a separate query to reduce Athena load for query. Useful
                        when Athena runs into "Query exhausted resources ..." errors.
        timestamp_grouping_func: One of 'hour', 'day' or 'month' or None. If provided, perform timeseries
                                aggregation of specified granularity.
        get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                        submitting multiple queries or debugging

        agg_func: The aggregation function to use. Defaults to 'sum'.
                  See other options in https://prestodb.io/docs/current/functions/aggregate.html
        Returns:
                if get_query_only is True, returns the query_string, otherwise, returns the DataFrame

        """

    @typing.overload
    def aggregate_timeseries(
        self,
        *,
        params: TSQuery,
    ) -> str | pd.DataFrame: ...
    @typing.overload
    def get_building_average_kws_at(
        self,
        at_hour: Sequence[float] | float,
        at_days: Sequence[float],
        enduses: Sequence[str],
        get_query_only: Literal[False] = False,
    ) -> pd.DataFrame: ...
    @typing.overload
    def get_building_average_kws_at(
        self,
        at_hour: Sequence[float] | float,
        at_days: Sequence[float],
        enduses: Sequence[str],
        get_query_only: Literal[True],
    ) -> str: ...
    @typing.overload
    def query(
        self,
        *,
        get_query_only: Literal[True],
        upgrade_id: int | str = "0",
        enduses: Sequence[AnyColType],
        group_by: Sequence[AnyColType | tuple[str, str]] = Field(default_factory=list),
        annual_only: bool = True,
        include_upgrade: bool = True,
        include_savings: bool = False,
        include_baseline: bool = False,
        sort: bool = True,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list),
        weights: Sequence[str | tuple] = Field(default_factory=list),
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        applied_only: bool = False,
        get_quartiles: bool = False,
        get_nonzero_count: bool = False,
        unload_to: str = "",
        partition_by: Sequence[str] | None = None,
        timestamp_grouping_func: str | None = None,
        limit: int | None = None,
        agg_func: str | Callable | None = "sum",
    ) -> str: ...
    @typing.overload
    def query(
        self,
        *,
        upgrade_id: int | str = "0",
        get_query_only: Literal[False] = False,
        enduses: Sequence[AnyColType],
        group_by: Sequence[AnyColType | tuple[str, str]] = Field(default_factory=list),
        annual_only: bool = True,
        include_upgrade: bool = True,
        include_savings: bool = False,
        include_baseline: bool = False,
        sort: bool = True,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list),
        weights: Sequence[str | tuple] = Field(default_factory=list),
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        applied_only: bool = False,
        get_quartiles: bool = False,
        get_nonzero_count: bool = False,
        unload_to: str = "",
        partition_by: Sequence[str] | None = None,
        timestamp_grouping_func: str | None = None,
        limit: int | None = None,
        agg_func: str | Callable | None = "sum",
    ) -> pd.DataFrame: ...
    @typing.overload
    def query(
        self,
        *,
        get_query_only: bool,
        upgrade_id: int | str = "0",
        enduses: Sequence[AnyColType],
        group_by: Sequence[AnyColType | tuple[str, str]] = Field(default_factory=list),
        annual_only: bool = True,
        include_upgrade: bool = True,
        include_savings: bool = False,
        include_baseline: bool = False,
        sort: bool = True,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list),
        weights: Sequence[str | tuple] = Field(default_factory=list),
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        applied_only: bool = False,
        get_quartiles: bool = False,
        get_nonzero_count: bool = False,
        unload_to: str = "",
        partition_by: Sequence[str] | None = None,
        timestamp_grouping_func: str | None = None,
        limit: int | None = None,
        agg_func: str | Callable | None = "sum",
    ) -> str | pd.DataFrame:
        """Calculate savings shape for an upgrade
        Args:
            upgrade_id: id of the upgrade scenario from the ResStock analysis
            enduses: Enduses to query, defaults to ['fuel_use__electricity__total']
            group_by: Building characteristics columns to group by, defaults to []
            annual_only: If true, calculates only the annual savings using baseline and upgrades table
            sort: Whether the result should be sorted. Sorting takes extra time.
            join_list: Additional table to join to baseline table to perform operation. All the inputs (`enduses`,
                  `group_by` etc) can use columns from these additional tables. It should be specified as a list of
                  tuples.
                  Example: `[(new_table_name, baseline_column_name, new_column_name), ...]`
                        where baseline_column_name and new_column_name are the columns on which the new_table
                        should be joined to baseline table.
            applied_only: Calculate savings shape based on only buildings to which the upgrade applied
            weights: The additional columns to use as weight. The "build_existing_model.sample_weight" is already used.
                     It is specified as either list of string or list of tuples. When only string is used, the string
                     is the column name, when tuple is passed, the second element is the table name.

            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`

            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging
            get_quartiles: If true, return the following quartiles in addition to the sum for each enduses:
                           [0, 0.02, .25, .5, .75, .98, 1]. The 0% quartile is the minimum and the 100% quartile
                           is the maximum.
            unload_to: Writes the output of the query to this location in s3. Consider using run_async = True with this
                       to unload multiple queries simulataneuosly
            partition_by: List of columns to partition when writing to s3. To be used with unload_to.
            timestamp_grouping_func: One of 'hour', 'day' or 'month' or 'year' or None. If provided, perform timeseries
                        aggregation of specified granularity. For 'year' - it collapses the timeseries into a single
                        annual value. Useful for quality checking or finding the annual max and min.
         Returns:
                if get_query_only is True, returns the query_string, otherwise returns a pandas dataframe
        """

    @typing.overload
    def query(self, *, params: Query) -> str | pd.DataFrame: ...
