from typing import Optional, Union, Literal, Sequence
import pandas as pd
import typing
from buildstock_query.schema import TSQuery, AnnualQuery
import buildstock_query.main as main
from buildstock_query.schema.query_params import AnyColType, AnyTableType


class BuildStockAggregate:

    def __init__(self, buildstock_query: 'main.BuildStockQuery') -> None:
        ...

    @typing.overload
    def aggregate_annual(self, *,
                         enduses: Sequence[str],
                         get_query_only: Literal[True],
                         group_by: Sequence[Union[AnyColType, tuple[str, str]]] = [],
                         sort: bool = False,
                         upgrade_id: Union[int, str] = '0',
                         join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
                         weights: Sequence[Union[str, tuple]] = [],
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = [],
                         get_quartiles: bool = False,
                         ) -> str:
        ...

    @typing.overload
    def aggregate_annual(self, *,
                         enduses: Sequence[str],
                         get_query_only: Literal[False] = False,
                         group_by: Sequence[Union[AnyColType, tuple[str, str]]] = [],
                         sort: bool = False,
                         upgrade_id: Union[int, str] = '0',
                         join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
                         weights: Sequence[Union[str, tuple]] = [],
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = [],
                         get_quartiles: bool = False,
                         ) -> pd.DataFrame:
        ...

    @typing.overload
    def aggregate_annual(self, *,
                         enduses: Sequence[str],
                         get_query_only: bool,
                         group_by: Sequence[Union[AnyColType, tuple[str, str]]] = [],
                         sort: bool = False,
                         upgrade_id: Union[int, str] = '0',
                         join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
                         weights: Sequence[Union[str, tuple]] = [],
                         restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = [],
                         get_quartiles: bool = False,
                         ) -> Union[pd.DataFrame, str]:
        """
        Aggregates the baseline annual result on select enduses.
        Check the argument description below to learn about additional features and options.
        Args:
            enduses: The list of enduses to aggregate. Defaults to all electricity enduses

            group_by: The list of columns to group the aggregation by.

            sort: Whether to sort the results by group_by colummns

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
            get_quartiles: If true, return the following quartiles in addition to the sum for each enduses:
                        [0, 0.02, .25, .5, .75, .98, 1]. The 0% quartile is the minimum and the 100% quartile
                        is the maximum.

            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging

        Returns:
                if get_query_only is True, returns the query_string, otherwise returns the dataframe
        """
        ...

    @typing.overload
    def aggregate_annual(self, *,
                         params: AnnualQuery
                         ) -> Union[str, pd.DataFrame]:
        ...

    @typing.overload
    def aggregate_timeseries(self, *,
                             enduses: Sequence[str],
                             get_query_only: Literal[True],
                             group_by: Sequence[Union[AnyColType, tuple[str, str]]] = [],
                             upgrade_id: Union[int, str] = '0',
                             sort: bool = False,
                             join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
                             weights: Sequence[Union[str, tuple]] = [],
                             restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = [],
                             split_enduses: bool = False,
                             collapse_ts: bool = False,
                             timestamp_grouping_func: Optional[str] = None,
                             limit: Optional[int] = None
                             ) -> str:
        ...

    @typing.overload
    def aggregate_timeseries(self, *,
                             enduses: Sequence[str],
                             group_by: Sequence[Union[AnyColType, tuple[str, str]]] = [],
                             upgrade_id: Union[int, str] = '0',
                             sort: bool = False,
                             join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
                             weights: Sequence[Union[str, tuple]] = [],
                             restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = [],
                             split_enduses: bool = False,
                             collapse_ts: bool = False,
                             timestamp_grouping_func: Optional[str] = None,
                             get_query_only: Literal[False] = False,
                             limit: Optional[int] = None
                             ) -> pd.DataFrame:
        ...

    @typing.overload
    def aggregate_timeseries(self, *,
                             enduses: Sequence[str],
                             get_query_only: bool,
                             group_by: Sequence[Union[AnyColType, tuple[str, str]]] = [],
                             upgrade_id: Union[int, str] = '0',
                             sort: bool = False,
                             join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = [],
                             weights: Sequence[Union[str, tuple]] = [],
                             restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = [],
                             split_enduses: bool = False,
                             collapse_ts: bool = False,
                             timestamp_grouping_func: Optional[str] = None,
                             limit: Optional[int] = None
                             ) -> Union[str, pd.DataFrame]:
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


        Returns:
                if get_query_only is True, returns the query_string, otherwise, returns the DataFrame

        """
        ...

    @typing.overload
    def aggregate_timeseries(self, *,
                             params: TSQuery,
                             ) -> Union[str, pd.DataFrame]:
        ...

    @typing.overload
    def get_building_average_kws_at(self,
                                    at_hour: Union[Sequence[float], float],
                                    at_days: Sequence[float],
                                    enduses: Sequence[str],
                                    get_query_only: Literal[False] = False) -> pd.DataFrame:
        ...

    @typing.overload
    def get_building_average_kws_at(self,
                                    at_hour: Union[Sequence[float], float],
                                    at_days: Sequence[float],
                                    enduses: Sequence[str],
                                    get_query_only: Literal[True]) -> str:
        ...
