from typing import Optional, Union, Literal, LiteralString
import pandas as pd
import typing
from buildstock_query.helpers import AthenaFutureDf, CachedFutureDf
import sqlalchemy as sa

class BuildStockAggregate:

    @typing.overload
    def aggregate_annual(self,
                         enduses: Optional[list[str]] = None,
                         group_by: Optional[list[Union[sa.sql.elements.Label, sa.Column, str, tuple[str, str]]]] = None,
                         sort: bool = False,
                         upgrade_id: Optional[Union[str, int]] = None,
                         join_list: Optional[list[tuple[str, str, str]]] = None,
                         weights: Optional[list[Union[str, tuple]]] = None,
                         restrict: Optional[list[tuple[str, Union[str, int, list]]]] = None,
                         get_quartiles: bool = False,
                         run_async: Literal[False] = False,
                         get_query_only: Optional[Literal[False]]= False) -> pd.DataFrame:
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
            run_async: Whether to run the query in the background. Returns immediately if running in background,
                    blocks otherwise.
            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging

        Returns:
                if get_query_only is True, returns the query_string, otherwise,
                    if run_async is True, it returns a query_execution_id.
                    if run_async is False, it returns the result_dataframe

        """
        ...

    @typing.overload
    def aggregate_annual(self, get_query_only: Literal[True],
                         enduses: Optional[list[str]] = None,
                         group_by: Optional[list[Union[sa.sql.elements.Label, sa.Column, str, tuple[str, str]]]] = None,
                         sort: bool = False,
                         upgrade_id: Optional[Union[str, int]] = None,
                         join_list: Optional[list[tuple[str, str, str]]] = None,
                         weights: Optional[list[Union[str, tuple]]] = None,
                         restrict: Optional[list[tuple[str, Union[str, int, list]]]] = None,
                         get_quartiles: bool = False,
                         run_async: bool = False,
                         ) -> str:
        ...

    @typing.overload
    def aggregate_annual(self,
                         run_async: Literal[True],
                         get_query_only: Optional[Literal[False]]= False,
                         enduses: Optional[list[str]] = None,
                         group_by: Optional[list[Union[sa.sql.elements.Label, sa.Column, str, tuple[str, str]]]] = None,
                         sort: bool = False,
                         upgrade_id: Optional[Union[str, int]] = None,
                         join_list: Optional[list[tuple[str, str, str]]] = None,
                         weights: Optional[list[Union[str, tuple]]] = None,
                         restrict: Optional[list[tuple[str, Union[str, int, list]]]] = None,
                         get_quartiles: bool = False,
                         ) -> Union[tuple[Literal["CACHED"], CachedFutureDf], tuple[str, AthenaFutureDf]]:
        ...

    @typing.overload
    def aggregate_timeseries(self,
                             enduses: Optional[list[str]],
                             group_by: Optional[list[Union[sa.sql.elements.Label,
                                                           sa.Column, str, tuple[str, str]]]] = None,
                             upgrade_id: Optional[int] = None,
                             sort: bool = False,
                             join_list: Optional[list[tuple[str, str, str]]] = None,
                             weights: Optional[list[str]] = None,
                             restrict: Optional[list[tuple[str, Union[str, int, list]]]] = None,
                             split_enduses: Optional[bool] = False,
                             collapse_ts: Optional[bool] = False,
                             timestamp_grouping_func: Optional[str] = None,
                             run_async: Literal[False] = False,
                             get_query_only: Optional[Literal[False]]= False,
                             limit: Optional[int] = None
                             ) -> pd.DataFrame:
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

        run_async: Whether to run the query in the background. Returns immediately if running in background,
                blocks otherwise.
        split_enduses: Whether to query for each enduses in a separate query to reduce Athena load for query. Useful
                        when Athena runs into "Query exhausted resources ..." errors.
        timestamp_grouping_func: One of 'hour', 'day' or 'month' or None. If provided, perform timeseries
                                aggregation of specified granularity.
        get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                        submitting multiple queries or debugging


        Returns:
                if get_query_only is True, returns the query_string, otherwise,
                if run_async is True, it returns a query_execution_id and futuredf.
                if run_async is False, it returns the result_dataframe

        """
        ...

    @typing.overload
    def aggregate_timeseries(self,
                             enduses: Optional[list[str]],
                             get_query_only: Literal[True],
                             group_by: Optional[list[Union[sa.sql.elements.Label,
                                                           sa.Column, str, tuple[str, str]]]] = None,
                             upgrade_id: Optional[int] = None,
                             sort: bool = False,
                             join_list: Optional[list[tuple[str, str, str]]] = None,
                             weights: Optional[list[str]] = None,
                             restrict: Optional[list[tuple[str, Union[str, int, list]]]] = None,
                             split_enduses: Optional[bool] = False,
                             collapse_ts: Optional[bool] = False,
                             timestamp_grouping_func: Optional[str] = None,
                             run_async: Literal[False] = False,
                             limit: Optional[int] = None
                             ) -> str:
        ...

    @typing.overload
    def aggregate_timeseries(self,
                             run_async: Literal[True],
                             enduses: Optional[list[str]],
                             group_by: Optional[list[Union[sa.sql.elements.Label,
                                                           sa.Column, str, tuple[str, str]]]] = None,
                             upgrade_id: Optional[int] = None,
                             sort: bool = False,
                             join_list: Optional[list[tuple[str, str, str]]] = None,
                             weights: Optional[list[str]] = None,
                             restrict: Optional[list[tuple[str, Union[str, int, list]]]] = None,
                             split_enduses: Optional[bool] = False,
                             collapse_ts: Optional[bool] = False,
                             timestamp_grouping_func: Optional[str] = None,
                             get_query_only: Optional[Literal[False]]= False,
                             limit: Optional[int] = None
                             ) -> Union[tuple[Literal["CACHED"], CachedFutureDf], tuple[str, AthenaFutureDf]]:
        ...
