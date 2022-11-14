"""
# ResStockAthena
- - - - - - - - -
A class to run AWS Athena queries to get various data from a ResStock run. All queries and aggregation that can be
common accross different ResStock projects should be implemented in this class. For queries that are project specific, a
new class can be created by inheriting ResStockAthena and adding in the project specific logic and queries.

:author: Rajendra.Adhikari@nrel.gov
"""


import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
import datetime
import numpy as np
import logging
import buildstock_query.main as main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FUELS = ['electricity', 'natural_gas', 'propane', 'fuel_oil', 'coal', 'wood_cord', 'wood_pellets']


class BuildStockAggregate:
    def __init__(self, buildstock_query: 'main.BuildStockQuery') -> None:
        self.bsq = buildstock_query

    def aggregate_annual(self,
                         enduses: list[str] | None = None,
                         group_by: list[str] | None = None,
                         sort: bool = False,
                         upgrade_id: str | int | None = None,
                         join_list: list[tuple[str, str, str]] | None = None,
                         weights: list[str | tuple] | None = None,
                         restrict: list[tuple[str, list]] | None = None,
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
        join_list = list(join_list) if join_list else []
        weights = list(weights) if weights else []
        restrict = list(restrict) if restrict else []

        [self.bsq.get_table(jl[0]) for jl in join_list]  # ingress all tables in join list
        if upgrade_id in {None, 0, '0'}:
            enduse_cols = self.bsq._get_enduse_cols(enduses, table='baseline')
        else:
            upgrade_id = self.bsq.validate_upgrade(upgrade_id)
            enduse_cols = self.bsq._get_enduse_cols(enduses, table='upgrade')

        total_weight = self.bsq._get_weight(weights)
        enduse_selection = [safunc.sum(enduse * total_weight).label(self.bsq._simple_label(enduse.name))
                            for enduse in enduse_cols]
        if get_quartiles:
            enduse_selection += [sa.func.approx_percentile(enduse, [0, 0.02, 0.25, 0.5, 0.75, 0.98, 1]).label(
                f"{self.bsq._simple_label(enduse.name)}__quartiles") for enduse in enduse_cols]

        grouping_metrics_selction = [safunc.sum(1).label("sample_count"),
                                     safunc.sum(total_weight).label("units_count")]

        if not group_by:
            query = sa.select(grouping_metrics_selction + enduse_selection)
            group_by_selection = []
        else:
            group_by_selection = self.bsq._process_groupby_cols(group_by, annual_only=True)
            query = sa.select(group_by_selection + grouping_metrics_selction + enduse_selection)
        # jj = self.bs_table.join(self.ts_table, self.ts_table.c['building_id']==self.bs_table.c['building_id'])
        # self._compile(query.select_from(jj))
        if upgrade_id not in [None, 0, '0']:
            tbljoin = self.bsq.bs_table.join(
                self.bsq.up_table, sa.and_(self.bsq.bs_table.c[self.bsq.building_id_column_name] ==
                                           self.bsq.up_table.c[self.bsq.building_id_column_name],
                                           self.bsq.up_table.c["upgrade"] == str(upgrade_id),
                                           self.bsq.up_table.c["completed_status"] == "Success"))
            query = query.select_from(tbljoin)

        restrict = [(self.bsq.bs_table.c['completed_status'], ['Success'])] + restrict
        query = self.bsq._add_join(query, join_list)
        query = self.bsq._add_restrict(query, restrict)
        query = self.bsq._add_group_by(query, group_by_selection)
        query = self.bsq._add_order_by(query, group_by_selection if sort else [])

        if get_query_only:
            return self.bsq._compile(query)

        return self.bsq.execute(query, run_async=run_async)

    def _aggregate_timeseries_light(self,
                                    enduses: list[str] | None = None,
                                    group_by: list[str] | None = None,
                                    sort: bool = False,
                                    join_list: list[tuple[str, str, str]] | None = None,
                                    weights: list[str] | None = None,
                                    restrict: list[tuple[str, list]] | None = None,
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

        enduse_cols = self.bsq._get_enduse_cols(enduses, table='timeseries')
        print(enduses)
        batch_queries_to_submit = []
        for enduse in enduse_cols:
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

        batch_query_id = self.bsq.submit_batch_query(batch_queries_to_submit)

        result_dfs = self.bsq.get_batch_query_result(batch_id=batch_query_id, combine=False)
        logger.info("Joining the individual enduses result into a single DataFrame")
        group_by = self.bsq._clean_group_by(group_by)
        for res in result_dfs:
            res.set_index(group_by, inplace=True)
        self.result_dfs = result_dfs
        joined_enduses_df = result_dfs[0].drop(columns=['query_id'])
        for enduse, res in list(zip(enduses, result_dfs))[1:]:
            joined_enduses_df = joined_enduses_df.join(res[[enduse.name]])

        logger.info("Joining Completed.")
        return joined_enduses_df.reset_index()

    def aggregate_timeseries(self,
                             enduses: list[str] | None = None,
                             group_by: list[str] | None = None,
                             upgrade_id: int | None = None,
                             sort: bool = False,
                             join_list: list[tuple[str, str, str]] | None = None,
                             weights: list[str] | None = None,
                             restrict: list[tuple[str, list]] | None = None,
                             run_async: bool = False,
                             split_enduses: bool = False,
                             collapse_ts: bool = False,
                             get_query_only: bool = False,
                             limit: int | None = None
                             ):
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
        upgrade_id = self.bsq.validate_upgrade(upgrade_id)

        if split_enduses:
            return self._aggregate_timeseries_light(enduses=enduses, group_by=group_by, sort=sort,
                                                   join_list=join_list, weights=weights, restrict=restrict,
                                                   run_async=run_async, get_query_only=get_query_only,
                                                   limit=limit)
        [self.bsq.get_table(jl[0]) for jl in join_list]  # ingress all tables in join list
        enduses_cols = self.bsq._get_enduse_cols(enduses, table='timeseries')
        total_weight = self.bsq._get_weight(weights)

        enduse_selection = [safunc.sum(enduse * total_weight).label(self.bsq._simple_label(enduse.name))
                            for enduse in enduses_cols]

        if self.bsq.timestamp_column_name not in group_by and collapse_ts:
            logger.info("Aggregation done accross timestamps. Result no longer a timeseries.")
            # The aggregation is done across time so we should correct sample_count and units_count
            rows_per_building = self.bsq._get_rows_per_building()
            grouping_metrics_selection = [(safunc.sum(1) / rows_per_building).label(
                "sample_count"), safunc.sum(total_weight / rows_per_building).label("units_count")]
        elif self.bsq.timestamp_column_name not in group_by:
            group_by.append(self.bsq.timestamp_column_name)
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(total_weight).label("units_count")]
        elif collapse_ts:
            raise ValueError("collapse_ts is true, but there is timestamp column in group_by.")
        else:
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(total_weight).label("units_count")]
        group_by_selection = self.bsq._process_groupby_cols(group_by, annual_only=False)

        query = sa.select(group_by_selection + grouping_metrics_selection + enduse_selection)
        query = query.join(self.bsq.bs_table, self.bsq.bs_bldgid_column == self.bsq.ts_bldgid_column)
        if join_list:
            query = self.bsq._add_join(query, join_list)

        group_by_names = [g.name for g in group_by_selection]
        upgrade_in_restrict = all(entry[0] != 'ugrade' for entry in restrict)
        if self.bsq.up_table is not None and not upgrade_in_restrict and 'upgrade' not in group_by_names:
            logger.info(f"Restricting query to Upgrade {upgrade_id}.")
            restrict.append((self.bsq.ts_table.c['upgrade'], [upgrade_id]))

        query = self.bsq._add_restrict(query, restrict)
        query = self.bsq._add_group_by(query, group_by_selection)
        query = self.bsq._add_order_by(query, group_by_selection if sort else [])
        query = query.limit(limit) if limit else query

        if get_query_only:
            return self.bsq._compile(query)

        return self.bsq.execute(query, run_async=run_async)

    def get_building_average_kws_at(self,
                                    at_hour: list[int],
                                    at_days: list[int],
                                    enduses: list[str] | None = None,
                                    get_query_only: bool = False):
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
            if len(at_hour) != len(at_days) or not at_hour:
                raise ValueError("The length of at_hour list should be the same as length of at_days list and"
                                 " not be empty")
        elif isinstance(at_hour, (float, int)):
            at_hour = [at_hour] * len(at_days)
        else:
            raise ValueError("At hour should be a list or a number")

        enduse_cols = self.bsq._get_enduse_cols(enduses, table='timeseries')
        total_weight = self.bsq._get_weight([])

        sim_year, sim_interval_seconds = self.bsq._get_simulation_info()
        kw_factor = 3600.0 / sim_interval_seconds

        enduse_selection = [safunc.avg(enduse * total_weight * kw_factor).label(self.bsq._simple_label(enduse.name))
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

        query = sa.select([self.bsq.ts_bldgid_column] + grouping_metrics_selection + enduse_selection)
        query = query.join(self.bsq.bs_table, self.bsq.bs_bldgid_column == self.bsq.ts_bldgid_column)
        query = self.bsq._add_group_by(query, [self.bsq.ts_bldgid_column])
        query = self.bsq._add_order_by(query, [self.bsq.ts_bldgid_column])

        lower_val_query = self.bsq._add_restrict(query, [(self.bsq.timestamp_column_name, lower_timestamps)])
        upper_val_query = self.bsq._add_restrict(query, [(self.bsq.timestamp_column_name, upper_timestamps)])

        if exact_times:
            # only one query is sufficient if the hours fall in exact timestamps
            queries = [lower_val_query]
        else:
            queries = [lower_val_query, upper_val_query]

        if get_query_only:
            return [self.bsq._compile(q) for q in queries]

        batch_id = self.bsq.submit_batch_query(queries)
        if exact_times:
            vals, = self.bsq.get_batch_query_result(batch_id, combine=False)
            return vals
        else:
            lower_vals, upper_vals = self.bsq.get_batch_query_result(batch_id, combine=False)
            avg_upper_weight = np.mean([min_of_hour / sim_interval_seconds for hour in at_hour if
                                        (min_of_hour := hour * 3600 % sim_interval_seconds)])
            avg_lower_weight = 1 - avg_upper_weight
            # modify the lower vals to make it weighted average of upper and lower vals
            lower_vals[enduses] = lower_vals[enduses] * avg_lower_weight + upper_vals[enduses] * avg_upper_weight
            return lower_vals
