import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
import datetime
import numpy as np
import logging
import buildstock_query.main as main
from typing import Optional
from buildstock_query.schema import AnnualQuery, TSQuery, accept_query
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FUELS = ['electricity', 'natural_gas', 'propane', 'fuel_oil', 'coal', 'wood_cord', 'wood_pellets']


class BuildStockAggregate:
    """A class to perform various aggregation queries for both timeseries and annual results.
    """

    def __init__(self, buildstock_query: 'main.BuildStockQuery') -> None:
        self._bsq = buildstock_query

    @accept_query(AnnualQuery)
    def aggregate_annual(self, *,
                         query_params: AnnualQuery):
        join_list = list(query_params.join_list) if query_params.join_list else []
        weights = list(query_params.weights) if query_params.weights else []
        restrict = list(query_params.restrict) if query_params.restrict else []

        [self._bsq.get_table(jl[0]) for jl in join_list]  # ingress all tables in join list
        if query_params.upgrade_id in {None, 0, '0'}:
            enduse_cols = self._bsq._get_enduse_cols(query_params.enduses, table='baseline')
            upgrade_id = None
        else:
            upgrade_id = self._bsq._validate_upgrade(query_params.upgrade_id)
            enduse_cols = self._bsq._get_enduse_cols(query_params.enduses, table='upgrade')

        total_weight = self._bsq._get_weight(weights)
        enduse_selection = [safunc.sum(enduse * total_weight).label(self._bsq._simple_label(enduse.name))
                            for enduse in enduse_cols]
        if query_params.get_quartiles:
            enduse_selection += [sa.func.approx_percentile(enduse, [0, 0.02, 0.25, 0.5, 0.75, 0.98, 1]).label(
                f"{self._bsq._simple_label(enduse.name)}__quartiles") for enduse in enduse_cols]

        grouping_metrics_selction = [safunc.sum(1).label("sample_count"),
                                     safunc.sum(total_weight).label("units_count")]

        if not query_params.group_by:
            query = sa.select(grouping_metrics_selction + enduse_selection)
            group_by_selection = []
        else:
            group_by_selection = self._bsq._process_groupby_cols(query_params.group_by, annual_only=True)
            query = sa.select(group_by_selection + grouping_metrics_selction + enduse_selection)
        # jj = self.bs_table.join(self.ts_table, self.ts_table.c['building_id']==self.bs_table.c['building_id'])
        # self._compile(query.select_from(jj))
        if upgrade_id not in [None, 0, '0']:
            tbljoin = self._bsq.bs_table.join(
                self._bsq.up_table, sa.and_(self._bsq.bs_table.c[self._bsq.building_id_column_name] ==
                                            self._bsq.up_table.c[self._bsq.building_id_column_name],
                                            self._bsq.up_table.c["upgrade"] == str(upgrade_id),
                                            self._bsq.up_table.c["completed_status"] == "Success"))
            query = query.select_from(tbljoin)

        restrict = [(self._bsq.bs_table.c['completed_status'], ['Success'])] + restrict
        query = self._bsq._add_join(query, join_list)
        query = self._bsq._add_restrict(query, restrict)
        query = self._bsq._add_group_by(query, group_by_selection)
        query = self._bsq._add_order_by(query, group_by_selection if query_params.sort else [])

        if query_params.get_query_only:
            return self._bsq._compile(query)

        return self._bsq.execute(query, run_async=query_params.run_async)

    def _aggregate_timeseries_light(self,
                                    query_params: TSQuery
                                    ):
        """
        Lighter version of aggregate_timeseries where each enduse is submitted as a separate query to be light on
        Athena. For information on the input parameters, check the documentation on aggregate_timeseries.
        """
        qp = query_params
        if qp.run_async:
            raise ValueError("Async run is not available for aggregate_timeseries_light since it needs to combine"
                             "the result after the query finishes.")

        enduse_cols = self._bsq._get_enduse_cols(qp.enduses, table='timeseries')
        batch_queries_to_submit = []
        for enduse in enduse_cols:
            new_query = qp.copy()
            new_query.enduses = [enduse.name]
            query = self.aggregate_timeseries(new_query)
            batch_queries_to_submit.append(query)

        if qp.get_query_only:
            logger.warning("Not recommended to use get_query_only and split_enduses used together."
                           " The results from the queries cannot be directly combined to get the desired result."
                           " There are further processing done in the function. The queries should be used for"
                           " information or debugging purpose only. Use get_query_only=False to get proper result.")
            return batch_queries_to_submit

        batch_query_id = self._bsq.submit_batch_query(batch_queries_to_submit)

        result_dfs = self._bsq.get_batch_query_result(batch_id=batch_query_id, combine=False)
        logger.info("Joining the individual enduses result into a single DataFrame")
        group_by = self._bsq._clean_group_by(qp.group_by)
        for res in result_dfs:
            res.set_index(group_by, inplace=True)
        self.result_dfs = result_dfs
        joined_enduses_df = result_dfs[0].drop(columns=['query_id'])
        for enduse, res in list(zip(qp.enduses, result_dfs))[1:]:
            joined_enduses_df = joined_enduses_df.join(res[[enduse.name]])

        logger.info("Joining Completed.")
        return joined_enduses_df.reset_index()

    @accept_query(TSQuery)
    def aggregate_timeseries(self, query_params: TSQuery):
        qp = query_params
        upgrade_id = self._bsq._validate_upgrade(qp.upgrade_id)
        if qp.timestamp_grouping_func and \
                qp.timestamp_grouping_func not in ['hour', 'day', 'month']:
            raise ValueError("timestamp_grouping_func must be one of ['hour', 'day', 'month']")

        if qp.split_enduses:
            return self._aggregate_timeseries_light(qp)
        [self._bsq.get_table(jl[0]) for jl in qp.join_list]  # ingress all tables in join list
        enduses_cols = self._bsq._get_enduse_cols(qp.enduses, table='timeseries')
        total_weight = self._bsq._get_weight(qp.weights)

        enduse_selection = [safunc.sum(enduse * total_weight).label(self._bsq._simple_label(enduse.name))
                            for enduse in enduses_cols]

        if self._bsq.timestamp_column_name not in qp.group_by and qp.collapse_ts:
            logger.info("Aggregation done accross timestamps. Result no longer a timeseries.")
            # The aggregation is done across time so we should correct sample_count and units_count
            rows_per_building = self._bsq._get_rows_per_building()
            grouping_metrics_selection = [(safunc.sum(1) / rows_per_building).label(
                "sample_count"), safunc.sum(total_weight / rows_per_building).label("units_count")]
        elif self._bsq.timestamp_column_name not in qp.group_by:
            qp.group_by.append(self._bsq.timestamp_column_name)
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(total_weight).label("units_count")]
        elif qp.collapse_ts:
            raise ValueError("collapse_ts is true, but there is timestamp column in group_by.")
        else:
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(total_weight).label("units_count")]

        if (colname := self._bsq.timestamp_column_name) in qp.group_by and \
                qp.timestamp_grouping_func:
            # sample_count = count(distinct(building_id))
            # units_count = count(distinct(buuilding_id)) * sum(total_weight) / sum(1)
            grouping_metrics_selection = [safunc.count(safunc.distinct(self._bsq.ts_bldgid_column)).
                                          label("sample_count"),
                                          (safunc.count(safunc.distinct(self._bsq.ts_bldgid_column)) *
                                           safunc.sum(total_weight) / safunc.sum(1)).label("units_count"),
                                          (safunc.sum(1) / safunc.count(safunc.distinct(self._bsq.ts_bldgid_column))).
                                          label("rows_per_sample"), ]
            indx = qp.group_by.index(colname)
            _, _, start_offset = self._bsq._get_simulation_info()
            if start_offset > 0:
                # If timestamps are not period begining we should make them so for timestamp_grouping_func aggregation.
                new_col = sa.func.date_trunc(qp.timestamp_grouping_func,
                                             sa.func.date_add('second',
                                                              -start_offset, self._bsq.timestamp_column)).label(colname)
            else:
                new_col = sa.func.date_trunc(qp.timestamp_grouping_func, self._bsq.timestamp_column).label(colname)
            qp.group_by[indx] = new_col

        group_by_selection = self._bsq._process_groupby_cols(qp.group_by, annual_only=False)

        query = sa.select(group_by_selection + grouping_metrics_selection + enduse_selection)
        query = query.join(self._bsq.bs_table, self._bsq.bs_bldgid_column == self._bsq.ts_bldgid_column)
        if qp.join_list:
            query = self._bsq._add_join(query, qp.join_list)

        group_by_names = [g.name for g in group_by_selection]
        upgrade_in_restrict = any(entry[0] == 'upgrade' for entry in qp.restrict)
        if self._bsq.up_table is not None and not upgrade_in_restrict and 'upgrade' not in group_by_names:
            logger.info(f"Restricting query to Upgrade {upgrade_id}.")
            qp.restrict.append((self._bsq.ts_table.c['upgrade'], [upgrade_id]))

        query = self._bsq._add_restrict(query, qp.restrict)
        query = self._bsq._add_group_by(query, group_by_selection)
        query = self._bsq._add_order_by(query, group_by_selection if qp.sort else [])
        query = query.limit(qp.limit) if qp.limit else query

        if qp.get_query_only:
            return self._bsq._compile(query)

        return self._bsq.execute(query, run_async=qp.run_async)

    def get_building_average_kws_at(self,
                                    at_hour: list[int],
                                    at_days: list[int],
                                    enduses: Optional[list[str]] = None,
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

        enduse_cols = self._bsq._get_enduse_cols(enduses, table='timeseries')
        total_weight = self._bsq._get_weight([])

        sim_year, sim_interval_seconds, _ = self._bsq._get_simulation_info()
        kw_factor = 3600.0 / sim_interval_seconds

        enduse_selection = [safunc.avg(enduse * total_weight * kw_factor).label(self._bsq._simple_label(enduse.name))
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

        query = sa.select([self._bsq.ts_bldgid_column] + grouping_metrics_selection + enduse_selection)
        query = query.join(self._bsq.bs_table, self._bsq.bs_bldgid_column == self._bsq.ts_bldgid_column)
        query = self._bsq._add_group_by(query, [self._bsq.ts_bldgid_column])
        query = self._bsq._add_order_by(query, [self._bsq.ts_bldgid_column])

        lower_val_query = self._bsq._add_restrict(query, [(self._bsq.timestamp_column_name, lower_timestamps)])
        upper_val_query = self._bsq._add_restrict(query, [(self._bsq.timestamp_column_name, upper_timestamps)])

        if exact_times:
            # only one query is sufficient if the hours fall in exact timestamps
            queries = [lower_val_query]
        else:
            queries = [lower_val_query, upper_val_query]

        if get_query_only:
            return [self._bsq._compile(q) for q in queries]

        batch_id = self._bsq.submit_batch_query(queries)
        if exact_times:
            vals, = self._bsq.get_batch_query_result(batch_id, combine=False)
            return vals
        else:
            lower_vals, upper_vals = self._bsq.get_batch_query_result(batch_id, combine=False)
            avg_upper_weight = np.mean([min_of_hour / sim_interval_seconds for hour in at_hour if
                                        (min_of_hour := hour * 3600 % sim_interval_seconds)])
            avg_lower_weight = 1 - avg_upper_weight
            # modify the lower vals to make it weighted average of upper and lower vals
            lower_vals[enduses] = lower_vals[enduses] * avg_lower_weight + upper_vals[enduses] * avg_upper_weight
            return lower_vals
