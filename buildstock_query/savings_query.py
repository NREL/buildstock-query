import pandas as pd
import sqlalchemy as sa
from typing import List, Tuple
from sqlalchemy.sql import func as safunc
import buildstock_query.main as main
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuildStockSavings:
    """Class for doing savings query (both timeseries and annual).
    """

    def __init__(self, buildstock_query: 'main.BuildStockQuery') -> None:
        self._bsq = buildstock_query

    def _validate_partition_by(self, partition_by: list[str]):
        if not partition_by:
            return []
        [self._bsq._get_gcol(col) for col in partition_by]  # making sure all entries are valid
        return partition_by

    def __get_timeseries_bs_up_table(self, enduses: list[str], upgrade_id: int, applied_only: bool,
                                     restrict: Optional[list] = None, ts_group_by: Optional[list[str]] = None):
        restrict = list(restrict) if restrict else []
        ts_group_by = list(ts_group_by) if ts_group_by else []
        ts = self._bsq.ts_table
        base = self._bsq.bs_table
        sa_ts_cols = [ts.c[self._bsq.building_id_column_name], ts.c[self._bsq.timestamp_column_name]] + ts_group_by
        sa_ts_cols.extend(enduses)
        ucol = ts.c["upgrade"]
        ts_b = self._bsq._add_restrict(sa.select(sa_ts_cols), [[ucol, ("0")]] + restrict).alias("ts_b")
        ts_u = self._bsq._add_restrict(sa.select(sa_ts_cols), [[ucol, (str(upgrade_id))]] + restrict).alias("ts_u")

        if applied_only:
            tbljoin = (
                ts_b.join(
                   ts_u, sa.and_(ts_b.c[self._bsq.building_id_column_name] == ts_u.c[self._bsq.building_id_column_name],
                                 ts_b.c[self._bsq.timestamp_column_name] == ts_u.c[self._bsq.timestamp_column_name])
                ).join(base, ts_b.c[self._bsq.building_id_column_name] == base.c[self._bsq.building_id_column_name])
            )
        else:
            tbljoin = (
                ts_b.outerjoin(
                   ts_u, sa.and_(ts_b.c[self._bsq.building_id_column_name] == ts_u.c[self._bsq.building_id_column_name],
                                 ts_b.c[self._bsq.timestamp_column_name] == ts_u.c[self._bsq.timestamp_column_name])
                ).join(base, ts_b.c[self._bsq.building_id_column_name] == base.c[self._bsq.building_id_column_name])
            )
        return ts_b, ts_u, tbljoin

    def __get_annual_bs_up_table(self, upgrade_id: int, applied_only: bool):
        if applied_only:
            tbljoin = (
                self._bsq.bs_table.join(
                    self._bsq.up_table, sa.and_(self._bsq.bs_table.c[self._bsq.building_id_column_name] ==
                                                self._bsq.up_table.c[self._bsq.building_id_column_name],
                                                self._bsq.up_table.c["upgrade"] == str(upgrade_id),
                                                self._bsq.up_table.c["completed_status"] == "Success"))
            )
        else:
            tbljoin = (
                self._bsq.bs_table.outerjoin(
                    self._bsq.up_table, sa.and_(self._bsq.bs_table.c[self._bsq.building_id_column_name] ==
                                                self._bsq.up_table.c[self._bsq.building_id_column_name],
                                                self._bsq.up_table.c["upgrade"] == str(upgrade_id),
                                                self._bsq.up_table.c["completed_status"] == "Success")))

        return self._bsq.bs_table, self._bsq.up_table, tbljoin

    def savings_shape(
        self,
        upgrade_id: int,
        enduses: Optional[List[str]] = None,
        group_by: Optional[List[str]] = None,
        annual_only: bool = True,
        sort: bool = True,
        join_list: Optional[List[Tuple[str, str, str]]] = None,
        weights: Optional[List[Tuple]] = None,
        restrict: Optional[List[Tuple[str, List]]] = None,
        run_async: bool = False,
        applied_only: bool = False,
        get_quartiles: bool = False,
        timestamp_grouping_func: Optional[str] = None,
        get_query_only: bool = False,
        unload_to: str = '',
        partition_by: Optional[List[str]] = None,
        collapse_ts: bool = False,
    ) -> pd.DataFrame:
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

            run_async: Whether to run the query in the background. Returns immediately if running in background,
                       blocks otherwise.
            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging
            get_quartiles: If true, return the following quartiles in addition to the sum for each enduses:
                           [0, 0.02, .25, .5, .75, .98, 1]. The 0% quartile is the minimum and the 100% quartile
                           is the maximum.
            unload_to: Writes the ouput of the query to this location in s3. Consider using run_async = True with this
                       to unload multiple queries simulataneuosly
            partition_by: List of columns to partition when writing to s3. To be used with unload_to.
            collapse_ts: Only used when annual_only=False. When collapse_ts=True, the timeseries values are summed into
                         a single annual value. Useful for quality checking and comparing with annual values.
         Returns:
                if get_query_only is True, returns the query_string, otherwise,
                    if run_async is True, it returns a (query_execution_id, future_object).
                    if run_async is False, it returns the result_dataframe
        """
        enduses = list(enduses) if enduses else []
        group_by = list(group_by) if group_by else []
        join_list = list(join_list) if join_list else []
        weights = list(weights) if weights else []
        restrict = list(restrict) if restrict else []
        partition_by = list(partition_by) if partition_by else []

        [self._bsq.get_table(jl[0]) for jl in join_list]  # ingress all tables in join list

        upgrade_id = self._bsq._validate_upgrade(upgrade_id)
        enduse_cols = self._bsq._get_enduse_cols(enduses, table="baseline" if annual_only else "timeseries")
        partition_by = self._validate_partition_by(partition_by)
        total_weight = self._bsq._get_weight(weights)
        group_by_selection = self._bsq._process_groupby_cols(group_by, annual_only=annual_only)

        if annual_only:
            ts_b, ts_u, tbljoin = self.__get_annual_bs_up_table(upgrade_id, applied_only)
        else:
            restrict, ts_restrict = self._bsq._split_restrict(restrict)
            bs_group_by, ts_group_by = self._bsq._split_group_by(group_by_selection)
            ts_b, ts_u, tbljoin = self.__get_timeseries_bs_up_table(enduse_cols, upgrade_id, applied_only, ts_restrict,
                                                                    ts_group_by)
            ts_group_by = [ts_b.c[c.name] for c in ts_group_by]  # Refer to the columns using ts_b table
            group_by_selection = bs_group_by + ts_group_by
        query_cols = []
        for col in enduse_cols:
            if annual_only:
                savings_col = (safunc.coalesce(ts_b.c[col.name], 0) -
                               safunc.coalesce(sa.case((ts_u.c['completed_status'] == 'Success', ts_u.c[col.name]),
                                               else_=ts_b.c[col.name]), 0)
                               )
            else:
                savings_col = (safunc.coalesce(ts_b.c[col.name], 0) -
                               safunc.coalesce(sa.case((ts_u.c['building_id'] == None, ts_b.c[col.name]),  # noqa E711
                                               else_=ts_u.c[col.name]), 0)
                               )
            query_cols.extend(
                [
                    sa.func.sum(ts_b.c[col.name] *
                                total_weight).label(f"{self._bsq._simple_label(col.name)}__baseline"),
                    sa.func.sum(savings_col * total_weight).label(f"{self._bsq._simple_label(col.name)}__savings"),
                ]
            )
            if get_quartiles:
                query_cols.extend(
                    [sa.func.approx_percentile(savings_col, [0, 0.02, 0.25, 0.5, 0.75, 0.98, 1]).
                     label(f"{self._bsq._simple_label(col.name)}__savings__quartiles")
                     ]
                )

        query_cols.extend(group_by_selection)
        if annual_only:  # Use annual tables
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(1 * total_weight).label("units_count")]
            query_cols = grouping_metrics_selection + query_cols
        elif collapse_ts:  # Use timeseries tables but collapse timeseries
            rows_per_building = self._bsq._get_rows_per_building()
            grouping_metrics_selection = [(safunc.sum(1) / rows_per_building).label(
                "sample_count"), safunc.sum(total_weight / rows_per_building).label("units_count")]
            query_cols = grouping_metrics_selection + query_cols
        else:  # Use timeseries table and return timeseries results
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(1 * total_weight).label("units_count")]
            query_cols = grouping_metrics_selection + query_cols
            #time_col = ts_b.c[self._bsq.timestamp_column_name].label(self._bsq.timestamp_column_name)
            #query_cols.insert(0, time_col)
            #group_by_selection.append(time_col)

        if timestamp_grouping_func and timestamp_grouping_func not in ['hour', 'day', 'month']:
            raise ValueError("timestamp_grouping_func must be one of ['hour', 'day', 'month']")
        
        enduse_selection = [safunc.sum(enduse * total_weight).label(self._bsq._simple_label(enduse.name))
                            for enduse in enduse_cols]

        if self._bsq.timestamp_column_name not in group_by and collapse_ts:
            logger.info("Aggregation done accross timestamps. Result no longer a timeseries.")
            # The aggregation is done across time so we should correct sample_count and units_count
            rows_per_building = self._bsq._get_rows_per_building()
            grouping_metrics_selection = [(safunc.sum(1) / rows_per_building).label(
                "sample_count"), safunc.sum(total_weight / rows_per_building).label("units_count")]
        elif self._bsq.timestamp_column_name not in group_by:
            group_by.append(self._bsq.timestamp_column_name)
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(total_weight).label("units_count")]
        elif collapse_ts:
            raise ValueError("collapse_ts is true, but there is timestamp column in group_by.")
        else:
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(total_weight).label("units_count")]

        if (colname := self._bsq.timestamp_column_name) in group_by and timestamp_grouping_func:
            # sample_count = count(distinct(building_id))
            # units_count = count(distinct(buuilding_id)) * sum(total_weight) / sum(1)
            grouping_metrics_selection = [safunc.count(safunc.distinct(self._bsq.ts_bldgid_column)).
                                          label("sample_count"),
                                          (safunc.count(safunc.distinct(self._bsq.ts_bldgid_column)) *
                                           safunc.sum(total_weight) / safunc.sum(1)).label("units_count"),
                                          (safunc.sum(1) / safunc.count(safunc.distinct(self._bsq.ts_bldgid_column))).
                                          label("rows_per_sample"), ]
            indx = group_by.index(colname)
            _, _, start_offset = self._bsq._get_simulation_info()
            if start_offset > 0:
                # If timestamps are not period begining we should make them so for timestamp_grouping_func aggregation.
                new_col = sa.func.date_trunc(timestamp_grouping_func,
                                             sa.func.date_add('second',
                                                              -start_offset, self._bsq.timestamp_column)).label(colname)
            else:
                new_col = sa.func.date_trunc(timestamp_grouping_func, self._bsq.timestamp_column).label(colname)
            group_by[indx] = new_col

        group_by_selection = self._bsq._process_groupby_cols(group_by, annual_only=False)

        #query = sa.select(group_by_selection + grouping_metrics_selection + enduse_selection)
        #query = query.join(self._bsq.bs_table, self._bsq.bs_bldgid_column == self._bsq.ts_bldgid_column)
        #if join_list:
            #query = self._bsq._add_join(query, join_list)

        #group_by_names = [g.name for g in group_by_selection]
        #upgrade_in_restrict = any(entry[0] == 'upgrade' for entry in restrict)
        #if self._bsq.up_table is not None and not upgrade_in_restrict and 'upgrade' not in group_by_names:
            #logger.info(f"Restricting query to Upgrade {upgrade_id}.")
            #restrict.append((self._bsq.ts_table.c['upgrade'], [upgrade_id]))

        #query = self._bsq._add_restrict(query, restrict)
        #query = self._bsq._add_group_by(query, group_by_selection)
        #query = self._bsq._add_order_by(query, group_by_selection if sort else [])

        query_cols = group_by_selection + query_cols
        query_cols = grouping_metrics_selection + query_cols
        query_cols = enduse_selection + query_cols
        time_col = ts_b.c[self._bsq.timestamp_column_name].label(self._bsq.timestamp_column_name)
        query_cols.insert(0, time_col)
        group_by_selection.append(time_col)
    
        query = sa.select(query_cols).select_from(tbljoin)

        query = self._bsq._add_join(query, join_list)
        query = self._bsq._add_restrict(query, restrict)
        if annual_only:
            query = query.where(self._bsq.bs_table.c["completed_status"] == "Success")
        query = self._bsq._add_group_by(query, group_by_selection)
        query = self._bsq._add_order_by(query, group_by_selection if sort else [])

        compiled_query = self._bsq._compile(query)
        if unload_to:
            if partition_by:
                compiled_query = f"UNLOAD ({compiled_query}) \n TO 's3://{unload_to}' \n "\
                                 f"WITH (format = 'PARQUET', partitioned_by = ARRAY{partition_by})"
            else:
                compiled_query = f"UNLOAD ({compiled_query}) \n TO 's3://{unload_to}' \n "\
                                 f"WITH (format = 'PARQUET')"

        if get_query_only:
            return compiled_query

        return self._bsq.execute(compiled_query, run_async=run_async)
