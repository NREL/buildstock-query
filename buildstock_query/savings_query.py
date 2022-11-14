"""
# ResStockAthena
- - - - - - - - -
A class to calculate savings shapes for various upgrade runs.

:author: Rajendra.Adhikari@nrel.gov
:author: Noel.Merket@nrel.gov
"""
import pandas as pd
import sqlalchemy as sa
from typing import List, Tuple
from sqlalchemy.sql import functions as safunc
import buildstock_query.main as main


class BuildStockSavings:
    def __init__(self, buildstock_query: 'main.BuildStockQuery') -> None:
        self.bsq = buildstock_query

    def _validate_partition_by(self, partition_by: list[str]):
        if not partition_by:
            return []
        [self.bsq._get_gcol(col) for col in partition_by]  # making sure all entries are valid
        return partition_by

    def __get_timeseries_bs_up_table(self, enduses: list[str], upgrade_id: int, applied_only: bool,
                                     restrict: list | None = None, ts_group_by: list[str] | None = None):
        restrict = list(restrict) if restrict else []
        ts_group_by = list(ts_group_by) if ts_group_by else []
        ts = self.bsq.ts_table
        base = self.bsq.bs_table
        sa_ts_cols = [ts.c[self.bsq.building_id_column_name], ts.c[self.bsq.timestamp_column_name]] + ts_group_by
        sa_ts_cols.extend(enduses)
        ucol = ts.c["upgrade"]
        ts_b = self.bsq._add_restrict(sa.select(sa_ts_cols), [[ucol, ("0")]] + restrict).alias("ts_b")
        ts_u = self.bsq._add_restrict(sa.select(sa_ts_cols), [[ucol, (str(upgrade_id))]] + restrict).alias("ts_u")

        if applied_only:
            tbljoin = (
                ts_b.join(
                    ts_u, sa.and_(ts_b.c[self.bsq.building_id_column_name] == ts_u.c[self.bsq.building_id_column_name],
                                  ts_b.c[self.bsq.timestamp_column_name] == ts_u.c[self.bsq.timestamp_column_name])
                ).join(base, ts_b.c[self.bsq.building_id_column_name] == base.c[self.bsq.building_id_column_name])
            )
        else:
            tbljoin = (
                ts_b.outerjoin(
                    ts_u, sa.and_(ts_b.c[self.bsq.building_id_column_name] == ts_u.c[self.bsq.building_id_column_name],
                                  ts_b.c[self.bsq.timestamp_column_name] == ts_u.c[self.bsq.timestamp_column_name])
                ).join(base, ts_b.c[self.bsq.building_id_column_name] == base.c[self.bsq.building_id_column_name])
            )
        return ts_b, ts_u, tbljoin

    def __get_annual_bs_up_table(self, upgrade_id: int, applied_only: bool):
        if applied_only:
            tbljoin = (
                self.bsq.bs_table.join(
                    self.bsq.up_table, sa.and_(self.bsq.bs_table.c[self.bsq.building_id_column_name] ==
                                               self.bsq.up_table.c[self.bsq.building_id_column_name],
                                               self.bsq.up_table.c["upgrade"] == str(upgrade_id),
                                               self.bsq.up_table.c["completed_status"] == "Success"))
            )
        else:
            tbljoin = (
                self.bsq.bs_table.outerjoin(
                    self.bsq.up_table, sa.and_(self.bsq.bs_table.c[self.bsq.building_id_column_name] ==
                                               self.bsq.up_table.c[self.bsq.building_id_column_name],
                                               self.bsq.up_table.c["upgrade"] == str(upgrade_id),
                                               self.bsq.up_table.c["completed_status"] == "Success")))

        return self.bsq.bs_table, self.bsq.up_table, tbljoin

    def savings_shape(
        self,
        upgrade_id: int,
        enduses: List[str] | None = None,
        group_by: List[str] | None = None,
        annual_only: bool = True,
        sort: bool = True,
        join_list: List[Tuple[str, str, str]] | None = None,
        weights: List[Tuple] | None = None,
        restrict: List[Tuple[str, List]] | None = None,
        run_async: bool = False,
        applied_only: bool = False,
        get_quartiles: bool = False,
        get_query_only: bool = False,
        unload_to: str = '',
        partition_by: List[str] | None = None,
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

        [self.bsq.get_table(jl[0]) for jl in join_list]  # ingress all tables in join list

        upgrade_id = self.bsq.validate_upgrade(upgrade_id)
        enduse_cols = self.bsq._get_enduse_cols(enduses, table="baseline" if annual_only else "timeseries")
        partition_by = self._validate_partition_by(partition_by)
        total_weight = self.bsq._get_weight(weights)
        group_by_selection = self.bsq._process_groupby_cols(group_by, annual_only=annual_only)

        if annual_only:
            ts_b, ts_u, tbljoin = self.__get_annual_bs_up_table(upgrade_id, applied_only)
        else:
            restrict, ts_restrict = self.bsq._split_restrict(restrict)
            bs_group_by, ts_group_by = self.bsq._split_group_by(group_by_selection)
            ts_b, ts_u, tbljoin = self.__get_timeseries_bs_up_table(enduse_cols, upgrade_id, applied_only, ts_restrict,
                                                                    ts_group_by)
            ts_group_by = [ts_b.c[c.name] for c in ts_group_by]  # Refer to the columns using ts_b table
            group_by_selection = bs_group_by + ts_group_by
        query_cols = []
        for col in enduse_cols:
            savings_col = ts_b.c[col.name] - safunc.coalesce(ts_u.c[col.name], ts_b.c[col.name])  # noqa E711
            query_cols.extend(
                [
                    sa.func.sum(ts_b.c[col.name] * total_weight).label(f"{self.bsq._simple_label(col.name)}__baseline"),
                    sa.func.sum(savings_col * total_weight).label(f"{self.bsq._simple_label(col.name)}__savings"),
                ]
            )
            if get_quartiles:
                query_cols.extend(
                    [sa.func.approx_percentile(savings_col, [0, 0.02, 0.25, 0.5, 0.75, 0.98, 1]).
                     label(f"{self.bsq._simple_label(col.name)}__savings__quartiles")
                     ]
                )

        query_cols.extend(group_by_selection)
        if annual_only:  # Use annual tables
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(1 * total_weight).label("units_count")]
            query_cols = grouping_metrics_selection + query_cols
        elif collapse_ts:  # Use timeseries tables but collapse timeseries
            rows_per_building = self.bsq._get_rows_per_building()
            grouping_metrics_selection = [(safunc.sum(1) / rows_per_building).label(
                "sample_count"), safunc.sum(total_weight / rows_per_building).label("units_count")]
            query_cols = grouping_metrics_selection + query_cols
        else:  # Use timeseries table and return timeseries results
            grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(1 * total_weight).label("units_count")]
            query_cols = grouping_metrics_selection + query_cols
            time_col = ts_b.c[self.bsq.timestamp_column_name].label(self.bsq.timestamp_column_name)
            query_cols.insert(0, time_col)
            group_by_selection.append(time_col)

        query = sa.select(query_cols).select_from(tbljoin)

        query = self.bsq._add_join(query, join_list)
        query = self.bsq._add_restrict(query, restrict)
        if annual_only:
            query = query.where(self.bsq.bs_table.c["completed_status"] == "Success")
        query = self.bsq._add_group_by(query, group_by_selection)
        query = self.bsq._add_order_by(query, group_by_selection if sort else [])

        compiled_query = self.bsq._compile(query)
        if unload_to:
            if partition_by:
                compiled_query = f"UNLOAD ({compiled_query}) \n TO 's3://{unload_to}' \n "\
                                 f"WITH (format = 'PARQUET', partitioned_by = ARRAY{partition_by})"
            else:
                compiled_query = f"UNLOAD ({compiled_query}) \n TO 's3://{unload_to}' \n "\
                                 f"WITH (format = 'PARQUET')"

        if get_query_only:
            return compiled_query

        return self.bsq.execute(compiled_query, run_async=run_async)
