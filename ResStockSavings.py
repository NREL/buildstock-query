"""
# ResStockAthena
- - - - - - - - -
A class to calculate savings shapes for various upgrade runs.

:author: Rajendra.Adhikari@nrel.gov
:author: Noel.Merket@nrel.gov
"""
from eulpda.smart_query.ResStockAthena import ResStockAthena
import pandas as pd
import sqlalchemy as sa
from typing import List, Tuple
from sqlalchemy.sql import functions as safunc


class ResStockSavings(ResStockAthena):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._available_upgrades = None
        self._resstock_timestep = None

    @property
    def resstock_timestep(self):
        if self._resstock_timestep is None:
            sim_year, sim_interval_seconds = self._get_simulation_info()
            self._resstock_timestep = int(sim_interval_seconds // 60)
        return self._resstock_timestep

    def get_available_upgrades(self) -> dict:
        """Get the available upgrade scenarios and their identifier numbers
        :return: Upgrade scenario names
        :rtype: dict
        """
        return list(self.get_success_report().query("Success>0").index)

    def get_groupby_cols(self) -> List[str]:
        cols = set(y[21:] for y in filter(lambda x: x.startswith("build_existing_model."), self.bs_table.c.keys()))
        cols.difference_update(["applicable", "sample_weight"])
        return list(cols)

    def get_ts_enduse_cols(self) -> List[str]:
        ts_cols = list(map(str, filter(lambda x: "__" in x,
                       self.ts_table.columns.keys())))
        return ts_cols

    def get_bs_enduse_cols(self) -> List[str]:
        bs_cols = list(map(str, filter(lambda x: x.startswith("report_simulation_output.end_use") or
                       x.startswith("report_simulation_output.fuel_use"),
                       self.bs_table.columns.keys())))
        return bs_cols

    def validate_upgrade(self, upgrade_id):
        available_upgrades = self.get_available_upgrades()
        if upgrade_id not in set(available_upgrades):
            raise ValueError(f"`upgrade_id` = {upgrade_id} is not a valid upgrade.")
        return upgrade_id

    def validate_enduses(self, enduses, annual_only):
        if annual_only:
            if not enduses:
                return [self.bs_table.c["report_simulation_output.fuel_use_electricity_net_m_btu"]]
            valid_cols = set(self.get_bs_enduse_cols())
            enduses = [f"report_simulation_output.{e}" for e in enduses]
            if not set(enduses).issubset(valid_cols):
                invalid_cols = ", ".join(f'"{x}"' for x in set(enduses).difference(valid_cols))
                raise ValueError(f"The following are not valid columns in the baseline table: {invalid_cols}")
            enduses = self._get_enduse_cols(enduses, table='baseline')
        else:
            if not enduses:
                return [self.ts_table.c["fuel_use__electricity__total__kwh"]]
            valid_cols = set(self.get_ts_enduse_cols())
            if not set(enduses).issubset(valid_cols):
                invalid_cols = ", ".join(f'"{x}"' for x in set(enduses).difference(valid_cols))
                raise ValueError(f"The following are not valid columns in the timeseries table: {invalid_cols}")
            return self._get_enduse_cols(enduses, table='timeseries')

    def validate_group_by(self, group_by):
        valid_groupby_cols = self.get_groupby_cols()
        group_by_cols = [g[0] if isinstance(g, tuple) else g for g in group_by]
        if not set(group_by_cols).issubset(valid_groupby_cols):
            invalid_cols = ", ".join(f'"{x}"' for x in set(group_by).difference(valid_groupby_cols))
            raise ValueError(f"The following are not valid groupby columns in the database: {invalid_cols}")
        return group_by
        # TODO: intelligently select groupby columns order by cardinality (most to least groups) for
        # performance

    def get_timeseries_bs_up_table(self, enduses, upgrade_id, applied_only):
        ts = self.ts_table
        base = self.bs_table

        sa_ts_cols = [ts.c[self.building_id_column_name], ts.c[self.timestamp_column_name]]
        sa_ts_cols.extend(enduses)
        # adj_ts_col = sa.func.date_add("minute", -self.resstock_timestep, ts.c["time"]).label("shifted_time")

        ts_b = sa.select(sa_ts_cols).where(ts.c["upgrade"] == "0").alias("ts_b")
        ts_u = sa.select(sa_ts_cols).where(ts.c["upgrade"] == str(upgrade_id)).alias("ts_u")

        if applied_only:
            tbljoin = (
                ts_b.join(
                    ts_u, sa.and_(ts_b.c[self.building_id_column_name] == ts_u.c[self.building_id_column_name],
                                  ts_b.c[self.timestamp_column_name] == ts_u.c[self.timestamp_column_name])
                ).join(base, ts_b.c[self.building_id_column_name] == base.c[self.building_id_column_name])
            )
        else:
            tbljoin = (
                ts_b.outerjoin(
                    ts_u, sa.and_(ts_b.c[self.building_id_column_name] == ts_u.c[self.building_id_column_name],
                                  ts_b.c[self.timestamp_column_name] == ts_u.c[self.timestamp_column_name])
                ).join(base, ts_b.c[self.building_id_column_name] == base.c[self.building_id_column_name])
            )
        return ts_b, ts_u, tbljoin

    def get_annual_bs_up_table(self, enduses, upgrade_id, applied_only):
        if applied_only:
            tbljoin = (
                self.bs_table.join(
                    self.up_table, sa.and_(self.bs_table.c[self.building_id_column_name] ==
                                           self.up_table.c[self.building_id_column_name],
                                           self.up_table.c["upgrade"] == str(upgrade_id)))
            )
        else:
            tbljoin = (
                self.bs_table.outerjoin(
                    self.up_table, sa.and_(self.bs_table.c[self.building_id_column_name] ==
                                           self.up_table.c[self.building_id_column_name],
                                           self.up_table.c["upgrade"] == str(upgrade_id)))
            )
        return self.bs_table, self.up_table, tbljoin

    def savings_shape(
        self,
        upgrade_id: int,
        enduses: List[str] = None,
        group_by: List[str] = None,
        annual_only: bool = True,
        sort: bool = True,
        join_list: List[Tuple[str, str, str]] = [],
        weights: List[Tuple] = [],
        restrict: List[Tuple[str, List]] = [],
        run_async: bool = False,
        applied_only=False,
        get_query_only=False
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
         Returns:
                if get_query_only is True, returns the query_string, otherwise,
                    if run_async is True, it returns a (query_execution_id, future_object).
                    if run_async is False, it returns the result_dataframe
        """

        [self._get_tbl(jl[0]) for jl in join_list]  # ingress all tables in join list

        upgrade_id = self.validate_upgrade(upgrade_id)
        enduses = self.validate_enduses(enduses, annual_only)
        group_by = self.validate_group_by(group_by)
        total_weight = self._get_weight(weights)
        # cols_list = list(enduses)
        # groupby_list = list(group_by)
        group_by_selection = [self._get_gcol(g[0]).label(g[1]) if isinstance(
                              g, tuple) else self._get_gcol(g).label(g) for g in group_by]
        grouping_metrics_selection = [safunc.sum(1).label(
                "sample_count"), safunc.sum(1 * total_weight).label("unit_count")]

        if annual_only:
            ts_b, ts_u, tbljoin = self.get_annual_bs_up_table(enduses, upgrade_id, applied_only)
        else:
            ts_b, ts_u, tbljoin = self.get_timeseries_bs_up_table(enduses, upgrade_id, applied_only)

        if not annual_only:
            group_by_selection.append(ts_b.c[self.timestamp_column_name].label(self.timestamp_column_name))

        query_cols = list(group_by_selection)
        query_cols += grouping_metrics_selection
        for col in enduses:
            savings_col = ts_b.c[col.name] - safunc.coalesce(ts_u.c[col.name], ts_b.c[col.name])  # noqa E711
            query_cols.extend(
                [
                    sa.func.sum(ts_b.c[col.name] * total_weight).label(f"{self._simple_label(col.name)}__baseline"),
                    sa.func.sum(savings_col * total_weight).label(f"{self._simple_label(col.name)}__savings"),
                ]
            )
        query = (
            sa.select(query_cols)
            .select_from(tbljoin)
        )
        query = self._add_join(query, join_list)
        query = self._add_restrict(query, restrict)
        query = self._add_group_by(query, group_by_selection)
        query = self._add_order_by(query, group_by_selection if sort else [])
        if get_query_only:
            return self._compile(query)

        return self.execute(query, run_async=run_async)
