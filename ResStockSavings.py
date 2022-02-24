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
from typing import List, Sequence


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

    def get_available_upgrades(self, get_query_only=False) -> dict:
        """Get the available upgrade scenarios and their identifier numbers
        :return: Upgrade scenario names
        :rtype: dict
        """
        if self._available_upgrades is not None and not get_query_only:
            return self._available_upgrades
        else:
            upg_name_col = "apply_upgrade.upgrade_name"
            q = (
                sa.select(
                    [
                        self.up_table.c["upgrade"],
                        self.up_table.c[upg_name_col],
                    ]
                )
                .where(self.up_table.c[upg_name_col].isnot(None))
                .distinct()
            )
            if get_query_only:
                return self._compile(q)

            df = self.execute(q)
            df["upgrade"] = df["upgrade"].astype(int)
            self._available_upgrades = df.set_index("upgrade").sort_index()[upg_name_col].to_dict()
            return self._available_upgrades

    def get_groupby_cols(self) -> List[str]:
        cols = set(y[21:] for y in filter(lambda x: x.startswith("build_existing_model."), self.bs_table.c.keys()))
        cols.difference_update(["applicable", "sample_weight"])
        return list(cols)

    def get_electric_timeseries_cols(self) -> List[str]:
        ts_cols = list(map(str, filter(lambda x: x.startswith("end use") or x.startswith("fuel use"),
                       self.ts_table.columns.keys())))
        return ts_cols

    def savings_shape(
        self,
        upgrade_id: int,
        cols: Sequence[str] = ["fuel use: electricity: total"],
        groupby: Sequence[str] = [],
        get_query_only=False
    ) -> pd.DataFrame:
        """Calculate a savings shape for an upgrade
        :param upgrade_id: id of the upgrade scenario from the ResStock analysis
        :type upgrade_id: int
        :param cambium_scenario: The Cambium scenario to evaluate
        :type cambium_scenario: str
        :param cambium_year: Cambium projection year to use
        :type cambium_year: int
        :param cols: Electricity output columns to query, defaults to ['fuel use: electricity: total']
        :type cols: Sequence[str], optional
        :param groupby: Building characteristics columns to group by, defaults to []
        :type groupby: Sequence[str], optional
        :return: Dataframe of aggregated savings shape and carbon calculations
        :rtype: pd.DataFrame
        """
        cols_list = list(cols)
        groupby_list = list(groupby)

        available_upgrades = self.get_available_upgrades()
        if upgrade_id not in available_upgrades.keys():
            raise ValueError(f"`upgrade_id` = {upgrade_id} is not a valid upgrade.")

        valid_electric_ts_cols = self.get_electric_timeseries_cols()
        if not set(cols_list).issubset(valid_electric_ts_cols):
            invalid_cols = ", ".join(f'"{x}"' for x in set(cols_list).difference(valid_electric_ts_cols))
            raise ValueError(f"The following are not valid timeseries columns in the database: {invalid_cols}")
        valid_groupby_cols = self.get_groupby_cols()
        if not set(groupby_list).issubset(valid_groupby_cols):
            invalid_cols = ", ".join(f'"{x}"' for x in set(groupby_list).difference(valid_groupby_cols))
            raise ValueError(f"The following are not valid groupby columns in the database: {invalid_cols}")

        ts = self.ts_table
        base = self.bs_table

        sa_ts_cols = [ts.c["building_id"], ts.c["time"]]
        sa_ts_cols.extend(ts.c[col] for col in cols_list)
        adj_ts_col = sa.func.date_add("minute", -self.resstock_timestep, ts.c["time"]).label("shifted_time")

        ts_b = sa.select(sa_ts_cols + [adj_ts_col]).where(ts.c["upgrade"] == "0").alias("ts_b")
        ts_u = sa.select(sa_ts_cols).where(ts.c["upgrade"] == str(upgrade_id)).alias("ts_u")

        # FIXME: Figure out what to do about leap day
        tbljoin = (
            ts_b.outerjoin(
                ts_u, sa.and_(ts_b.c["building_id"] == ts_u.c["building_id"], ts_b.c["time"] == ts_u.c["time"])
            )
            .join(base, ts_b.c["building_id"] == base.c["building_id"])

        )

        query_cols = []
        for col in groupby_list:
            query_cols.append(base.c[f"build_existing_model.{col}"].label(col))
        query_cols.append(ts_b.c["time"])
        n_groupby = len(groupby_list) + 1
        for col in cols_list:
            savings_col = sa.case((ts_u.c[col] == None, 0.0), else_=(ts_b.c[col] - ts_u.c[col]))  # noqa E711
            query_cols.extend(
                [
                    sa.func.sum(ts_b.c[col]).label(f"{col}_baseline"),
                    sa.func.sum(savings_col).label(f"{col}_savings"),
                ]
            )

        a = [sa.text(str(x + 1)) for x in range(n_groupby)]
        # TODO: intelligently select groupby and orderby columns by order of cardinality (most to least groups) for
        # performance
        q = (
            sa.select(query_cols)
            .select_from(tbljoin)
            .group_by(*a)
            .order_by(*a)
        )
        if get_query_only:
            return self._compile(q)
        return self.execute(q)
