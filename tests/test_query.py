from __future__ import annotations

import pandas as pd
import pytest
from buildstock_query import BuildStockQuery


@pytest.fixture(scope="module")
def bsq() -> BuildStockQuery:  # pylint: disable=invalid-name
    """Shared BuildStockQuery instance for all tests."""
    obj = BuildStockQuery(
        db_name="resstock_core",
        table_name="sdr_magic17",
        workgroup="rescore",
        buildstock_type="resstock",
    )
    # Warm-up – ensures that subsequent queries can leverage the local cache when available
    obj.save_cache()
    return obj


class TestBuildStockQuery:
    # ------------------------------------------------------------------
    # Basic annual aggregations
    # ------------------------------------------------------------------

    def test_annual_electricity_agg_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
        )
        df2 = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["time", "geometry_building_type_recs"],
        )
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_annual_electricity_agg_vs_query_upg1(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            upgrade_id="1",
        )
        df2 = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["time", "geometry_building_type_recs"],
            upgrade_id="1",
        )
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_annual_electricity_agg_vs_query_quartiles(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_quartiles=True,
        )
        df2 = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["time", "geometry_building_type_recs"],
            get_quartiles=True,
        ).rename({'fuel_use_electricity_total_m_btu__upgrade__quartiles':
                  'fuel_use_electricity_total_m_btu__quartiles'}, axis=1)
        pd.testing.assert_frame_equal(df1, df2)

    def test_annual_electricity_agg_max_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            agg_func="max",
            get_nonzero_count=True,
        )
        df2 = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            agg_func="max",
            get_nonzero_count=True,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_annual_natural_gas_agg_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_natural_gas_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_nonzero_count=True,
        )
        df2 = bsq.query(
            enduses=["fuel_use_natural_gas_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_nonzero_count=True,
        )
        pd.testing.assert_frame_equal(df1, df2)

    # ------------------------------------------------------------------
    # Time-series aggregations
    # ------------------------------------------------------------------

    def test_timeseries_tx_agg_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            get_query_only=False,
        )
        df2 = bsq.query(
            annual_only=False,
            timestamp_grouping_func="month",
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            group_by=[
                "geometry_building_type_recs",
                "build_existing_model.state",
                "time",
            ],
            get_query_only=False,
        )
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_timeseries_tx_agg_vs_query_upg1(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            get_query_only=False,
            upgrade_id="1",
        )
        df2 = bsq.query(
            annual_only=False,
            timestamp_grouping_func="month",
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            group_by=[
                "geometry_building_type_recs",
                "build_existing_model.state",
                "time",
            ],
            get_query_only=False,
            upgrade_id="1"
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_peak_electricity_per_building_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            collapse_ts=True,
            agg_func="max",
            group_by=[bsq.bs_bldgid_column],
            get_query_only=False,
        )
        df2 = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="year",
            agg_func="max",
            group_by=[bsq.bs_bldgid_column],
            get_query_only=False,
        )
        pd.testing.assert_frame_equal(df1, df2)

    # ------------------------------------------------------------------
    # Savings-shape & upgrade tests
    # ------------------------------------------------------------------

    def test_savings_shape_tx_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            annual_only=False,
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            annual_only=False,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_savings_shape_geometry_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            annual_only=False,
        )
        df2 = bsq.query(
            annual_only=False,
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_quartile_savings_shape(self, bsq: BuildStockQuery):
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            get_quartiles=True,
            annual_only=False,
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            get_quartiles=True,
            annual_only=False,
        ).drop(columns=["fuel_use__electricity__total__kwh__baseline__quartiles"])
        pd.testing.assert_frame_equal(df1, df2)

    # ------------------------------------------------------------------
    # Upgrade-10 – applied vs all buildings
    # ------------------------------------------------------------------

    def _upgrade10_applied_bldgs(self, bsq: BuildStockQuery):  # helper
        return bsq.execute(
            f"select distinct(building_id) from {bsq.up_table.name} "
            "where upgrade = '10' and completed_status = 'Success'"
        )["building_id"].tolist()

    def _upgrade10_all_bldgs(self, bsq: BuildStockQuery):  # helper
        return bsq.execute(
            f"select distinct(building_id) from {bsq.up_table.name} where upgrade = '10'"
        )["building_id"].tolist()

    def test_upgrade10_applied_building_list(self, bsq: BuildStockQuery):
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            upgrade_id="10",
            group_by=["building_id"],
            applied_only=True,
        )
        expected = sorted(self._upgrade10_applied_bldgs(bsq))
        assert sorted(df["building_id"].tolist()) == expected

    def test_upgrade10_all_building_list(self, bsq: BuildStockQuery):
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            upgrade_id="10",
            group_by=["building_id"],
            applied_only=False,
        )
        expected = sorted(self._upgrade10_all_bldgs(bsq))
        assert sorted(df["building_id"].tolist()) == expected

    # ------------------------------------------------------------------
    # Time-series checks for upgrade 10
    # ------------------------------------------------------------------

    def test_timeseries_upgrade10_applied_vs_aggregate(self, bsq: BuildStockQuery):
        ts_df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            include_savings=False,
            include_baseline=False,
            include_upgrade=True,
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["building_id"],
            applied_only=True,
            get_query_only=False,
        )
        ts_df2 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["time", "building_id"],
            get_query_only=False,
        )
        pd.testing.assert_frame_equal(ts_df, ts_df2)

    def test_timeseries_upgrade10_applied_bldgs_match(self, bsq: BuildStockQuery):
        applied = self._upgrade10_applied_bldgs(bsq)
        ts_df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["building_id", "time"],
            applied_only=True,
        )
        assert sorted(set(ts_df["building_id"].tolist())) == sorted(applied)

    def test_timeseries_upgrade10_all_bldgs_match(self, bsq: BuildStockQuery):
        all_bldgs = self._upgrade10_all_bldgs(bsq)
        ts_df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["building_id", "time"],
            applied_only=False,
        )
        assert sorted(set(ts_df["building_id"].tolist())) == sorted(all_bldgs)

    # ------------------------------------------------------------------
    # Baseline verification for a single unapplied building
    # ------------------------------------------------------------------

    def test_baseline_for_unapplied_building(self, bsq: BuildStockQuery):
        all_bldgs = set(self._upgrade10_all_bldgs(bsq))
        applied = set(self._upgrade10_applied_bldgs(bsq))
        unapplied_bldgs = list(all_bldgs - applied)
        assert unapplied_bldgs, "Expected at least one unapplied building"
        ubldg = int(unapplied_bldgs[0])

        ts_df_baseline = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="0",
            group_by=["building_id", "time"],
            restrict=[("building_id", ubldg)],
        )
        ts_df_baseline_2 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="0",
            group_by=["building_id", "time"],
            restrict=[("building_id", ubldg)],
        )
        pd.testing.assert_frame_equal(ts_df_baseline, ts_df_baseline_2)

        ts_df_all = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["building_id", "time"],
            applied_only=False,
        )
        pd.testing.assert_frame_equal(
            ts_df_baseline,
            ts_df_all[ts_df_all["building_id"] == ubldg].reset_index(drop=True),
        )

    def test_baseline_column_match_applied(self, bsq: BuildStockQuery):
        all_bldgs = set(self._upgrade10_all_bldgs(bsq))
        applied = set(self._upgrade10_applied_bldgs(bsq))
        unapplied_bldgs = list(all_bldgs - applied)
        other_df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="0",
            group_by=["building_id", "time"],
            avoid=[("building_id", [int(u) for u in unapplied_bldgs])],
        ).rename(
            columns={"fuel_use__electricity__total__kwh": "fuel_use__electricity__total__kwh__baseline"}
        )
        applied_with_baseline = bsq.query(
            annual_only=False,
            include_baseline=True,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["building_id", "time"],
            applied_only=True,
        )
        pd.testing.assert_frame_equal(
            other_df[["fuel_use__electricity__total__kwh__baseline"]],
            applied_with_baseline[["fuel_use__electricity__total__kwh__baseline"]],
        )

    # ------------------------------------------------------------------
    # Simple length checks
    # ------------------------------------------------------------------

    def test_query_len_10_avoid(self, bsq: BuildStockQuery):
        all_bldgs = set(self._upgrade10_all_bldgs(bsq))
        applied = set(self._upgrade10_applied_bldgs(bsq))
        unapplied_bldgs = list(all_bldgs - applied)
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="year",
            upgrade_id="0",
            group_by=["building_id", "time"],
            avoid=[("building_id", [int(u) for u in unapplied_bldgs])],
            limit=10,
            get_nonzero_count=False,
            sort=True,
        )
        assert len(df) == 10

    def test_query_len_10_upgrade1_annual(self, bsq: BuildStockQuery):
        df = bsq.query(
            annual_only=True,
            enduses=["fuel_use_natural_gas_total_m_btu"],
            upgrade_id="1",
            group_by=[bsq.up_bldgid_column, "time"],
            limit=10,
            get_nonzero_count=False,
            sort=True,
        )
        assert len(df) == 10
