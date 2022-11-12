import numpy as np
from unittest.mock import MagicMock
from buildstock_query.base import BuildStockQuery
import pytest
from tests.utils import assert_query_equal, load_tbl_from_pkl, load_cache_from_pkl
import buildstock_query.core as core
from buildstock_query.utils import KWH2MBTU
import re
import tempfile

core.sa.Table = load_tbl_from_pkl  # mock the sqlalchemy table loading
core.sa.create_engine = MagicMock()  # mock creating engine
core.Connection = MagicMock() # type: ignore # NOQA
core.boto3 = MagicMock()


@pytest.fixture
def temp_history_file():
    history_file = tempfile.NamedTemporaryFile()
    name = history_file.name
    history_file.close()
    return name


class TestResStockSavings:

    @pytest.fixture
    def my_athena(self, temp_history_file):
        my_athena = BuildStockQuery(
            workgroup='eulp',
            db_name='buildstock_testing',
            buildstock_type='resstock',
            table_name='res_n250_15min_v19',
            execution_history=temp_history_file,
            skip_reports=True
        )
        my_athena._query_cache = load_cache_from_pkl('res_n250_15min_v19')
        return my_athena

    def test_get_available_upgrades(self, my_athena):
        available_upgrade = my_athena.get_available_upgrades()
        assert available_upgrade == [0, 1, 2, 3, 6, 8]

    def test_savings_shape(self, my_athena: BuildStockQuery):
        enduses = ['fuel_use_electricity_total_m_btu']
        success_report = my_athena.report.get_success_report()
        annual_savings_full = my_athena.savings.savings_shape(1, enduses=enduses)
        annual_savings_applied = my_athena.savings.savings_shape(1, enduses=enduses, applied_only=True)
        annual_bs_consumtion = my_athena.agg.aggregate_annual(enduses=enduses)
        annual_up_consumption = my_athena.agg.aggregate_annual(upgrade_id=1, enduses=enduses)
        assert len(annual_savings_full) == len(annual_savings_applied) == 1
        assert annual_savings_full['sample_count'].iloc[0] == success_report.loc[0].Success
        assert annual_savings_applied['sample_count'].iloc[0] == success_report.loc[1].Success
        assert annual_up_consumption['sample_count'].iloc[0] == success_report.loc[1].Success
        assert np.isclose(annual_savings_full[f"{enduses[0]}__baseline"],
                          annual_bs_consumtion[f"{enduses[0]}"],
                          rtol=1e-3)
        # Absolute savings between applied and full should be same
        assert np.isclose(annual_savings_applied[f"{enduses[0]}__savings"],
                          annual_savings_full[f"{enduses[0]}__savings"], rtol=1e-3)
        # The savings for applied buildings must equal diff of upgrade vs corresponding baseline
        diff = annual_savings_applied[f"{enduses[0]}__baseline"] - annual_up_consumption[f"{enduses[0]}"]
        assert np.isclose(diff, annual_savings_applied[f"{enduses[0]}__savings"], rtol=1e-3)
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        ts_savings_full = my_athena.savings.savings_shape(1, enduses=ts_enduses, annual_only=False)
        ts_savings_applied = my_athena.savings.savings_shape(
            1, enduses=ts_enduses, applied_only=True, annual_only=False)
        assert len(ts_savings_full) == len(ts_savings_applied) == 35040
        assert ts_savings_full['sample_count'].iloc[0] == success_report.loc[0].Success
        assert ts_savings_applied['sample_count'].iloc[0] == success_report.loc[1].Success
        # Match with annual result
        ts_full_bsline = ts_savings_full[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(ts_full_bsline, annual_savings_full[f"{enduses[0]}__baseline"], rtol=1e-3)
        ts_applied_bsline = ts_savings_applied[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(ts_applied_bsline, annual_savings_applied[f"{enduses[0]}__baseline"], rtol=1e-3)
        # Match savings with annual savings
        ts_full_savings = ts_savings_full[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(ts_full_savings, annual_savings_full[f"{enduses[0]}__savings"], rtol=1e-3)
        ts_applied_savings = ts_savings_applied[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(ts_applied_savings, annual_savings_applied[f"{enduses[0]}__savings"], rtol=1e-3)

    def test_savings_shape_with_grouping(self, my_athena: BuildStockQuery):
        enduses = ['fuel_use_electricity_total_m_btu']
        group_by = ["geometry_building_type_recs"]
        n_groups = 4  # Number of groups available in the dataset for the given group_by
        rtol = 2e-3  # 0.2% relative tolerance for matching annual values to timeseries values
        mbtu_atol = 0.01  # absolute mbtu tolerance per unit for closeness comparison
        # group_by = [my_athena.bs_bldgid_column]
        success_report = my_athena.report.get_success_report()
        annual_savings_full = my_athena.savings.savings_shape(1, enduses=enduses, group_by=group_by, sort=True)
        annual_savings_applied = my_athena.savings.savings_shape(1, enduses=enduses, group_by=group_by,
                                                                 applied_only=True, sort=True)
        annual_bs_consumtion = my_athena.agg.aggregate_annual(enduses=enduses, group_by=group_by, sort=True)
        annual_up_consumption = my_athena.agg.aggregate_annual(upgrade_id=1, enduses=enduses, group_by=group_by,
                                                               sort=True)
        assert len(annual_savings_full) == len(annual_savings_applied) == n_groups
        assert annual_savings_full['sample_count'].sum() == success_report.loc[0].Success
        assert annual_savings_applied['sample_count'].sum() == success_report.loc[1].Success
        assert annual_up_consumption['sample_count'].sum() == success_report.loc[1].Success
        applied_units = annual_savings_applied['units_count']
        assert np.isclose(annual_savings_full[f"{enduses[0]}__baseline"],
                          annual_bs_consumtion[f"{enduses[0]}"],
                          rtol=rtol, atol=mbtu_atol * applied_units).all()
        # Absolute savings between applied and full should be same
        assert np.isclose(annual_savings_applied[f"{enduses[0]}__savings"],
                          annual_savings_full[f"{enduses[0]}__savings"],
                          rtol=rtol, atol=mbtu_atol * applied_units).all()
        # The savings for applied buildings must equal diff of upgrade vs corresponding baseline
        diff = annual_savings_applied[f"{enduses[0]}__baseline"] - annual_up_consumption[f"{enduses[0]}"]
        assert np.isclose(diff, annual_savings_applied[f"{enduses[0]}__savings"],
                          rtol=rtol, atol=mbtu_atol * applied_units).all()
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        ts_savings_full = my_athena.savings.savings_shape(1, enduses=ts_enduses, annual_only=False, group_by=group_by)
        ts_savings_applied = my_athena.savings.savings_shape(1, enduses=ts_enduses, applied_only=True,
                                                             annual_only=False, group_by=group_by)
        assert len(ts_savings_full) == len(ts_savings_applied) == 35040 * n_groups
        assert ts_savings_full.groupby(group_by)['sample_count'].mean().sum() == success_report.loc[0].Success
        assert ts_savings_applied.groupby(group_by)['sample_count'].mean().sum() == success_report.loc[1].Success
        # Match with annual result
        ts_full_bsline = ts_savings_full.groupby(group_by)[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(ts_full_bsline, annual_savings_full[f"{enduses[0]}__baseline"],
                          rtol=rtol, atol=mbtu_atol * applied_units).all()
        ts_applied_bsline = ts_savings_applied.groupby(group_by)[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(ts_applied_bsline, annual_savings_applied[f"{enduses[0]}__baseline"],
                          rtol=rtol, atol=mbtu_atol * applied_units).all()
        # Match savings with annual savings
        ts_full_savings = ts_savings_full.groupby(group_by)[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(ts_full_savings, annual_savings_full[f"{enduses[0]}__savings"],
                          rtol=rtol, atol=mbtu_atol * applied_units).all()
        ts_applied_savings = ts_savings_applied.groupby(group_by)[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(ts_applied_savings, annual_savings_applied[f"{enduses[0]}__savings"],
                          rtol=rtol, atol=mbtu_atol * applied_units).all()

    def test_collapse_ts(self, my_athena: BuildStockQuery):
        group_by = ["geometry_building_type_recs"]
        n_groups = 4  # Number of groups available in the dataset for the given group_by
        rtol = 2e-3  # 0.2% relative tolerance for matching annual values to timeseries values
        mbtu_atol = 0.01  # absolute mbtu tolerance per unit for closeness comparison
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        ts_savings_full = my_athena.savings.savings_shape(1, enduses=ts_enduses, annual_only=False, sort=True,
                                                          group_by=group_by)
        ts_savings_applied = my_athena.savings.savings_shape(1, enduses=ts_enduses, applied_only=True,
                                                             annual_only=False, group_by=group_by, sort=True)
        ts_savings_full_collapsed = my_athena.savings.savings_shape(1, enduses=ts_enduses, annual_only=False,
                                                                    group_by=group_by, sort=True, collapse_ts=True)
        ts_savings_applied_collapsed = my_athena.savings.savings_shape(1, enduses=ts_enduses, applied_only=True,
                                                                       annual_only=False, group_by=group_by, sort=True,
                                                                       collapse_ts=True)

        assert len(ts_savings_applied_collapsed) == n_groups
        assert len(ts_savings_full_collapsed) == n_groups
        assert (ts_savings_full_collapsed['sample_count'].values ==
                ts_savings_full.groupby(group_by)['sample_count'].mean().values).all()
        assert (ts_savings_applied_collapsed['sample_count'].values ==
                ts_savings_applied.groupby(group_by)['sample_count'].mean().values).all()
        applied_units = ts_savings_applied_collapsed['units_count']
        value_cols = [f"{ts_enduses[0]}__baseline", f"{ts_enduses[0]}__savings"]
        for value_col in value_cols:
            assert np.isclose(ts_savings_full_collapsed[value_col].values,
                              ts_savings_full.groupby(group_by)[value_col].sum().values,
                              rtol=rtol, atol=mbtu_atol * applied_units).all()
            assert np.isclose(ts_savings_applied_collapsed[value_col].values,
                              ts_savings_applied.groupby(group_by)[value_col].sum().values,
                              rtol=rtol, atol=mbtu_atol * applied_units).all()

    def test_restrict(self, my_athena: BuildStockQuery):
        group_by = ["state", "geometry_building_type_recs"]
        n_groups = 1  # There is only one building type in CO in the test dataset
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        restrict = [('state', ['CO'])]
        ts_savings = my_athena.savings.savings_shape(1, enduses=ts_enduses, applied_only=True, annual_only=False,
                                                     group_by=group_by, sort=True,
                                                     restrict=restrict)
        assert len(ts_savings) == n_groups * 35040
        assert (ts_savings['state'] == 'CO').all()

    def test_unload(self, my_athena: BuildStockQuery):
        group_by = ["state", "geometry_building_type_recs"]
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        restrict = [('state', ['CO'])]
        part_by = ["geometry_building_type_recs"]
        unload_loc = "buildstock-testing/unload_test/test1/state=CO"
        ts_savings_unload_query = my_athena.savings.savings_shape(1, enduses=ts_enduses, applied_only=True,
                                                                  annual_only=False,
                                                                  group_by=group_by, sort=True,
                                                                  unload_to=unload_loc,
                                                                  partition_by=part_by,
                                                                  restrict=restrict,
                                                                  get_query_only=True)
        pattern = r"UNLOAD\s+\((.*)\)\s+TO 's3://" + unload_loc + r".* partitioned_by = ARRAY\['" + part_by[0] + r"'\]"
        match = re.match(pattern, ts_savings_unload_query, re.DOTALL)
        assert match
        ts_savings_query = my_athena.savings.savings_shape(1, enduses=ts_enduses, applied_only=True, annual_only=False,
                                                           group_by=group_by, sort=True,
                                                           restrict=restrict,
                                                           get_query_only=True)
        assert_query_equal(ts_savings_query, match.groups()[0])
