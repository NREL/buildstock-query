from unittest.mock import MagicMock
import tempfile
import pytest
import buildstock_query.query_core as query_core
from buildstock_query.main import BuildStockQuery
from tests.utils import assert_query_equal, load_tbl_from_pkl

query_core.sa.Table = load_tbl_from_pkl  # mock the sqlalchemy table loading
query_core.sa.create_engine = MagicMock()  # mock creating engine
query_core.Connection = MagicMock()  # type: ignore # NOQA
query_core.boto3 = MagicMock()
query_core.Connection.cursor = MagicMock()


@pytest.fixture
def temp_history_file():
    history_file = tempfile.NamedTemporaryFile()
    name = history_file.name
    history_file.close()
    return name


def test_aggregated_ts_by_eiaid(temp_history_file: str):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )
    my_athena._conn.cursor = MagicMock()  # type: ignore
    query = my_athena.utility.aggregate_ts_by_eiaid(eiaid_list=['1121', '1123'],
                                                    enduses=['end use: electricity: cooling',
                                                             'end use: electricity: heating'],
                                                    group_by=['time'],
                                                    get_query_only=True,
                                                    query_group_size=1)

    expected_query1 = """
    select "eiaid_weights"."eiaid" as "eiaid", "res_n250_hrly_v1_timeseries"."time" as "time",  sum(1) as "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight")
    as "units_count", sum("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight") as "end use: electricity: cooling", sum("res_n250_hrly_v1_timeseries"."end use: electricity: heating" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" *
    "eiaid_weights"."weight") as "end use: electricity: heating" from "res_n250_hrly_v1_timeseries" join
    "res_n250_hrly_v1_baseline" on "res_n250_hrly_v1_baseline"."building_id" =
    "res_n250_hrly_v1_timeseries"."building_id" join "eiaid_weights" on
    "res_n250_hrly_v1_baseline"."build_existing_model.county" = "eiaid_weights"."county"
    where "eiaid_weights"."eiaid" = '1121' group by 1, 2 order by 1, 2
    """  # noqa: E501

    expected_query2 = """
    select "eiaid_weights"."eiaid" as "eiaid", "res_n250_hrly_v1_timeseries"."time" as "time",  sum(1) as "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight")
    as "units_count", sum("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight") as "end use: electricity: cooling", sum("res_n250_hrly_v1_timeseries"."end use: electricity: heating" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" *
    "eiaid_weights"."weight") as "end use: electricity: heating" from "res_n250_hrly_v1_timeseries" join
    "res_n250_hrly_v1_baseline" on "res_n250_hrly_v1_baseline"."building_id" =
    "res_n250_hrly_v1_timeseries"."building_id"  join "eiaid_weights" on
    "res_n250_hrly_v1_baseline"."build_existing_model.county" = "eiaid_weights"."county" where
    "eiaid_weights"."eiaid" = '1123' group by 1, 2 order by 1, 2
    """  # noqa: E501
    assert len(query) == 2
    assert_query_equal(query[0].lower(), expected_query1)
    assert_query_equal(query[1].lower(), expected_query2)

    query = my_athena.utility.aggregate_ts_by_eiaid(eiaid_list=['1121', '1123'],
                                                    enduses=['end use: electricity: cooling',
                                                             'end use: electricity: heating'],
                                                    group_by=['time'],
                                                    get_query_only=True,
                                                    )
    expected_query3 = """
    select "eiaid_weights"."eiaid" as "eiaid", "res_n250_hrly_v1_timeseries"."time" as "time",  sum(1) as "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight")
    as "units_count", sum("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight") as "end use: electricity: cooling", sum("res_n250_hrly_v1_timeseries"."end use: electricity: heating" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" *
    "eiaid_weights"."weight") as "end use: electricity: heating" from "res_n250_hrly_v1_timeseries" join
    "res_n250_hrly_v1_baseline" on "res_n250_hrly_v1_baseline"."building_id" =
    "res_n250_hrly_v1_timeseries"."building_id" join "eiaid_weights" on
    "res_n250_hrly_v1_baseline"."build_existing_model.county" = "eiaid_weights"."county"
    where "eiaid_weights"."eiaid" in ('1121', '1123') group by 1, 2 order by 1, 2
    """  # noqa: E501
    assert len(query) == 1
    assert_query_equal(query[0], expected_query3)


def test_aggregate_unit_counts_by_eiaid(temp_history_file: str):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )

    query = my_athena.utility.aggregate_unit_counts_by_eiaid(eiaid_list=['1121', '1123'],
                                                             group_by=[
                                                                 'build_existing_model.geometry_building_type_recs'],
                                                             get_query_only=True)

    expected_query = """
    select "eiaid_weights"."eiaid" as "eiaid", "res_n250_hrly_v1_baseline"."build_existing_model.geometry_building_type_recs" as "geometry_building_type_recs",  sum(1) as "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight")
    as "units_count" from "res_n250_hrly_v1_baseline" join "eiaid_weights" on "res_n250_hrly_v1_baseline"."build_existing_model.county" =
    "eiaid_weights"."county" where "res_n250_hrly_v1_baseline"."completed_status" = 'Success' and "eiaid_weights"."eiaid" in ('1121', '1123') group by 1, 2 order by 1, 2
    """  # noqa: E501

    assert_query_equal(query, expected_query)


def test_aggregate_annual_by_eiaid(temp_history_file: str):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )
    enduses = ['report_simulation_output.end_use_electricity_cooling_m_btu',
               'report_simulation_output.end_use_electricity_heating_m_btu']
    query = my_athena.utility.aggregate_annual_by_eiaid(enduses=enduses,
                                                        group_by=['build_existing_model.geometry_building_type_recs'],
                                                        get_query_only=True)

    expected_query = """
    select "eiaid_weights"."eiaid" as "eiaid", "res_n250_hrly_v1_baseline"."build_existing_model.geometry_building_type_recs" as "geometry_building_type_recs",  sum(1) as "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight")
    as "units_count", sum("res_n250_hrly_v1_baseline"."report_simulation_output.end_use_electricity_cooling_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight") as "end_use_electricity_cooling_m_btu", sum("res_n250_hrly_v1_baseline"."report_simulation_output.end_use_electricity_heating_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" *
    "eiaid_weights"."weight") as "end_use_electricity_heating_m_btu" from "res_n250_hrly_v1_baseline" join
    "eiaid_weights" on "res_n250_hrly_v1_baseline"."build_existing_model.county" = "eiaid_weights"."county" where "res_n250_hrly_v1_baseline"."completed_status" = 'Success' group by 1, 2 order by 1, 2
    """  # noqa: E501

    assert_query_equal(query, expected_query)


def test_get_buildings_by_eiaid(temp_history_file: str):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )

    query = my_athena.utility.get_buildings_by_eiaids(eiaids=['1123', '1234'],
                                                      get_query_only=True)

    expected_query = """
    select distinct "res_n250_hrly_v1_baseline"."building_id" from "res_n250_hrly_v1_baseline" join "eiaid_weights" on
    "res_n250_hrly_v1_baseline"."build_existing_model.county" = "eiaid_weights"."county" where "eiaid_weights"."eiaid" in ('1123', '1234') and "eiaid_weights"."weight" > 0 order by "res_n250_hrly_v1_baseline"."building_id"
    """  # noqa: E501

    assert_query_equal(query, expected_query)


def test_get_filtered_results_csvs_by_eiaid(temp_history_file: str):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )

    query = my_athena.utility.get_filtered_results_csv_by_eiaid(['4110', '1167', '3249'],
                                                                get_query_only=True)

    expected_query = """
    select * from "res_n250_hrly_v1_baseline" join "eiaid_weights" on "res_n250_hrly_v1_baseline"."build_existing_model.county" = "eiaid_weights"."county"
    where "eiaid_weights"."eiaid" in ('4110', '1167', '3249') and "eiaid_weights"."weight" > 0
    """  # noqa: E501

    assert_query_equal(query, expected_query)


def test_get_locations_by_eiaids(temp_history_file: str):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )

    query = my_athena.utility.get_locations_by_eiaids(eiaids=['4110', '1167', '3249'],
                                                      get_query_only=True)

    expected_query = """
    select distinct "eiaid_weights"."county" from "eiaid_weights" where "eiaid_weights"."eiaid" in ('4110', '1167', '3249') and "eiaid_weights"."weight" > 0
    """  # noqa: E501

    assert_query_equal(query, expected_query)


def test_get_buildings_by_county(temp_history_file: str):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )

    query = my_athena.get_buildings_by_locations(location_col="build_existing_model.county",
                                                 locations=['loc1', 'loc2'], get_query_only=True)

    expected_query = """
    select "res_n250_hrly_v1_baseline"."building_id" from "res_n250_hrly_v1_baseline" where "res_n250_hrly_v1_baseline"."build_existing_model.county" in ('loc1', 'loc2')
    order by 1
    """  # noqa: E501
    assert_query_equal(query, expected_query)
