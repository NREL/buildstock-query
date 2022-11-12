import contextlib
from pyathena.pandas.result_set import AthenaPandasResultSet
from unittest.mock import MagicMock
import tempfile
import pytest
from tests.utils import assert_query_equal, load_tbl_from_pkl
from buildstock_query.helpers import FutureDf
import buildstock_query.query_core as query_core
from buildstock_query.main import BuildStockQuery
import pandas as pd
import uuid
import time
query_core.sa.Table = load_tbl_from_pkl  # mock the sqlalchemy table loading
query_core.sa.create_engine = MagicMock()  # mock creating engine
query_core.Connection = MagicMock()  # type: ignore # NOQA
query_core.boto3 = MagicMock()


@pytest.fixture
def temp_history_file():
    history_file = tempfile.NamedTemporaryFile()
    name = history_file.name
    history_file.close()
    return name


class FunctionNotCalledException(Exception):
    pass


DEFAULT_DF = pd.DataFrame({'col1': [1, 2], 'col2': [10, 20]})


class MockResultSet(AthenaPandasResultSet):
    def __init__(self, df):
        self.saved_df = df

    def as_pandas(self):
        return self.saved_df

    @property
    def state(self):
        return "SUCCEEDED"


def fake_sync_executer(query, *args, run_async=False, **kwargs):
    df = DEFAULT_DF.copy()
    with contextlib.suppress(TypeError, IndexError, ValueError):
        val = float(query.split(',')[-1])
        df['val'] = val
    return MockResultSet(df)


def fake_async_executor(query, *args, run_async=False, **kwargs):
    return str(uuid.uuid4()), FutureDf(fake_sync_executer(query))


fake_async_cursor = MagicMock()
fake_async_cursor.execute = fake_async_executor
fake_sync_cursor = MagicMock()
fake_sync_cursor.execute = fake_sync_executer


def assert_mock_func_call(mock_obj, function, *args, **kwargs):
    for call in mock_obj.mock_calls:
        call_function = call[0].split('.')[-1]  # 0 is for the function name
        if call_function == function:
            if call[1] != args:  # 1 is args
                continue
            for key in kwargs:
                # if kwargs is supplied, each key must be present and correct argument must be supplied in the function
                if key in call[2] and call[2][key] == kwargs[key]:
                    return
                else:
                    break
            else:  # Doesn't enter else in case of break. So, this occurs only when no kwargs is supplied
                return
    raise FunctionNotCalledException(f'Function {function} not called.')


def assert_list_equal(list1, list2):
    assert len(list1) == len(list2)
    for i, val in enumerate(list1):
        assert val == list2[i]


def test_clean_group_by(temp_history_file):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )
    group_by = ['time', '"res_national_53_2018_baseline"."build_existing_model.state"',
                '"build_existing_model.county"']
    clean_group_by = my_athena._clean_group_by(group_by)
    assert clean_group_by == ['time', 'build_existing_model.state', 'build_existing_model.county']

    # Test tuple cleaning
    group_by = ['time', ('month(time)', 'MOY'),
                '"build_existing_model.county"']
    clean_group_by = my_athena._clean_group_by(group_by)
    assert clean_group_by == ['time', 'MOY', 'build_existing_model.county']

# Side effect necessary to make it return new Mock objects for different calls


def test_query_execution_pass_through(temp_history_file):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )
    my_athena._log_execution_cost = MagicMock()
    my_athena._async_conn.cursor = MagicMock(return_value=fake_async_cursor)

    qid, future = my_athena.execute("Some mock query", run_async=True)
    # Mock the list_query_executions function to return the queryID.
    # It needs be mocked because athena API library is mocked.
    my_athena._aws_athena.list_query_executions = lambda WorkGroup: {'QueryExecutionIds': ['id1', 'id2', qid]}
    my_athena.get_query_status = lambda _: 'RUNNING'
    my_athena.stop_all_queries()
    assert_mock_func_call(my_athena._aws_athena, 'stop_query_execution', QueryExecutionId=str(qid))

    # Test that queries not running under this user is not stopped
    with pytest.raises(FunctionNotCalledException):
        assert_mock_func_call(my_athena._aws_athena, 'stop_query_execution', QueryExecutionId='id1')

    # Test that the query returns proper dataframe
    my_athena._conn.cursor = MagicMock(return_value=fake_sync_cursor)
    pd.testing.assert_frame_equal(future.as_pandas(), DEFAULT_DF)
    df = my_athena.execute("Some mock query", run_async=False)
    pd.testing.assert_frame_equal(df, DEFAULT_DF)


def test_aggregate_annual(temp_history_file):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )

    enduses = ["report_simulation_output.fuel_use_electricity_net_m_btu",
               "report_simulation_output.end_use_electricity_cooling_m_btu"]

    state_str = 'build_existing_model.state'
    bldg_type = 'build_existing_model.geometry_building_type_recs'

    query1 = my_athena.agg.aggregate_annual(enduses=enduses,
                                            group_by=[state_str, bldg_type],
                                            sort=True,
                                            run_async=True,
                                            get_query_only=True)

    valid_query_string = """
        select "res_n250_hrly_v1_baseline"."build_existing_model.state" as "state", "res_n250_hrly_v1_baseline"."build_existing_model.geometry_building_type_recs" as "geometry_building_type_recs",
        sum(1) as "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "units_count",
        sum("res_n250_hrly_v1_baseline"."report_simulation_output.fuel_use_electricity_net_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "fuel_use_electricity_net_m_btu",
        sum("res_n250_hrly_v1_baseline"."report_simulation_output.end_use_electricity_cooling_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "end_use_electricity_cooling_m_btu" from "res_n250_hrly_v1_baseline" where "res_n250_hrly_v1_baseline"."completed_status" = 'Success'
        group by 1, 2
        order by 1, 2
        """  # noqa: E501
    assert_query_equal(query1, valid_query_string)  # Test that proper query is formed for annual aggregation

    query1_1 = my_athena.agg.aggregate_annual(enduses=enduses,
                                              group_by=[(state_str, 'state'), bldg_type],
                                              sort=True,
                                              run_async=True,
                                              get_query_only=True)

    valid_query_string1_1 = """
            select "res_n250_hrly_v1_baseline"."build_existing_model.state" as "state", "res_n250_hrly_v1_baseline"."build_existing_model.geometry_building_type_recs"  as "geometry_building_type_recs",
            sum(1) as "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "units_count",
            sum("res_n250_hrly_v1_baseline"."report_simulation_output.fuel_use_electricity_net_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "fuel_use_electricity_net_m_btu",
            sum("res_n250_hrly_v1_baseline"."report_simulation_output.end_use_electricity_cooling_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "end_use_electricity_cooling_m_btu" from "res_n250_hrly_v1_baseline" where "res_n250_hrly_v1_baseline"."completed_status" = 'Success'
            group by 1, 2
            order by 1, 2
            """  # noqa: E501
    assert_query_equal(query1_1, valid_query_string1_1)  # Test that proper query is formed for annual aggregation
    eiaid_col = my_athena.get_column("eiaid", "eiaid_weights")
    query2 = my_athena.agg.aggregate_annual(enduses=enduses,
                                            group_by=[eiaid_col, state_str, bldg_type],
                                            sort=True,
                                            join_list=[
                                                (
                                                    'eiaid_weights', 'build_existing_model.county',
                                                    'county')],
                                            weights=[('weight', 'eiaid_weights')],
                                            restrict=[('eiaid', ['1167', '3249']),
                                                      (state_str, ['AL', 'VA', 'TX'])],
                                            run_async=True,
                                            get_query_only=True)

    valid_query_string2 = """
        select "eiaid_weights"."eiaid" as "eiaid", "res_n250_hrly_v1_baseline"."build_existing_model.state"  as "state", "res_n250_hrly_v1_baseline"."build_existing_model.geometry_building_type_recs" as "geometry_building_type_recs",
        sum(1) as "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight") as "units_count",
        sum("res_n250_hrly_v1_baseline"."report_simulation_output.fuel_use_electricity_net_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" *
        "eiaid_weights"."weight") as "fuel_use_electricity_net_m_btu",
        sum("res_n250_hrly_v1_baseline"."report_simulation_output.end_use_electricity_cooling_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" *
        "eiaid_weights"."weight") as "end_use_electricity_cooling_m_btu"
        from "res_n250_hrly_v1_baseline" join "eiaid_weights" on "res_n250_hrly_v1_baseline"."build_existing_model.county" = "eiaid_weights"."county"
        where "res_n250_hrly_v1_baseline"."completed_status" = 'Success' and "eiaid_weights"."eiaid" in ('1167', '3249') and "res_n250_hrly_v1_baseline"."build_existing_model.state" in ('AL', 'VA', 'TX')
        group by 1, 2, 3
        order by 1, 2, 3
        """  # noqa: E501
    assert_query_equal(query2, valid_query_string2)  # Test that proper query is formed for annual aggregation

    query3 = my_athena.agg.aggregate_annual(enduses=enduses,
                                            run_async=True,
                                            get_query_only=True)
    valid_query_string3 = """
        select sum(1) AS "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") AS "units_count", sum("res_n250_hrly_v1_baseline"."report_simulation_output.fuel_use_electricity_net_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "fuel_use_electricity_net_m_btu",
        sum("res_n250_hrly_v1_baseline"."report_simulation_output.end_use_electricity_cooling_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "end_use_electricity_cooling_m_btu" from "res_n250_hrly_v1_baseline" where "res_n250_hrly_v1_baseline"."completed_status" = 'Success'
        """  # noqa: E501
    assert_query_equal(query3, valid_query_string3)

    valid_query_string4 = """
        select sum(1) AS "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight") AS "units_count", sum("res_n250_hrly_v1_baseline"."report_simulation_output.fuel_use_electricity_net_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" *
        "eiaid_weights"."weight") as "fuel_use_electricity_net_m_btu",
        sum("res_n250_hrly_v1_baseline"."report_simulation_output.end_use_electricity_cooling_m_btu" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" *
        "eiaid_weights"."weight") as "end_use_electricity_cooling_m_btu" from
        "res_n250_hrly_v1_baseline" join "eiaid_weights" on "res_n250_hrly_v1_baseline"."build_existing_model.county" = "eiaid_weights"."county" where
        "res_n250_hrly_v1_baseline"."completed_status" = 'Success' and "eiaid_weights"."eiaid" in ('1167', '3249') and "res_n250_hrly_v1_baseline"."build_existing_model.state" in ('AL', 'VA', 'TX')
        """  # noqa: E501
    query4 = my_athena.agg.aggregate_annual(enduses=enduses,
                                            join_list=[
                                                (
                                                    'eiaid_weights', 'build_existing_model.county',
                                                    'county')],
                                            weights=["weight"],
                                            restrict=[('eiaid', ['1167', '3249']),
                                                      (state_str, ['AL', 'VA', 'TX'])],
                                            run_async=True,
                                            get_query_only=True)

    assert_query_equal(query4, valid_query_string4)

    # Custom sample weight
    my_athena2 = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        sample_weight=29.0,
        execution_history=temp_history_file,
        skip_reports=True
    )
    query5 = my_athena2.agg.aggregate_annual(enduses=enduses,
                                             run_async=True,
                                             get_query_only=True,
                                             )
    valid_query_string5 = """
        select sum(1) AS "sample_count", sum(29.0) AS "units_count", sum("res_n250_hrly_v1_baseline"."report_simulation_output.fuel_use_electricity_net_m_btu" * 29.0) as "fuel_use_electricity_net_m_btu",
        sum("res_n250_hrly_v1_baseline"."report_simulation_output.end_use_electricity_cooling_m_btu" * 29.0) as
        "end_use_electricity_cooling_m_btu" from "res_n250_hrly_v1_baseline" where "res_n250_hrly_v1_baseline"."completed_status" = 'Success'
    """  # noqa: E501
    assert_query_equal(query5, valid_query_string5)


def test_aggregate_ts(temp_history_file):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )
    my_athena.get_available_upgrades = lambda: [0]
    enduses = ["fuel use: electricity: total", "end use: electricity: cooling"]
    state_str = "build_existing_model.state"
    bldg_type = "build_existing_model.geometry_building_type_recs"
    query1 = my_athena.agg.aggregate_timeseries(enduses=enduses,
                                                group_by=['time', state_str, bldg_type],
                                                sort=True,
                                                run_async=True,
                                                get_query_only=True)
    valid_query_string1 = """
    select "res_n250_hrly_v1_timeseries"."time" as "time", "res_n250_hrly_v1_baseline"."build_existing_model.state" as "state", "res_n250_hrly_v1_baseline"."build_existing_model.geometry_building_type_recs" as "geometry_building_type_recs",  sum(1) as
    "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "units_count", sum("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" *
    "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "fuel use: electricity: total",
    sum("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as
    "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join "res_n250_hrly_v1_baseline" on
    "res_n250_hrly_v1_baseline"."building_id" =
    "res_n250_hrly_v1_timeseries"."building_id"  group by 1, 2, 3 order by 1, 2, 3
    """  # noqa: E501
    assert_query_equal(query1, valid_query_string1)  # Test that proper query is formed for timeseries aggregation

    query1_1 = my_athena.agg.aggregate_timeseries(enduses=enduses,
                                                  group_by=['time', (state_str, 'state'), bldg_type],
                                                  sort=True,
                                                  run_async=True,
                                                  get_query_only=True)
    valid_query_string1_1 = """
        select "res_n250_hrly_v1_timeseries"."time" as "time", "res_n250_hrly_v1_baseline"."build_existing_model.state" as "state",
        "res_n250_hrly_v1_baseline"."build_existing_model.geometry_building_type_recs" as "geometry_building_type_recs",  sum(1) as
        "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "units_count", sum("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" *
        "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "fuel use: electricity: total",
        sum("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight")
        as "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join "res_n250_hrly_v1_baseline" on
        "res_n250_hrly_v1_baseline"."building_id" =
        "res_n250_hrly_v1_timeseries"."building_id"  group by 1, 2, 3 order by 1, 2, 3
        """  # noqa: E501
    assert_query_equal(query1_1, valid_query_string1_1)  # Using tuple group_by

    enduses = ["fuel use: electricity: total", "end use: electricity: cooling"]
    state_str = "build_existing_model.state"
    bldg_type = "build_existing_model.geometry_building_type_recs"
    query2 = my_athena.agg.aggregate_timeseries(enduses=enduses,
                                                group_by=['eiaid', state_str, bldg_type, 'time'],
                                                sort=True,
                                                join_list=[
                                                    ('eiaid_weights', 'build_existing_model.county',
                                                     'county')],
                                                weights=["weight"],
                                                restrict=[('eiaid', ['1167', '3249']),
                                                          (state_str, ['AL', 'VA', 'TX'])],
                                                run_async=True,
                                                get_query_only=True)
    valid_query_string2 = """
            select "eiaid_weights"."eiaid" as "eiaid", "res_n250_hrly_v1_baseline"."build_existing_model.state" as "state", "res_n250_hrly_v1_baseline"."build_existing_model.geometry_building_type_recs" as "geometry_building_type_recs",
            "res_n250_hrly_v1_timeseries"."time" as "time",  sum(1) as "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight") as
            "units_count", sum("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" *
            "eiaid_weights"."weight") as "fuel use: electricity: total",
            sum("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight") as "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join
            "res_n250_hrly_v1_baseline" on "res_n250_hrly_v1_baseline"."building_id" = "res_n250_hrly_v1_timeseries"."building_id"  join
            "eiaid_weights" on "res_n250_hrly_v1_baseline"."build_existing_model.county" = "eiaid_weights"."county"
            where "eiaid_weights"."eiaid" in ('1167', '3249') and "res_n250_hrly_v1_baseline"."build_existing_model.state" in ('AL', 'VA', 'TX')
            group by 1, 2, 3, 4 order by 1, 2, 3, 4
            """  # noqa: E501
    assert_query_equal(query2, valid_query_string2)  # Test that proper query is formed for timeseries aggregation

    # test without grouping
    my_athena._get_rows_per_building = lambda: 35040

    query3 = my_athena.agg.aggregate_timeseries(enduses=enduses,
                                                run_async=True,
                                                collapse_ts=True,
                                                get_query_only=True)
    valid_query_string3 = """
        select sum(1) / 35040 as "sample_count", sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" / 35040) as "units_count",
        sum("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "fuel use: electricity: total",
        sum("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join
        "res_n250_hrly_v1_baseline" on "res_n250_hrly_v1_baseline"."building_id" = "res_n250_hrly_v1_timeseries"."building_id"
        """  # noqa: E501
    assert_query_equal(query3, valid_query_string3)

    enduses = ["fuel use: electricity: total", "end use: electricity: cooling"]
    state_str = "build_existing_model.state"
    query4 = my_athena.agg.aggregate_timeseries(enduses=enduses,
                                                join_list=[
                                                    ('eiaid_weights', 'build_existing_model.county',
                                                     'county')],
                                                weights=[('weight', 'eiaid_weights')],
                                                restrict=[('eiaid', ['1167', '3249']),
                                                          (state_str, ['AL', 'VA', 'TX'])],
                                                run_async=True,
                                                collapse_ts=True,
                                                get_query_only=True)
    valid_query_string4 = """
        select sum(1) / 35040 as "sample_count", sum(("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight") / 35040) as
         "units_count",
        sum("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight") as "fuel use: electricity: total", sum("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" *
        "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * "eiaid_weights"."weight")
        as "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join "res_n250_hrly_v1_baseline" on
        "res_n250_hrly_v1_baseline"."building_id" = "res_n250_hrly_v1_timeseries"."building_id"  join "eiaid_weights" on
        "res_n250_hrly_v1_baseline"."build_existing_model.county" = "eiaid_weights"."county" where "eiaid_weights"."eiaid" in
        ('1167', '3249') and "res_n250_hrly_v1_baseline"."build_existing_model.state" in ('AL', 'VA', 'TX')
        """  # noqa: E501
    assert_query_equal(query4, valid_query_string4)  # Test that proper query is formed for timeseries aggregation

    my_athena2 = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        sample_weight=29.0,
        skip_reports=True
    )
    my_athena2.get_available_upgrades = lambda: [0]
    my_athena2._get_rows_per_building = lambda: 35040

    query5 = my_athena2.agg.aggregate_timeseries(enduses=enduses,
                                                 collapse_ts=True,
                                                 run_async=True,
                                                 get_query_only=True)
    valid_query_string5 = """
            select sum(1) / 35040 as "sample_count", sum(29.0 / 35040) as "units_count",
            sum("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" * 29.0) as
            "fuel use: electricity: total", sum("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * 29.0)
            as "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join "res_n250_hrly_v1_baseline" on
            "res_n250_hrly_v1_baseline"."building_id" = "res_n250_hrly_v1_timeseries"."building_id"
            """  # noqa: E501
    assert_query_equal(query5, valid_query_string5)


def test_batch_query(temp_history_file):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )
    my_athena._async_conn.cursor = MagicMock(return_value=fake_async_cursor)

    queries = ["select * from mocked_query1, 12.1", "select 2 from mocked_query2, 13.2"]
    batch_id = my_athena.submit_batch_query(queries)
    time.sleep(0.1)  # Wait for the threads to run those queries
    report = my_athena.get_batch_query_report(batch_id)
    execution_ids = my_athena._batch_query_status_map[batch_id]['submitted_execution_ids']
    assert report['Submitted'] == 2
    assert len(execution_ids) == 2
    my_athena.did_batch_query_complete = lambda _: True
    my_athena.get_batch_query_report = lambda _: {'Submitted': 2, 'Completed': 2, 'Running': 0, 'Pending': 0,
                                                  'Failed': 0}
    batch_result = my_athena.get_batch_query_result(batch_id)
    df1, df2 = DEFAULT_DF.copy(), DEFAULT_DF.copy()
    df1['val'], df2['val'] = 12.1, 13.2
    df1['query_id'], df2['query_id'] = 0, 1
    full_result = pd.concat([df1, df2])
    pd.testing.assert_frame_equal(batch_result.reset_index(drop=True),
                                  full_result.reset_index(drop=True))

    # TODO: Add test for when a batch query partially fails and some queries are resubmitted.


def test_get_results_csv(temp_history_file):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )
    query1 = my_athena.get_results_csv(get_query_only=True)
    valid_query_string1 = """
        select * from "res_n250_hrly_v1_baseline"
        """
    assert_query_equal(query1, valid_query_string1)

    query2 = my_athena.get_results_csv(restrict=[('building_id', (549, 487, 759)),
                                                 ('build_existing_model.geometry_foundation_type', 'Heated Basement')],
                                       get_query_only=True)
    valid_query_string2 = """
        select * from "res_n250_hrly_v1_baseline"  where "res_n250_hrly_v1_baseline"."building_id" in (549, 487, 759) and
        "res_n250_hrly_v1_baseline"."build_existing_model.geometry_foundation_type" = 'Heated Basement'
    """  # noqa: E501
    assert_query_equal(query2, valid_query_string2)


def test_get_building_average_kws_at(temp_history_file):
    my_athena = BuildStockQuery(
        workgroup='eulp',
        db_name='buildstock_testing',
        buildstock_type='resstock',
        table_name='res_n250_hrly_v1',
        execution_history=temp_history_file,
        skip_reports=True
    )
    enduses = ["fuel use: electricity: total", "end use: electricity: cooling"]
    my_athena._get_simulation_info = lambda: (2012, 10 * 60)  # over-ride the function to return interval of 10 mins
    query1, query2 = my_athena.agg.get_building_average_kws_at(at_days=[1, 2, 3, 4], at_hour=12.3,
                                                               enduses=enduses, get_query_only=True)
    valid_query_string1 = """
    select "res_n250_hrly_v1_timeseries"."building_id",  sum(1) as "sample_count",
    sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "units_count", avg("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" *
    "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * 6.0) as
    "fuel use: electricity: total", avg("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * 6.0) as "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join
    "res_n250_hrly_v1_baseline" on "res_n250_hrly_v1_baseline"."building_id" = "res_n250_hrly_v1_timeseries"."building_id"  where "res_n250_hrly_v1_timeseries"."time" in
    (timestamp '2012-01-01 12:10:00', timestamp '2012-01-02 12:10:00', timestamp '2012-01-03 12:10:00',
    timestamp '2012-01-04 12:10:00') group by 1 order by 1
    """  # noqa: E501

    valid_query_string2 = """
    select "res_n250_hrly_v1_timeseries"."building_id", sum(1) as "sample_count",
    sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "units_count", avg("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" *
    "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * 6.0) as
    "fuel use: electricity: total", avg("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * 6.0) as "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join
    "res_n250_hrly_v1_baseline" on "res_n250_hrly_v1_baseline"."building_id" = "res_n250_hrly_v1_timeseries"."building_id"  where "res_n250_hrly_v1_timeseries"."time" in
    (timestamp '2012-01-01 12:20:00', timestamp '2012-01-02 12:20:00', timestamp '2012-01-03 12:20:00',
    timestamp '2012-01-04 12:20:00') group by 1 order by 1
    """  # noqa: E501

    # verify that correct queries are run
    assert_query_equal(query1, valid_query_string1)
    assert_query_equal(query2, valid_query_string2)

    # verify that the weighted average is done correctly
    fake_lower_df = pd.DataFrame({'building_id': [1, 2], 'fuel use: electricity: total': [10.0, 20.0],
                                  'end use: electricity: cooling': [15.0, 30.0]})
    fake_upper_df = pd.DataFrame({'building_id': [1, 2], 'fuel use: electricity: total': [20.0, 30.0],
                                  'end use: electricity: cooling': [25.0, 35.0]})
    true_weighted_sum = pd.DataFrame({'building_id': [1, 2], 'fuel use: electricity: total': [18.0, 28.0],
                                      'end use: electricity: cooling': [23.0, 34.0]})

    my_athena.submit_batch_query = lambda *args, **kwargs: 0
    my_athena.get_batch_query_result = lambda *args, **kwargs: (fake_lower_df, fake_upper_df)
    res = my_athena.agg.get_building_average_kws_at(at_days=[1, 2, 3, 4], at_hour=12.3,
                                                    enduses=enduses)
    pd.testing.assert_frame_equal(res, true_weighted_sum)

    # Test at_hour as a list of hours that exactly coincide with timestamps. Single query must be returned
    my_athena._get_simulation_info = lambda: (2012, 15 * 60)  # over-ride the function to return interval of 15 mins
    query1, = my_athena.agg.get_building_average_kws_at(at_days=[1, 2, 3, 4], at_hour=[12.25, 12.5, 12.5, 12.75],
                                                        enduses=enduses, get_query_only=True)
    valid_query_string1 = """
    select "res_n250_hrly_v1_timeseries"."building_id",  sum(1) as "sample_count",
    sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "units_count", avg("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" *
    "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * 4.0) as
    "fuel use: electricity: total", avg("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * 4.0) as "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join
    "res_n250_hrly_v1_baseline" on "res_n250_hrly_v1_baseline"."building_id" = "res_n250_hrly_v1_timeseries"."building_id"  where "res_n250_hrly_v1_timeseries"."time" in
    (timestamp '2012-01-01 12:15:00', timestamp '2012-01-02 12:30:00', timestamp '2012-01-03 12:30:00',
    timestamp '2012-01-04 12:45:00') group by 1 order by 1
    """  # noqa: E501

    # verify that correct queries are run
    assert_query_equal(query1, valid_query_string1)

    # Test at_hour as a list of hours which have only a few hours that coincide with timestamps.
    # Two queries must be returned
    my_athena._get_simulation_info = lambda: (2012, 15 * 60)  # over-ride the function to return interval of 15 mins
    query1, query2 = my_athena.agg.get_building_average_kws_at(at_days=[1, 2, 3, 4], at_hour=[12.25, 12.5, 12.625,
                                                                                              12.75],
                                                               enduses=enduses, get_query_only=True)
    valid_lower_query = """
        select "res_n250_hrly_v1_timeseries"."building_id",  sum(1) as "sample_count",
        sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "units_count", avg("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" *
        "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * 4.0) as
        "fuel use: electricity: total", avg("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * 4.0) as "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join
        "res_n250_hrly_v1_baseline" on "res_n250_hrly_v1_baseline"."building_id" = "res_n250_hrly_v1_timeseries"."building_id"  where "res_n250_hrly_v1_timeseries"."time" in
        (timestamp '2012-01-01 12:15:00', timestamp '2012-01-02 12:30:00', timestamp '2012-01-03 12:30:00',
        timestamp '2012-01-04 12:45:00') group by 1 order by 1
        """  # noqa: E501

    valid_upper_query = """
        select "res_n250_hrly_v1_timeseries"."building_id",  sum(1) as "sample_count",
        sum("res_n250_hrly_v1_baseline"."build_existing_model.sample_weight") as "units_count", avg("res_n250_hrly_v1_timeseries"."fuel use: electricity: total" *
        "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * 4.0) as
        "fuel use: electricity: total", avg("res_n250_hrly_v1_timeseries"."end use: electricity: cooling" * "res_n250_hrly_v1_baseline"."build_existing_model.sample_weight" * 4.0) as "end use: electricity: cooling" from "res_n250_hrly_v1_timeseries" join
        "res_n250_hrly_v1_baseline" on "res_n250_hrly_v1_baseline"."building_id" = "res_n250_hrly_v1_timeseries"."building_id"  where "res_n250_hrly_v1_timeseries"."time" in
        (timestamp '2012-01-01 12:15:00', timestamp '2012-01-02 12:30:00', timestamp '2012-01-03 12:45:00',
        timestamp '2012-01-04 12:45:00') group by 1 order by 1
        """  # noqa: E501

    # verify that correct queries are run
    assert_query_equal(query1, valid_lower_query)
    assert_query_equal(query2, valid_upper_query)

    # verify that the weighted average is done correctly
    fake_lower_df = pd.DataFrame({'building_id': [1, 2], 'fuel use: electricity: total': [10.0, 20.0],
                                  'end use: electricity: cooling': [15.0, 30.0]})
    fake_upper_df = pd.DataFrame({'building_id': [1, 2], 'fuel use: electricity: total': [20.0, 30.0],
                                  'end use: electricity: cooling': [25.0, 35.0]})
    true_weighted_sum = pd.DataFrame({'building_id': [1, 2], 'fuel use: electricity: total': [15.0, 25.0],
                                      'end use: electricity: cooling': [20.0, 32.5]})

    my_athena.submit_batch_query = lambda *args, **kwargs: 0
    my_athena.get_batch_query_result = lambda *args, **kwargs: (fake_lower_df, fake_upper_df)
    res = my_athena.agg.get_building_average_kws_at(at_days=[1, 2, 3, 4], at_hour=[12.25, 12.5, 12.625, 12.75],
                                                    enduses=enduses)
    pd.testing.assert_frame_equal(res, true_weighted_sum)
