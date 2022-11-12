import pathlib
from buildstock_query.savings_query import BuildStockSavings
from sqlalchemy.exc import NoSuchTableError
from buildstock_query.helpers import save_pickle


def save_ref_pkl(name, obj):
    folder_path = pathlib.Path(__file__).parent.resolve()
    save_pickle(folder_path / f"reference_files/{name}.pkl", obj)
    print(f"Saved {name}.pkl")


def generate_table_and_cache(table_name):
    try:
        mya1 = BuildStockSavings(
            workgroup='eulp',
            db_name='buildstock_testing',
            buildstock_type='resstock',
            table_name=table_name,
            skip_reports=False,
        )
        mya1.get_buildstock_df()
        mya1.get_options_report(trim_missing_bs=False)
        mya1.get_options_report(trim_missing_bs=True)
        mya1.get_success_report(trim_missing_bs=True)
        mya1.get_success_report(trim_missing_bs=False)
        enduses = ['fuel_use_electricity_total_m_btu']
        mya1.savings_shape(1, enduses=enduses)  # annual_savings_full
        mya1.savings_shape(1, enduses=enduses, applied_only=True)  # annual_savings_applied =
        mya1.aggregate_annual(enduses=enduses)  # annual_bs_consumtion
        mya1.aggregate_annual(upgrade_id=1, enduses=enduses)  # annual_up_consumption
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        mya1.savings_shape(1, enduses=ts_enduses, annual_only=False)  # ts_savings_full
        mya1.savings_shape(1, enduses=ts_enduses, applied_only=True, annual_only=False)  # ts_savings_applied

        enduses = ['fuel_use_electricity_total_m_btu']
        group_by = ["geometry_building_type_recs"]
        mya1.savings_shape(1, enduses=enduses, group_by=group_by, sort=True)
        mya1.savings_shape(1, enduses=enduses, group_by=group_by, applied_only=True,
                           sort=True)
        mya1.aggregate_annual(enduses=enduses, group_by=group_by, sort=True)
        mya1.aggregate_annual(upgrade_id=1, enduses=enduses, group_by=group_by, sort=True)
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        mya1.savings_shape(1, enduses=ts_enduses, annual_only=False, group_by=group_by)
        mya1.savings_shape(1, enduses=ts_enduses, applied_only=True, annual_only=False, group_by=group_by)
        group_by = ["geometry_building_type_recs"]
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        mya1.savings_shape(1, enduses=ts_enduses, annual_only=False, sort=True, group_by=group_by)
        mya1.savings_shape(1, enduses=ts_enduses, applied_only=True, annual_only=False, group_by=group_by, sort=True)
        mya1.savings_shape(1, enduses=ts_enduses, annual_only=False, group_by=group_by, sort=True, collapse_ts=True)
        mya1.savings_shape(1, enduses=ts_enduses, applied_only=True, annual_only=False, group_by=group_by, sort=True,
                           collapse_ts=True)
        group_by = ["state", "geometry_building_type_recs"]
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        restrict = [['state', ('CO')]]
        mya1.savings_shape(1, enduses=ts_enduses, applied_only=True, annual_only=False, group_by=group_by, sort=True,
                           restrict=restrict)

        save_ref_pkl(f"{table_name}_query_cache", mya1._query_cache)
        save_ref_pkl(mya1.bs_table.name, mya1.bs_table)
        save_ref_pkl(mya1.up_table.name, mya1.up_table)
        save_ref_pkl(mya1.ts_table.name, mya1.ts_table)
    except NoSuchTableError:
        print(f"{table_name} no longer exists in Athena")


def generate_table(table_name):
    try:
        mya2 = BuildStockSavings(
            workgroup='eulp',
            db_name='buildstock_testing',
            buildstock_type='resstock',
            table_name=table_name,
            skip_reports=False,
        )
        save_ref_pkl(mya2.bs_table.name, mya2.bs_table)
        save_ref_pkl(mya2.bs_table.name, mya2.ts_table)
        try:
            eiaid_table = mya2.get_table('eiaid_weights')
            save_ref_pkl(eiaid_table.name, eiaid_table)
        except NoSuchTableError:
            print("res_n250_hrly_v1 no longer exists in Athena")
    except NoSuchTableError:
        print("res_n250_hrly_v1 no longer exists in Athena")


if __name__ == "__main__":
    generate_table_and_cache("res_n250_15min_v19")
    generate_table("res_n250_hrly_v1")
