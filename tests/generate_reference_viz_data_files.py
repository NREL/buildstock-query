from tests.utils import save_ref_pkl
from buildstock_query.tools.upgrades_visualizer.viz_data import VizData
import pathlib


def save_bsq_obj(bsq_obj):
    save_ref_pkl(f"{bsq_obj.table_name}_query_cache", bsq_obj._query_cache)
    save_ref_pkl(bsq_obj.bs_table.name, bsq_obj.bs_table)
    if bsq_obj.up_table is not None:
        save_ref_pkl(bsq_obj.up_table.name, bsq_obj.up_table)
    if bsq_obj.ts_table is not None:
        save_ref_pkl(bsq_obj.ts_table.name, bsq_obj.ts_table)


def generate_table_and_cache(table_name: str):
    folder_path = pathlib.Path(__file__).parent.resolve()
    yaml_path = str(folder_path / "reference_files" / "example_category_1.yml")
    opt_sat_path = str(folder_path / "reference_files" / "options_saturations.csv")
    mydata = VizData(
        yaml_path=yaml_path,
        opt_sat_path=opt_sat_path,
        workgroup='largeee',
        db_name='largeee_test_runs',
        run=('small_run_baseline_20230810_100', 'small_run_category_1_20230616'),
        buildstock_type='resstock',
    )
    assert mydata.baseline_run is not None
    save_bsq_obj(mydata.baseline_run)
    save_bsq_obj(mydata.main_run)


if __name__ == "__main__":
    generate_table_and_cache("res_n250_15min_v19")
