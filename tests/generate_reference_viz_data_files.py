from tests.utils import save_ref_pkl
from buildstock_query.tools.upgrades_visualizer.viz_data import VizData
from buildstock_query.tools.upgrades_visualizer.plot_utils import PlotParams, SavingsTypes, ValueTypes
import pathlib
import itertools as it


def save_bsq_obj(bsq_obj, cache_name=None):
    cache_name = bsq_obj.table_name if cache_name is None else cache_name
    save_ref_pkl(f"{cache_name}_query_cache", bsq_obj._query_cache)
    save_ref_pkl(bsq_obj.bs_table.name, bsq_obj.bs_table)
    if bsq_obj.up_table is not None:
        save_ref_pkl(bsq_obj.up_table.name, bsq_obj.up_table)
    if bsq_obj.ts_table is not None:
        save_ref_pkl(bsq_obj.ts_table.name, bsq_obj.ts_table)


def save_viz_data_reference_data():
    folder_path = pathlib.Path(__file__).parent.resolve()
    yaml_path = str(folder_path / "reference_files" / "example_category_1.yml")
    opt_sat_path = str(folder_path / "reference_files" / "options_saturations.csv")
    viz_data = VizData(
        yaml_path=yaml_path,
        opt_sat_path=opt_sat_path,
        workgroup='largeee',
        db_name='largeee_test_runs',
        run=('small_run_baseline_20230810_100', 'small_run_category_1_20230616'),
        buildstock_type='resstock',
    )
    assert viz_data.baseline_run is not None
    all_cols = viz_data.get_all_end_use_cols("annual")
    for resolution, value_type, savings_type in it.product(["annual", "monthly"], ValueTypes, SavingsTypes):
        params = PlotParams(enduses=all_cols, savings_type=savings_type, value_type=value_type,
                            change_type=None, resolution=resolution,
                            upgrade=[0] + viz_data.available_upgrades,
                            )
        print(f"Generating table for {resolution, value_type, savings_type}")
        viz_data.get_plotting_df_all_upgrades(params)
    save_bsq_obj(viz_data.baseline_run, cache_name="small_run_baseline_20230810_100")
    save_bsq_obj(viz_data.main_run, cache_name="small_run_category_1_20230616")


if __name__ == "__main__":
    save_viz_data_reference_data()
