from tests.utils import save_ref_pkl
from buildstock_query.tools.upgrades_visualizer.viz_data import VizData
from buildstock_query.tools.upgrades_visualizer.figure import UpgradesPlot
from buildstock_query.tools.upgrades_visualizer.plot_utils import PlotParams, SavingsTypes, ValueTypes
import pathlib
import itertools as it
import pytest
import buildstock_query.query_core as query_core
from tests.utils import load_tbl_from_pkl, load_cache_from_pkl
from unittest.mock import MagicMock

query_core.sa.Table = load_tbl_from_pkl  # mock the sqlalchemy table loading
query_core.sa.create_engine = MagicMock()  # mock creating engine
query_core.Connection = MagicMock() # type: ignore # NOQA
query_core.boto3 = MagicMock()


class TestViz:

    @pytest.fixture(scope='class')
    def viz_data(self):
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
        mydata.baseline_run._query_cache = load_cache_from_pkl('small_run_baseline_20230810_100')
        mydata.main_run._query_cache = load_cache_from_pkl('small_run_category_1_20230616')
        return mydata

    @pytest.fixture(scope='class')
    def upgrades_plot(self, viz_data) -> UpgradesPlot:
        upgrades_plot = UpgradesPlot(viz_data=viz_data)
        return upgrades_plot

    @pytest.mark.parametrize("resolution, value_type, savings_type, upgrade, group_by",
                             it.product(["annual", "monthly"], ValueTypes, SavingsTypes,
                                        [None, 0, 1],
                                        [[], ["ahs_region"], ["vacancy_status", "ahs_region"],
                                         ["month", "vacancy_status", "ahs_region"]]))
    def test_get_plotting_df_all_upgrades(self, viz_data, resolution, value_type, savings_type, upgrade, group_by):
        all_cols = viz_data.get_all_end_use_cols(resolution)
        params = PlotParams(enduses=all_cols, savings_type=savings_type, value_type=value_type,
                            change_type=None, resolution=resolution, group_by=group_by,
                            upgrade=upgrade,
                            )
        df = viz_data.get_plotting_df_all_upgrades(params)
        assert len(df) > 0

    @pytest.mark.parametrize("resolution, value_type, savings_type, upgrade, num_enduse, group_by",
                             it.product(["annual", "monthly"], ValueTypes, SavingsTypes,
                                        [None, 0, 1], [1, 3],
                                        [[], ["ahs_region"], ["vacancy_status", "ahs_region"],
                                         ["month", "vacancy_status", "ahs_region"]]))
    def test_get_plot(self, upgrades_plot: UpgradesPlot, resolution, value_type, savings_type, upgrade,
                      num_enduse, group_by):
        all_enduses = upgrades_plot.viz_data.get_all_end_use_cols(resolution=resolution)
        enduses = all_enduses[:num_enduse]
        params = PlotParams(enduses=enduses, savings_type=savings_type, value_type=value_type,
                            change_type=None, resolution=resolution, group_by=group_by,
                            upgrade=upgrade,
                            )
        fig, report_df = upgrades_plot.get_plot(params=params)
        assert len(report_df) > 0
        assert len(fig.data) > 0
