import pathlib
import numpy as np
import pandas as pd
from buildstock_query.upgrades_analyzer import UpgradesAnalyzer
import pytest


class TestUpgradesAnalyzer:
    @pytest.fixture
    def ua(self):
        folder_path = pathlib.Path(__file__).parent.resolve()
        yaml_path = folder_path / "reference_files" / "res_n250_15min_v19.yml"
        buildstock_path = folder_path / "reference_files" / "res_n250_15min_v19_buildstock.csv"
        ua = UpgradesAnalyzer(yaml_path, str(buildstock_path))
        return ua

    def test_read_cfg(self, ua):
        cfg = ua.get_cfg()
        assert isinstance(cfg, dict)
        assert "upgrades" in cfg
        assert "postprocessing" in cfg

    def test_custom_buildstock(self):
        folder_path = pathlib.Path(__file__).parent.resolve()
        yaml_path = folder_path / "reference_files" / "res_n250_15min_v19.yml"
        buildstock_path = folder_path / "reference_files" / "res_n250_15min_v19_buildstock.csv"
        bdf = pd.read_csv(buildstock_path)
        ua = UpgradesAnalyzer(yaml_path, bdf)
        ua.get_detailed_report(2)

    @pytest.mark.parametrize("test_case", [0, 1])
    def test_get_para_option(self, test_case):
        test_entries = [["Vintage|1980s", ("vintage", "1980s")],
                        ["Windows|Single, Clear, Metal, Exterior Low-E Storm",
                            ("windows", "Single, Clear, Metal, Exterior Low-E Storm")]
                        ]
        test_inp, expected_output = test_entries[test_case]
        assert expected_output == UpgradesAnalyzer._get_para_option(test_inp)

    @pytest.mark.parametrize("test_case", [0, 1])
    def test_get_eq_str(self, test_case):
        test_entries = [["Vintage|1980s", "`vintage`=='1980s'"],
                        ["Windows|Single, Clear, Metal, Exterior Low-E Storm",
                            "`windows`=='Single, Clear, Metal, Exterior Low-E Storm'"]
                        ]
        test_inp, expected_output = test_entries[test_case]
        assert expected_output == UpgradesAnalyzer._get_eq_str(test_inp)

    def test_get_mentioned_parameters(self):

        empty_logics = [{}, [], '', None]
        for logic in empty_logics:
            assert UpgradesAnalyzer.get_mentioned_parameters(logic) == []

        assert UpgradesAnalyzer.get_mentioned_parameters("Vintage|1980s") == ["vintage"]
        assert UpgradesAnalyzer.get_mentioned_parameters(
            "Windows|Single, Clear, Metal, Exterior Low-E Storm") == ["windows"]
        logic = {"and": [{"or": ["Vintage|1980s", "Vintage|1960s"]}, "Location Region|CR09"]}
        assert UpgradesAnalyzer.get_mentioned_parameters(logic) == ['vintage', 'location region']

    def test_reduce_logic(self, ua):
        logic = {"and": [{"or": ["Vintage|1980s", "Vintage|1960s"]}, "Location Region|CR09"]}
        reduced_logic = ua._reduce_logic(logic)
        cond1 = ua.buildstock_df['location region'] == 'CR09'
        cond2 = ua.buildstock_df['vintage'].isin(['1980s', '1960s'])
        assert (cond1 & cond2 == reduced_logic).all()

        logic = {"and": [{"or": ["Vintage|1980s", "Vintage|1960s"]}, "Vintage|1980s"]}
        reduced_logic = ua._reduce_logic(logic)
        cond1 = ua.buildstock_df['vintage'] == '1980s'
        assert (cond1 == reduced_logic).all()

        logic = {"and": [{"or": ["Vintage|1980s", "Vintage|1960s"]}, {"and": ["Vintage|1980s", "Vintage|1960s"]}]}
        reduced_logic = ua._reduce_logic(logic)
        cond1 = [False] * len(ua.buildstock_df)
        assert (cond1 == reduced_logic).all()

        # A list, except for that inside an `or` block is always interpreted as `and` block
        logic = {"not": ["Vintage|1980s", "Vintage|1960s"]}  # Since no buildings have both vintage, should select all
        reduced_logic = ua._reduce_logic(logic)
        cond1 = [True] * len(ua.buildstock_df)
        assert (cond1 == reduced_logic).all()

        logic = {"and": [{"or": ["Vintage|1980s", "Vintage|1960s"]}, {"not": ["Vintage|1980s"]}]}
        reduced_logic = ua._reduce_logic(logic)
        cond1 = ua.buildstock_df['vintage'] == '1960s'
        assert (cond1 == reduced_logic).all()

    def test_normalize_lists(self):
        logic = [["logic1", "logic2"], ["logic3", "logic4"]]
        flatened_logic = UpgradesAnalyzer._normalize_lists(logic)
        expected_logic = {'and': [{'and': ["logic1", "logic2"]}, {'and': ["logic3", "logic4"]}]}
        assert flatened_logic == expected_logic

        logic = {"or": [["logic1", "logic2"], ["logic3", "logic4"], {"and": [["logic5"], ["logic6"]]},
                        ["logic7"]]}
        flatened_logic = UpgradesAnalyzer._normalize_lists(logic)
        expected_logic = {"or": [{'and': ["logic1", "logic2"]},
                                 {'and': ["logic3", "logic4"]}, {"and": ["logic5", "logic6"]}, "logic7"]}
        assert expected_logic == flatened_logic

        logic = {"and": [{"or": ["Vintage|1980s", "Vintage|1960s"]}, {"and": ["Vintage|1980s", "Vintage|1960s"]}]}
        flatened_logic = UpgradesAnalyzer._normalize_lists(logic)
        expected_logic = logic.copy()
        assert flatened_logic == expected_logic

    def test_print_options_combination_report(self, ua, capsys):
        logic_dict = {0: np.array([True, True, True]),
                      1: np.array([True, False, True]),
                      2: np.array([True, True, False])}
        report_text = ua._get_options_combination_report(logic_dict, comb_type='and')
        assert "Option 1 and Option 2: 2 (66.7%)" in report_text
        assert "Option 1 and Option 3: 2 (66.7%)" in report_text
        assert "Option 2 and Option 3: 1 (33.3%)" in report_text
        assert "Option 1 and Option 2 and Option 3: 1 (33.3%)" in report_text

        report_text = ua._get_options_combination_report(logic_dict, comb_type='or')
        assert "Option 1 or Option 2: 3 (100.0%)" in report_text
        assert "Option 1 or Option 3: 3 (100.0%)" in report_text
        assert "Option 2 or Option 3: 3 (100.0%)" in report_text
        assert "Option 1 or Option 2 or Option 3: 3 (100.0%)" in report_text

    def test_get_report(self, ua: UpgradesAnalyzer):
        cfg = ua.get_cfg()
        new_cfg = cfg.copy()
        ua.get_cfg = lambda: new_cfg

        new_cfg['upgrades'] = [cfg['upgrades'][0]]  # keep only one upgrade
        report_df = ua.get_report()
        assert (report_df['upgrade_name'] == 'upgrade1').all()
        assert len(report_df) == 2
        opt1_cond = report_df['option'] == 'Insulation Wall|Wood Stud, Uninsulated, R-5 Sheathing'
        logic_cond1 = ua.buildstock_df['insulation wall'] == 'Wood Stud, Uninsulated'
        logic_cond2 = ua.buildstock_df['vintage'] == '1980s'
        assert report_df[opt1_cond].applicable_to.values[0] == sum(logic_cond1 | logic_cond2)
        assert report_df[report_df['option'] == 'All'].applicable_to.values[0] == sum(logic_cond1 | logic_cond2)

        new_cfg['upgrades'] = [cfg['upgrades'][1]]  # keep only one upgrade
        report_df = ua.get_report()
        assert (report_df['upgrade_name'] == 'upgrade2').all()
        assert len(report_df) == 4
        opt1_cond = report_df['option'] == 'Windows|Single, Clear, Metal, Exterior Low-E Storm'
        opt2_cond = report_df['option'] == 'Vintage|1980s'
        opt3_cond = report_df['option'] == 'Vintage|1970s'
        pkg_logic = ~ua.buildstock_df['vintage'].isin(['1990s', '2000s'])
        logic_cond1_1 = ua.buildstock_df['windows'] == 'Single, Clear, Metal'
        logic_cond1_2 = ua.buildstock_df['windows'] == 'Single, Clear, Metal, Exterior Clear Storm'
        opt1_logic = (logic_cond1_1 | logic_cond1_2) & pkg_logic
        logic_cond2_1 = ua.buildstock_df['vintage'] == '1960s'
        opt2_logic = logic_cond2_1 & pkg_logic
        logic_cond3_1 = ua.buildstock_df['vintage'] == '1980s'
        logic_cond3_2 = ua.buildstock_df['vintage'] == '1960s'
        logic_cond3_3 = ua.buildstock_df['location region'] == 'CR09'
        opt3_logic = (logic_cond3_1 | logic_cond3_2) & logic_cond3_3 & pkg_logic
        assert report_df[opt1_cond].applicable_to.values[0] == sum(opt1_logic)
        assert report_df[opt2_cond].applicable_to.values[0] == sum(opt2_logic)
        assert report_df[opt3_cond].applicable_to.values[0] == sum(opt3_logic)
        assert report_df[report_df['option'] == 'All'].applicable_to.values[0] ==\
            sum(opt1_logic | opt2_logic | opt3_logic)

        new_cfg['upgrades'] = [cfg['upgrades'][4]]  # No apply_logic
        report_df = ua.get_report()
        assert (report_df['upgrade_name'] == 'upgrade5').all()
        assert len(report_df) == 2
        opt1_cond = report_df['option'] == 'Vintage|1980s'
        assert report_df[opt1_cond].applicable_to.values[0] == len(ua.buildstock_df)
        assert report_df[report_df['option'] == 'All'].applicable_to.values[0] == len(ua.buildstock_df)

    def test_print_detailed_report(self, ua: UpgradesAnalyzer, capsys):
        with pytest.raises(ValueError):
            ua.get_detailed_report(0)  # upgrade 0 is invalid. It is 1-indexed

        with pytest.raises(ValueError):
            ua.get_detailed_report(1, 0)  # option 0 is invalid. It is 1-indexed

        _, report_text = ua.get_detailed_report(1)
        assert "Option1:'Insulation Wall|Wood Stud, Uninsulated, R-5 Sheathing'" in report_text
        logic_cond1 = ua.buildstock_df['insulation wall'] == 'Wood Stud, Uninsulated'
        cmp_str = f"Insulation Wall|Wood Stud, Uninsulated => {sum(logic_cond1)}"
        assert cmp_str in report_text
        logic_cond2 = ua.buildstock_df['vintage'] == '1980s'
        cmp_str = f"Vintage|1980s => {sum(logic_cond2)}"
        assert cmp_str in report_text
        assert f"or => {sum(logic_cond1 | logic_cond2)}" in report_text
        assert f"and => {sum(logic_cond1 | logic_cond2)}" in report_text
        assert f"Overall applied to => {sum(logic_cond1 | logic_cond2)}" in report_text

        _, report_text = ua.get_detailed_report(2)
        opt1_text = "Option1:'Windows|Single, Clear, Metal, Exterior Low-E Storm'"
        opt2_text = "Option2:'Vintage|1980s'"
        opt3_text = "Option3:'Vintage|1970s'"
        assert opt1_text in report_text
        assert opt2_text in report_text
        assert opt3_text in report_text

        substr1 = report_text[report_text.index(opt1_text): report_text.index(opt2_text)]
        assert 'Package Apply Logic Report' in substr1
        package_report = substr1[substr1.index('Package Apply Logic Report'):]
        main_report = substr1[:substr1.index('Package Apply Logic Report')]
        logic1 = ua.buildstock_df["windows"] == "Single, Clear, Metal"
        assert f"Windows|Single, Clear, Metal => {sum(logic1)}" in main_report
        logic2 = ua.buildstock_df["windows"] == "Single, Clear, Metal, Exterior Clear Storm"
        assert f"Single, Clear, Metal, Exterior Clear Storm => {sum(logic2)}" in main_report
        assert f"or => {sum(logic1 | logic2)}" in main_report
        assert f"and => {sum(logic1 | logic2)}" in main_report

        logic3 = ua.buildstock_df["vintage"] == "1990s"
        assert f"Vintage|1990s => {sum(logic3)}" in package_report
        logic4 = ua.buildstock_df["vintage"] == "2000s"
        assert f"Vintage|2000s => {sum(logic4)}" in package_report
        assert f"or => {sum(logic3 | logic4)}" in package_report
        assert f"not => {sum(~(logic3 | logic4))}" in package_report

        # TODO: Also add test for combination report output

    def test_get_logic_report(self, ua: UpgradesAnalyzer):
        for logic_cfg in ["Vintage|1980s", ["Vintage|1980s"]]:
            logic_arr = ua.buildstock_df["vintage"] == "1980s"
            logic_arr_out, logic_report = ua._get_logic_report(logic_cfg)
            assert (logic_arr == logic_arr_out).all()
            assert isinstance(logic_report, list)
            assert f'Vintage|1980s => {sum(logic_arr)}' in logic_report[0]

        for logic_cfg in [{"or": ["Vintage|1980s"]}, {"or": "Vintage|1980s"}]:
            logic_arr = ua.buildstock_df["vintage"] == "1980s"
            logic_arr_out, logic_report = ua._get_logic_report(logic_cfg)
            assert (logic_arr == logic_arr_out).all()
            assert isinstance(logic_report, list)
            assert len(logic_report) == 2
            assert f'or => {sum(logic_arr)}' in logic_report[0]
            assert f'Vintage|1980s => {sum(logic_arr)}' in logic_report[1]

        logic_cfg = ["Vintage|1980s", "Vintage|1960s"]
        logic_arr1 = ua.buildstock_df["vintage"] == "1980s"
        logic_arr2 = ua.buildstock_df["vintage"] == "1960s"
        logic_arr_out, logic_report = ua._get_logic_report(logic_cfg)
        assert (logic_arr1 & logic_arr2 == logic_arr_out).all()
        assert isinstance(logic_report, list)
        assert len(logic_report) == 2
        assert f'Vintage|1980s => {sum(logic_arr1)}' in logic_report[0]
        assert f'=> {sum(logic_arr1 & logic_arr2)}' in logic_report[0]  # for overall sum
        assert f'Vintage|1960s => {sum(logic_arr2)}' in logic_report[1]

        logic_cfg = {"and": {"or": ["Vintage|1980s", "Vintage|1960s"]}}
        logic_arr1 = ua.buildstock_df["vintage"] == "1980s"
        logic_arr2 = ua.buildstock_df["vintage"] == "1960s"
        logic_arr_out, logic_report = ua._get_logic_report(logic_cfg)
        assert (logic_arr1 | logic_arr2 == logic_arr_out).all()
        assert isinstance(logic_report, list)
        assert len(logic_report) == 4
        assert f'and => {sum(logic_arr1 | logic_arr2)}' in logic_report[0]
        assert f'or => {sum(logic_arr1 | logic_arr2)}' in logic_report[1]
        assert f'Vintage|1980s => {sum(logic_arr1)}' in logic_report[2]
        assert f'Vintage|1960s => {sum(logic_arr2)}' in logic_report[3]

        logic_cfg = {"not": ["Vintage|1980s"]}
        logic_arr = ua.buildstock_df["vintage"] == "1980s"
        logic_arr_out, logic_report = ua._get_logic_report(logic_cfg)
        assert ((~logic_arr) == logic_arr_out).all()
        assert isinstance(logic_report, list)
        assert len(logic_report) == 2
        assert f'not => {sum(~logic_arr)}' in logic_report[0]
        assert f'Vintage|1980s => {sum(logic_arr)}' in logic_report[1]

        logic_cfg = {"not": ["Vintage|1980s", "Vintage|1960s"]}
        logic_arr1 = ua.buildstock_df["vintage"] == "1980s"
        logic_arr2 = ua.buildstock_df["vintage"] == "1960s"
        logic_arr_out, logic_report = ua._get_logic_report(logic_cfg)
        assert (~(logic_arr1 & logic_arr2) == logic_arr_out).all()
        assert isinstance(logic_report, list)
        assert len(logic_report) == 3
        assert f'not => {sum(~(logic_arr1 & logic_arr2))}' in logic_report[0]
        assert f'Vintage|1980s => {sum(logic_arr1)}' in logic_report[1]
        assert f'Vintage|1960s => {sum(logic_arr2)}' in logic_report[2]

        logic_cfg = {"not": {"or": ["Vintage|1980s", "Vintage|1960s"]}}
        logic_arr1 = ua.buildstock_df["vintage"] == "1980s"
        logic_arr2 = ua.buildstock_df["vintage"] == "1960s"
        logic_arr_or = logic_arr1 | logic_arr2
        logic_arr_out, logic_report = ua._get_logic_report(logic_cfg)
        assert ((~logic_arr_or) == logic_arr_out).all()
        assert isinstance(logic_report, list)
        assert len(logic_report) == 4
        assert f'not => {sum(~logic_arr_or)}' in logic_report[0]
        assert f'or => {sum(logic_arr_or)}' in logic_report[1]
        assert f'Vintage|1980s => {sum(logic_arr1)}' in logic_report[2]
        assert f'Vintage|1960s => {sum(logic_arr2)}' in logic_report[3]

    def test_print_unique_characteristics(self, ua: UpgradesAnalyzer, capsys):
        compare_bldg_list = ua.buildstock_df[ua.buildstock_df['vintage'].isin(['2000s', '1990s'])].index
        other_bldg_list = ua.buildstock_df[~ua.buildstock_df['vintage'].isin(['2000s', '1990s'])].index
        ua.print_unique_characteristic(1, 'no-chng', other_bldg_list, compare_bldg_list)
        printed_text, err = capsys.readouterr()
        assert "Only no-chng buildings have vintage in ['1990s', '2000s']" in printed_text
        assert "Checking 2 column combinations out of ['insulation wall', 'vintage']" in printed_text
        assert "No 2-column unique chracteristics found."

        condition1 = ua.buildstock_df['vintage'].isin(['2000s', '1990s'])
        condition2 = ua.buildstock_df['windows'].isin(['Single, Clear, Metal',
                                                       'Single, Clear, Metal, Exterior Clear Storm'])
        condition3 = ua.buildstock_df['location region'].isin(['CR09'])
        compare_bldg_list = ua.buildstock_df[condition1 & condition2 & condition3].index
        other_bldg_list = ua.buildstock_df[~(condition1 & condition2 & condition3)].index
        ua.print_unique_characteristic(2, 'no-chng', other_bldg_list, compare_bldg_list)
        printed_text, err = capsys.readouterr()
        assert "Checking 2 column combinations out of ['windows', 'vintage', 'location region']" in printed_text
        assert "Checking 3 column combinations out of ['windows', 'vintage', 'location region']" in printed_text
        assert "Only no-chng buildings have ('windows', 'vintage', 'location region') in "\
               "[('Single, Clear, Metal', '1990s', 'CR09')]" in printed_text
