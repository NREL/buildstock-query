from functools import reduce
import yaml
import pandas as pd
import numpy as np
import logging
from itertools import combinations
from typing import Union
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
import os
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_COMBINATION_REPORT_COUNT = 5  # Don't print combination report; There would be 2^n - n - 1 rows


class UpgradesAnalyzer:
    """
    Analyze the apply logic for various upgrades in the project yaml file.
    """

    def __init__(self, yaml_file: str, buildstock: Union[str, pd.DataFrame]) -> None:
        """
        Initialize the analyzer instance.
        Args:
            yaml_file (str): The path to the yaml file.
            buildstock (Union[str, pd.DataFrame]): Either the buildstock dataframe, or path to the csv
        """
        self.yaml_file = yaml_file
        if isinstance(buildstock, str):
            self.buildstock_df_original = pd.read_csv(buildstock, dtype=str)
            self.buildstock_df = self.buildstock_df_original.copy()
            self.buildstock_df.columns = [c.lower() for c in self.buildstock_df.columns]
            self.buildstock_df.rename(columns={"building": "building_id"}, inplace=True)
            self.buildstock_df.set_index("building_id", inplace=True)
        elif isinstance(buildstock, pd.DataFrame):
            self.buildstock_df_original = buildstock.copy()
            self.buildstock_df = buildstock.reset_index().rename(columns=str.lower)
            self.buildstock_df.rename(columns={"building": "building_id"}, inplace=True)
            if "building_id" in self.buildstock_df.columns:
                self.buildstock_df.set_index("building_id", inplace=True)
            self.buildstock_df = self.buildstock_df.astype(str)
        self.total_samples = len(self.buildstock_df)
        self._logic_cache: dict = {}

    def get_cfg(self) -> dict:
        """Get the buildstock configuration file as a dictionary object.

        Returns:
            dict: The buildstock configuration file.
        """
        with open(self.yaml_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config

    @staticmethod
    def _get_eq_str(condition):
        para, option = UpgradesAnalyzer._get_para_option(condition)
        return f"`{para.lower()}`=='{option}'"

    @staticmethod
    def _get_para_option(condition):
        try:
            para, option = condition.split("|")
        except ValueError as e:
            raise ValueError(f"Condition {condition} is invalid") from e
        return para.lower(), option

    @staticmethod
    def get_mentioned_parameters(logic: Union[list, dict, str]) -> list:
        """
        Returns the list of all parameters referenced in a logic block. Useful for debugging

        Args:
            logic ( Union[list, dict, str]): The apply logic

        Raises:
            ValueError: If the input logic is invalid

        Returns:
            List: The list of parameters
        """
        if not logic:
            return []

        if isinstance(logic, str):
            return [UpgradesAnalyzer._get_para_option(logic)[0]]
        elif isinstance(logic, list):
            all_params = []
            for el in logic:
                all_params.extend(UpgradesAnalyzer.get_mentioned_parameters(el))
            return list(dict.fromkeys(all_params))  # remove duplicates while maintainig order
        elif isinstance(logic, dict):
            return UpgradesAnalyzer.get_mentioned_parameters(list(logic.values())[0])
        else:
            raise ValueError("Invalid logic type")

    def print_unique_characteristic(self, upgrade_num: int, name: str, base_bldg_list: list, compare_bldg_list: list):
        """Finds and prints what's unique among a list of buildings compared to baseline buildings.
           Useful for debugging why a certain set of buildings' energy consumption went up for an upgrade, for example.
        Args:
            upgrade_num (int): The upgrade for which the analysis is being done.
            name (str): Some name to identify the building set (only used for printing)
            base_bldg_list (list): The set of 'normal' buildings id to compare against.
            compare_bldg_list (list): The set of buildings whose unique characteristics is to be printed.
        """
        cfg = self.get_cfg()
        if upgrade_num == 0:
            raise ValueError(f"Upgrades are 1-indexed. Got {upgrade_num}")

        try:
            upgrade_cfg = cfg["upgrades"][upgrade_num - 1]
        except KeyError as e:
            raise ValueError(f"Invalid upgrade {upgrade_num}. Upgrades are 1-indexed, FYI.") from e

        parameter_list = []
        for option_cfg in upgrade_cfg["options"]:
            parameter_list.append(UpgradesAnalyzer._get_para_option(option_cfg["option"])[0])
            parameter_list.extend(UpgradesAnalyzer.get_mentioned_parameters(option_cfg.get("apply_logic")))
        res_df = self.buildstock_df
        # remove duplicates (dict.fromkeys) and remove parameters not existing in buildstock_df
        parameter_list = [param for param in dict.fromkeys(parameter_list) if param in res_df.columns]
        compare_df = res_df.loc[compare_bldg_list]
        base_df = res_df.loc[base_bldg_list]
        print(f"Comparing {len(compare_df)} buildings with {len(base_df)} other buildings.")
        unique_vals_dict: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
        for col in res_df.columns:
            no_change_set = set(compare_df[col].fillna("").unique())
            other_set = set(base_df[col].fillna("").unique())
            if only_in_no_change := no_change_set - other_set:
                print(f"Only {name} buildings have {col} in {sorted(only_in_no_change)}")
                unique_vals_dict[(col,)] = {(entry,) for entry in only_in_no_change}

        if not unique_vals_dict:
            print("No 1-column unique chracteristics found.")

        for combi_size in range(2, min(len(parameter_list) + 1, 5)):
            print(f"Checking {combi_size} column combinations out of {parameter_list}")
            found_uniq_chars = 0
            for cols in combinations(parameter_list, combi_size):
                compare_tups = compare_df[list(cols)].fillna("").drop_duplicates().itertuples(index=False, name=None)
                other_tups = base_df[list(cols)].fillna("").drop_duplicates().itertuples(index=False, name=None)
                only_in_compare = set(compare_tups) - set(other_tups)

                # remove cases arisen out of uniqueness found earlier with smaller susbset of cols
                for sub_combi_size in range(1, len(cols)):
                    for sub_cols in combinations(cols, sub_combi_size):
                        if sub_cols in unique_vals_dict:
                            new_set = set()
                            for val in only_in_compare:
                                relevant_val = tuple(val[cols.index(sub_col)] for sub_col in sub_cols)
                                if relevant_val not in unique_vals_dict[sub_cols]:
                                    new_set.add(val)
                            only_in_compare = new_set

                if only_in_compare:
                    print(f"Only {name} buildings have {cols} in {sorted(only_in_compare)} \n")
                    found_uniq_chars += 1
                    unique_vals_dict[cols] = only_in_compare

            if not found_uniq_chars:
                print(f"No {combi_size}-column unique chracteristics found.")

    def _reduce_logic(self, logic, parent=None):
        cache_key = str(logic) if parent is None else parent + "[" + str(logic) + "]"
        if cache_key in self._logic_cache:
            return self._logic_cache[cache_key]

        logic_array = np.ones((1, self.total_samples), dtype=bool)
        if parent not in [None, "and", "or", "not"]:
            raise ValueError(f"Logic can only inlcude and, or, not blocks. {parent} found in {logic}.")

        if isinstance(logic, str):
            para, opt = UpgradesAnalyzer._get_para_option(logic)
            logic_array = self.buildstock_df[para] == opt
        elif isinstance(logic, list):
            if len(logic) == 1:
                logic_array = self._reduce_logic(logic[0]).copy()
            elif parent in ["or"]:
                logic_array = reduce(
                    lambda l1, l2: l1 | self._reduce_logic(l2),
                    logic,
                    np.zeros((1, self.total_samples), dtype=bool),
                )
            else:
                logic_array = reduce(
                    lambda l1, l2: l1 & self._reduce_logic(l2),
                    logic,
                    np.ones((1, self.total_samples), dtype=bool),
                )
        elif isinstance(logic, dict):
            if len(logic) > 1:
                raise ValueError(f"Dicts cannot have more than one keys. {logic} has.")
            key = list(logic.keys())[0]
            logic_array = self._reduce_logic(logic[key], parent=key).copy()

        if parent == "not":
            return ~logic_array
        if not (isinstance(logic, str) or (isinstance(logic, list) and len(logic) == 1)):
            # Don't cache small logics - computing them again won't be too bad
            self._logic_cache[cache_key] = logic_array.copy()
        return logic_array

    def get_report(self, upgrade_num: Optional[int] = None) -> pd.DataFrame:
        """Analyses which how many buildings various options in all the upgrades is going to apply to and returns
        a report in DataFrame format.
        Args:
            upgrade_num: Numeric index of upgrade (1-indexed). If None, all upgrades are assessed

        Returns:
            pd.DataFrame: The upgrade and options report.

        """

        def _get_records(indx, upgrade):
            records = []
            logger.info(f"Analyzing upgrade {indx + 1}")
            all_applied_bldgs = np.zeros((1, self.total_samples), dtype=bool)
            package_applied_bldgs = np.ones((1, self.total_samples), dtype=bool)
            if "package_apply_logic" in upgrade:
                package_flat_logic = UpgradesAnalyzer._normalize_lists(upgrade["package_apply_logic"])
                package_applied_bldgs = self._reduce_logic(package_flat_logic, parent=None)

            for opt_index, option in enumerate(upgrade["options"]):
                applied_bldgs = np.ones((1, self.total_samples), dtype=bool)
                if "apply_logic" in option:
                    flat_logic = UpgradesAnalyzer._normalize_lists(option["apply_logic"])
                    applied_bldgs &= self._reduce_logic(flat_logic, parent=None)
                else:
                    applied_bldgs = np.ones((1, self.total_samples), dtype=bool)

                applied_bldgs &= package_applied_bldgs
                count = applied_bldgs.sum()
                all_applied_bldgs |= applied_bldgs
                record = {
                    "upgrade": indx + 1,
                    "upgrade_name": upgrade["upgrade_name"],
                    "option_num": opt_index + 1,
                    "option": option["option"],
                    "applicable_to": count,
                    "applicable_percent": self._to_pct(count),
                    "applicable_buildings": set(self.buildstock_df.loc[applied_bldgs[0]].index),
                }
                records.append(record)

            count = all_applied_bldgs.sum()
            record = {
                "upgrade": indx + 1,
                "upgrade_name": upgrade["upgrade_name"],
                "option_num": -1,
                "option": "All",
                "applicable_to": count,
                "applicable_buildings": set(self.buildstock_df.loc[all_applied_bldgs[0]].index),
                "applicable_percent": self._to_pct(count),
            }
            records.append(record)
            return records

        cfg = self.get_cfg()
        self._logic_cache = {}
        if "upgrades" not in cfg:
            raise ValueError("The project yaml has no upgrades defined")

        max_upg = len(cfg["upgrades"]) + 1
        if upgrade_num is not None:
            if upgrade_num <= 0 or upgrade_num > max_upg:
                raise ValueError(f"Invalid upgrade {upgrade_num}. Valid upgrade_num = {list(range(1, max_upg))}.")

        records = []
        for indx, upgrade in enumerate(cfg["upgrades"]):
            if upgrade_num is None or upgrade_num == indx + 1:
                records += _get_records(indx, upgrade)
            else:
                continue

        report_df = pd.DataFrame.from_records(records)
        return report_df

    def get_upgraded_buildstock(self, upgrade_num):
        report_df = self.get_report(upgrade_num)
        upgrade_name = report_df["upgrade_name"].unique()[0]
        logger.info(f" * Upgraded buildstock for upgrade {upgrade_num} : {upgrade_name}")

        df = self.buildstock_df_original.copy()
        for idx, row in report_df.iterrows():
            if row["option"] == "All":
                continue
            dimension, upgrade_option = row["option"].split("|")
            apply_logic = df["Building"].isin(row["applicable_buildings"])
            # apply upgrade
            df[dimension] = np.where(apply_logic, upgrade_option, df[dimension])

        # report
        cond = report_df["option"] == "All"
        n_total = len(self.buildstock_df_original)
        n_applied = report_df.loc[cond, "applicable_to"].iloc[0]
        n_applied_pct = report_df.loc[cond, "applicable_percent"].iloc[0]
        logger.info(
            f"   Upgrade package has {len(report_df)-1} options and "
            f"was applied to {n_applied} / {n_total} dwelling units ( {n_applied_pct} % )"
        )

        # QC
        n_diff = len(self.buildstock_df_original.compare(df)) - n_applied
        if n_diff > 0:
            raise ValueError(
                f"Relative to baseline buildstock, upgraded buildstock has {n_diff} more rows "
                "of difference than reported."
            )
        elif n_diff < 0:
            logger.warning(
                f"Relative to baseline buildstock, upgraded buildstock has {-1*n_diff} fewer rows "
                "of difference than reported. This is okay, but indicates that some parameters are "
                "being upgraded to the same incumbent option (e.g., LEDs to LEDs). Check that this is intentional."
            )
        else:
            logger.info("No cases of parameter upgraded with incumbent option detected.")

        return df

    @staticmethod
    def _normalize_lists(logic, parent=None):
        """Any list that is not in a or block is considered to be in an and block.
           This block will normalize this pattern by adding "and" wherever required.
        Args:
            logic (_type_): Logic structure (dict, list etc)
            parent (_type_, optional): The parent of the current logic block. If it is a list, and there is no parent,
            the list will be wrapped in a and block.

        Returns:
            _type_: _description_
        """
        if isinstance(logic, list):
            # If it is a single element list, just unwrap and return
            if len(logic) == 1:
                return UpgradesAnalyzer._normalize_lists(logic[0])
            new_logic = [UpgradesAnalyzer._normalize_lists(el) for el in logic]
            return {"and": new_logic} if parent is None else new_logic
        elif isinstance(logic, dict):
            new_dict = {key: UpgradesAnalyzer._normalize_lists(value, parent=key) for key, value in logic.items()}
            return new_dict
        else:
            return logic

    def _get_options_combination_report(self, logic_dict, comb_type="and"):
        report_str = ""
        n_options = len(logic_dict)
        assert comb_type in ["and", "or"]
        if n_options < 2:
            return ""
        header = f"Options '{comb_type}' combination report"
        report_str += "-" * len(header) + "\n"
        report_str += header + "\n"
        for combination_size in range(2, n_options + 1):
            report_str += "-" * len(header) + "\n"
            for group in combinations(list(range(n_options)), combination_size):
                if comb_type == "and":
                    combined_logic = reduce(
                        lambda c1, c2: c1 & c2,
                        [logic_dict[opt_indx] for opt_indx in group],
                    )
                else:
                    combined_logic = reduce(
                        lambda c1, c2: c1 | c2,
                        [logic_dict[opt_indx] for opt_indx in group],
                    )
                count = combined_logic.sum()
                text = f" {comb_type} ".join([f"Option {opt_indx + 1}" for opt_indx in group])
                report_str += f"{text}: {count} ({self._to_pct(count, len(combined_logic))}%)" + "\n"
        report_str += "-" * len(header) + "\n"
        return report_str

    def get_detailed_report(self, upgrade_num: int, option_num: Optional[int] = None) -> tuple[np.ndarray, str]:
        """Prints detailed report for a particular upgrade (and optionally, an option)
        Args:
            upgrade_num (int): The 1-indexed upgrade for which to print the report.
            option_num (int, optional): The 1-indexed option number for which to print report. Defaults to None, which
                                        will print report for all options.
        Returns:
            (np.ndarray, str): Returns a logic array of buildings to which the any of the option applied and report str.
        """
        cfg = self.get_cfg()
        if upgrade_num <= 0 or upgrade_num > len(cfg["upgrades"]) + 1:
            raise ValueError(f"Invalid upgrade {upgrade_num}. Upgrade num is 1-indexed.")

        if option_num is None:
            return self._get_detailed_report_all(upgrade_num)

        self._logic_cache = {}
        if upgrade_num == 0 or option_num == 0:
            raise ValueError(f"Upgrades and options are 1-indexed.Got {upgrade_num} {option_num}")
        report_str = ""
        try:
            upgrade = cfg["upgrades"][upgrade_num - 1]
            opt = upgrade["options"][option_num - 1]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"The yaml doesn't have {upgrade_num}/{option_num} upgrade/option") from e

        ugrade_name = upgrade.get("upgrade_name")
        header = f"Option Apply Report for - Upgrade{upgrade_num}:'{ugrade_name}', Option{option_num}:'{opt['option']}'"
        report_str += "-" * len(header) + "\n"
        report_str += header + "\n"
        report_str += "-" * len(header) + "\n"
        if "apply_logic" in opt:
            logic = UpgradesAnalyzer._normalize_lists(opt["apply_logic"])
            logic_array, logic_str = self._get_logic_report(logic)
            footer_len = len(logic_str[-1])
            report_str += "\n".join(logic_str) + "\n"
            report_str += "-" * footer_len + "\n"
        else:
            logic_array = np.ones((1, self.total_samples), dtype=bool)

        if "package_apply_logic" in upgrade:
            logic = UpgradesAnalyzer._normalize_lists(upgrade["package_apply_logic"])
            package_logic_array, logic_str = self._get_logic_report(logic)
            footer_len = len(logic_str[-1])
            report_str += "Package Apply Logic Report" + "\n"
            report_str += "--------------------------" + "\n"
            report_str += "\n".join(logic_str) + "\n"
            report_str += "-" * footer_len + "\n"
            logic_array = logic_array & package_logic_array

        count = logic_array.sum()
        footer_str = f"Overall applied to => {count} ({self._to_pct(count)}%)."
        report_str += footer_str + "\n"
        report_str += "-" * len(footer_str) + "\n"
        return logic_array, report_str

    def _get_detailed_report_all(self, upgrade_num):
        conds_dict = {}
        cfg = self.get_cfg()
        report_str = ""
        n_options = len(cfg["upgrades"][upgrade_num - 1]["options"])
        or_array = np.zeros((1, self.total_samples), dtype=bool)
        and_array = np.ones((1, self.total_samples), dtype=bool)
        for option_indx in range(n_options):
            logic_array, sub_report_str = self.get_detailed_report(upgrade_num, option_indx + 1)
            report_str += sub_report_str + "\n"
            if n_options <= MAX_COMBINATION_REPORT_COUNT:
                conds_dict[option_indx] = logic_array
            or_array |= logic_array
            and_array &= logic_array
        and_count = and_array.sum()
        or_count = or_array.sum()
        if n_options <= MAX_COMBINATION_REPORT_COUNT:
            report_str += self._get_options_combination_report(conds_dict, "and") + "\n"
            report_str += self._get_options_combination_report(conds_dict, "or") + "\n"
        else:
            text = (
                f"Combination report not printed because {n_options} options would require "
                f"{2**n_options - n_options - 1} rows."
            )
            report_str += text + "\n"
            report_str += "-" * len(text) + "\n"
        report_str += f"All of the options (and-ing) were applied to: {and_count} ({self._to_pct(and_count)}%)" + "\n"
        report_str += f"Any of the options (or-ing) were applied to: {or_count} ({self._to_pct(or_count)}%)" + "\n"
        return or_array, report_str

    def _to_pct(self, count, total=None):
        total = total or self.total_samples
        return round(100 * count / total, 1)

    def _get_logic_report(self, logic, parent=None):
        logic_array = np.ones((1, self.total_samples), dtype=bool)
        logic_str = [""]
        if parent not in [None, "and", "or", "not"]:
            raise ValueError(f"Logic can only inlcude and, or, not blocks. {parent} found in {logic}.")
        if isinstance(logic, str):
            logic_condition = UpgradesAnalyzer._get_eq_str(logic)
            logic_array = self.buildstock_df.eval(logic_condition, engine="python")
            count = logic_array.sum()
            logic_str = [logic + " => " + f"{count} ({self._to_pct(count)}%)"]
        elif isinstance(logic, list):
            if len(logic) == 1:
                logic_array, logic_str = self._get_logic_report(logic[0])
            elif parent in ["or"]:

                def reducer(l1, l2):
                    ll2 = self._get_logic_report(l2)
                    return l1[0] | ll2[0], l1[1] + ll2[1]

                logic_array, logic_str = reduce(reducer, logic, (np.zeros((1, self.total_samples), dtype=bool), []))
            else:

                def reducer(l1, l2):
                    ll2 = self._get_logic_report(l2)
                    return l1[0] & ll2[0], l1[1] + ll2[1]

                logic_array, logic_str = reduce(reducer, logic, (np.ones((1, self.total_samples), dtype=bool), []))
        elif isinstance(logic, dict):
            if len(logic) > 1:
                raise ValueError(f"Dicts cannot have more than one keys. {logic} has.")
            key = list(logic.keys())[0]
            sub_logic = self._get_logic_report(logic[key], parent=key)
            sub_logic_str = sub_logic[1]
            logic_array = sub_logic[0]
            if key == "not":
                logic_array = ~logic_array
            count = logic_array.sum()
            header_str = key + " => " + f"{count} ({self._to_pct(count)}%)"
            logic_str = [header_str] + [f"  {ls}" for ls in sub_logic_str]

        count = logic_array.sum()
        if parent is None and isinstance(logic, list) and len(logic) > 1:
            logic_str[0] = logic_str[0] + " => " + f"{count} ({self._to_pct(count)}%)"

        return logic_array, logic_str

    def save_detailed_report_all(self, file_path: str):
        """Save detailed text based upgrade report.

        Args:
            file_path (str): Output file.
        """
        cfg = self.get_cfg()
        all_report = ""
        for upgrade in range(1, len(cfg["upgrades"]) + 1):
            logger.info(f"Getting report for upgrade {upgrade}")
            _, report = self.get_detailed_report(upgrade)
            all_report += report + "\n"
        with open(file_path, "w") as file:
            file.write(all_report)


def main():
    yaml_file = inquirer.filepath(
        message="Project configuration file (EUSS-project-file.yml):",
        validate=PathValidator(),
        filter=lambda x: x or "EUSS-project-file.yml",
    ).execute()
    buildstock_file = inquirer.filepath(
        message="Project sample file (buildstock.csv):",
        validate=PathValidator(),
        filter=lambda x: x or "buildstock.csv",
    ).execute()
    output_prefix = inquirer.text(
        message="output file name prefix:",
        filter=lambda x: "" if x is None else f"{x}_",
    ).execute()
    ua = UpgradesAnalyzer(yaml_file, buildstock_file)
    report_df = ua.get_report()
    csv_name = f"{output_prefix}options_report.csv"
    txt_name = f"{output_prefix}detailed_report.txt"
    report_df.drop(columns=["applicable_buildings"]).to_csv(csv_name)
    ua.save_detailed_report_all(txt_name)
    print(f"Saved  {csv_name} and {txt_name} inside {os.getcwd()}")


if __name__ == "__main__":
    main()
