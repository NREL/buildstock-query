"""
# Utils
- - - - - - - - -
Different utility functions

:author: Rajendra.Adhikari@nrel.gov
"""
from cmath import log
from functools import reduce
import yaml
import pandas as pd
import numpy as np
import logging
from itertools import combinations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_COMBINATION_REPORT_COUNT = 5  # Don't print combination report; There would be 2^n - n - 1 rows


class UpgradesAnalyzer:
    def __init__(self, yaml_file, buildstock) -> None:
        self.yaml_file = yaml_file
        if isinstance(buildstock, str):
            self.buildstock_df = pd.read_csv(buildstock)
            self.buildstock_df.columns = [c.lower() for c in self.buildstock_df.columns]
        elif isinstance(buildstock, pd.DataFrame):
            self.buildstock_df = buildstock
        self.total_samples = len(self.buildstock_df)
        self.logic_cache = {}

    def read_cfg(self):
        with open(self.yaml_file) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        return cfg

    @staticmethod
    def _get_eq_str(condition):
        try:
            para, option = condition.split('|')
        except ValueError:
            raise ValueError(f"Condition {condition} is invalid")
        return f"(`{para.lower()}`=='{option}')"

    @staticmethod
    def _get_para_option(condition):
        try:
            para, option = condition.split('|')
        except ValueError:
            raise ValueError(f"Condition {condition} is invalid")
        return para.lower(), option

    def reduce_logic(self, logic, parent=None):
        if str(logic) in self.logic_cache:
            return self.logic_cache[str(logic)]

        logic_array = np.ones((1, self.total_samples), dtype=bool)
        if parent not in [None, 'and', 'or', 'not']:
            raise ValueError(f"Logic can only inlcude and, or, not blocks. {parent} found in {logic}.")

        if isinstance(logic, str):
            para, opt = UpgradesAnalyzer._get_para_option(logic)
            logic_array = (self.buildstock_df[para] == opt)
        elif isinstance(logic, list):
            if len(logic) == 1:
                logic_array = self.reduce_logic(logic[0])
            elif parent in ['or']:
                logic_array = reduce(lambda l1, l2: l1 | self.reduce_logic(l2), logic,
                                     np.zeros((1, self.total_samples), dtype=bool))
            else:
                logic_array = reduce(lambda l1, l2: l1 & self.reduce_logic(l2), logic,
                                     np.ones((1, self.total_samples), dtype=bool))
        elif isinstance(logic, dict):
            if len(logic) > 1:
                raise ValueError(f"Dicts cannot have more than one keys. {logic} has.")
            key = list(logic.keys())[0]
            logic_array = self.reduce_logic(logic[key], parent=key)

        if parent == 'not':
            return ~logic_array
        if not (isinstance(logic, str) or (isinstance(logic, list) and len(logic) == 1)):
            # Don't cache small logics - computing them again won't be too bad
            self.logic_cache[str(logic)] = logic_array
        return logic_array

    def get_report(self):
        cfg = self.read_cfg()
        self.logic_cache = {}
        if 'upgrades' not in cfg:
            raise ValueError("The project yaml has no upgrades defined")
        records = []
        for indx, upgrade in enumerate(cfg['upgrades']):
            logger.info(f"Analyzing upgrade {indx + 1}")
            all_applied_bldgs = np.zeros((1, self.total_samples), dtype=bool)
            package_applied_bldgs = np.ones((1, self.total_samples), dtype=bool)
            if "package_apply_logic" in upgrade:
                package_flat_logic = UpgradesAnalyzer.flatten_lists(upgrade['package_apply_logic'])
                package_applied_bldgs = self.reduce_logic(package_flat_logic, parent=None)

            for opt_index, option in enumerate(upgrade['options']):
                if 'apply_logic' in option:
                    flat_logic = UpgradesAnalyzer.flatten_lists(option['apply_logic'])
                    applied_bldgs = self.reduce_logic(flat_logic, parent=None)
                else:
                    applied_bldgs = np.ones((1, self.total_samples), dtype=bool)

                applied_bldgs &= package_applied_bldgs
                count = applied_bldgs.sum()
                all_applied_bldgs |= applied_bldgs
                record = {'upgrade': str(indx+1), 'upgrade_name': upgrade['upgrade_name'],
                          'option_num': opt_index + 1,
                          'option': option['option'], 'applicable_to': count,
                          'applicable_percent': self.to_pct(count),
                          'cost': option.get('cost', 0),
                          'lifetime': option.get('lifetime', float('inf'))}
                records.append(record)

            count = all_applied_bldgs.sum()
            record = {'upgrade': str(indx+1), 'upgrade_name': upgrade['upgrade_name'],
                      'option_num': -1,
                      'option': "All", 'applicable_to': count,
                      'applicable_percent': self.to_pct(count)}
            records.append(record)
        report_df = pd.DataFrame.from_records(records)
        return report_df

    @staticmethod
    def flatten_lists(logic):
        if isinstance(logic, list):
            flat_list = []
            [flat_list.extend(UpgradesAnalyzer.flatten_lists(el)) for el in logic]
            return flat_list
        elif isinstance(logic, dict):
            new_dict = {key: UpgradesAnalyzer.flatten_lists(value) for key, value in logic.items()}
            return [new_dict]
        else:
            return [logic]

    def _print_options_combination_report(self, logic_dict, comb_type='and'):
        n_options = len(logic_dict)
        assert comb_type in ['and', 'or']
        if n_options >= 2:
            header = f"Options '{comb_type}' combination report"
            print("-"*len(header))
            print(header)
        else:
            return
        for combination_size in range(2, n_options + 1):
            print("-"*len(header))
            for group in combinations(list(range(n_options)), combination_size):
                if comb_type == 'and':
                    combined_logic = reduce(lambda c1, c2: c1 & c2, [logic_dict[opt_indx] for opt_indx in group])
                else:
                    combined_logic = reduce(lambda c1, c2: c1 | c2, [logic_dict[opt_indx] for opt_indx in group])
                count = combined_logic.sum()
                text = f" {comb_type} ". join([f"Option {opt_indx + 1}" for opt_indx in group])
                print(f"{text}: {count} ({self.to_pct(count)}%)")
        print("-"*len(header))
        return

    def print_detailed_report(self, upgrade_num, option_num=None):
        cfg = self.read_cfg()
        self.logic_cache = {}
        if upgrade_num == 0 or option_num == 0:
            raise ValueError(f"Upgrades and options are 1-indexed.Got {upgrade_num} {option_num}")

        if option_num is None:
            conds_dict = {}
            n_options = len(cfg['upgrades'][upgrade_num - 1]['options'])
            or_array = np.zeros((1, self.total_samples), dtype=bool)
            and_array = np.ones((1, self.total_samples), dtype=bool)
            for option_indx in range(n_options):
                logic_array = self.print_detailed_report(upgrade_num, option_indx + 1)
                if n_options <= MAX_COMBINATION_REPORT_COUNT:
                    conds_dict[option_indx] = log
                or_array |= logic_array
                and_array &= logic_array
            and_count = and_array.sum()
            or_count = or_array.sum()
            if n_options <= MAX_COMBINATION_REPORT_COUNT:
                self._print_options_combination_report(conds_dict, 'and')
                self._print_options_combination_report(conds_dict, 'or')
            else:
                text = f"Combination report not printed because {n_options} options would require "\
                       f"{2**n_options - n_options - 1} rows."
                print(text)
                print("-"*len(text))
            print(f"All options (and) were applied to: {and_count} ({self.to_pct(and_count)}%)")
            print(f"Any of the options (or) were applied to: {or_count} ({self.to_pct(or_count)}%)")
            return

        try:
            upgrade = cfg['upgrades'][upgrade_num - 1]
            opt = upgrade['options'][option_num - 1]
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"The yaml doesn't have {upgrade_num}/{option_num} upgrade/option")

        ugrade_name = upgrade.get('upgrade_name')
        header = f"Option Apply Report for - Upgrade{upgrade_num}:'{ugrade_name}', Option{option_num}:'{opt['option']}'"
        print("-"*len(header))
        print(header)
        print("-"*len(header))
        if "apply_logic" in opt:
            logic = UpgradesAnalyzer.flatten_lists(opt['apply_logic'])
            logic_array, logic_str = self._get_logic_report(logic)
            footer_len = len(logic_str[-1])
            print("\n".join(logic_str))
            print("-"*footer_len)
            # print(cond)
        else:
            logic_array = np.ones((1, self.total_samples), dtype=bool)

        if "package_apply_logic" in upgrade:
            logic = UpgradesAnalyzer.flatten_lists(upgrade['package_apply_logic'])
            package_logic_array, logic_str = self._get_logic_report(logic)
            footer_len = len(logic_str[-1])
            print("Package Apply Logic Report")
            print("--------------------------")
            print("\n".join(logic_str))
            print("-"*footer_len)
            logic_array = logic_array & package_logic_array

        count = logic_array.sum()
        footer_str = f"Overall applied to => {count} ({self.to_pct(count)}%)."
        #  print('-'*len(footer_str))
        print(footer_str)
        print('-'*len(footer_str))
        return logic_array

    def to_pct(self, count):
        return round(100 * count / self.total_samples, 1)

    def _get_logic_report(self, logic, parent=None):
        logic_array = np.ones((1, self.total_samples), dtype=bool)
        logic_str = ['']
        if parent not in [None, 'and', 'or', 'not']:
            raise ValueError(f"Logic can only inlcude and, or, not blocks. {parent} found in {logic}.")
        if isinstance(logic, str):
            logic_condition = UpgradesAnalyzer._get_eq_str(logic)
            logic_array = self.buildstock_df.eval(logic_condition, engine='python')
            logic_str = [logic]
        elif isinstance(logic, list):
            if len(logic) == 1:
                logic_array, logic_str = self._get_logic_report(logic[0])
            elif parent in ['or']:
                def reducer(l1, l2):
                    ll2 = self._get_logic_report(l2)
                    return l1[0] | ll2[0], l1[1] + ll2[1]
                logic_array, logic_str = reduce(reducer, logic,
                                                (np.zeros((1, self.total_samples), dtype=bool), []))
            else:
                def reducer(l1, l2):
                    ll2 = self._get_logic_report(l2)
                    return l1[0] & ll2[0], l1[1] + ll2[1]
                logic_array, logic_str = reduce(reducer, logic,
                                                (np.ones((1, self.total_samples), dtype=bool), []))
        elif isinstance(logic, dict):
            if len(logic) > 1:
                raise ValueError(f"Dicts cannot have more than one keys. {logic} has.")
            key = list(logic.keys())[0]
            sub_logic = self._get_logic_report(logic[key], parent=key)
            # print(sub_logic, logic, key)
            sub_logic_str = sub_logic[1]
            logic_str = [key] + [f"  {ls}" for ls in sub_logic_str]
            logic_array = sub_logic[0]

        if parent == 'not':
            logic_array = ~logic_array

        count = logic_array.sum()
        if parent is None and (not isinstance(logic, list) or len(logic) > 1):
            logic_str[0] = logic_str[0] + " => " + f"{count} ({self.to_pct(count)}%)"

        return logic_array, logic_str
