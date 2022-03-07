"""
# Utils
- - - - - - - - -
Different utility functions

:author: Rajendra.Adhikari@nrel.gov
"""
from functools import reduce
import yaml
import pandas as pd
import numpy as np
import logging
from itertools import combinations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UpgradesAnalyzer:
    def __init__(self, yaml_file, buildstock) -> None:
        self.yaml_file = yaml_file
        if isinstance(buildstock, str):
            self.buildstock_df = pd.read_csv(buildstock)
            self.buildstock_df.columns = [c.lower() for c in self.buildstock_df.columns]
        elif isinstance(buildstock, pd.DataFrame):
            self.buildstock_df = buildstock
        self.total_samples = len(self.buildstock_df)

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
    def reduce_logic(logic, parent=None):
        return_str = None
        if parent not in [None, 'and', 'or', 'not']:
            raise ValueError(f"Logic can only inlcude and, or, not blocks. {parent} found in {logic}.")

        if isinstance(logic, str):
            logic_str = UpgradesAnalyzer._get_eq_str(logic)
            return_str = logic_str
        elif isinstance(logic, list):
            logics = [UpgradesAnalyzer.reduce_logic(entry) for entry in logic]
            if parent in ['or']:
                grouped_logic = reduce(lambda l1, l2: l1 + ' | ' + l2, logics)
            else:
                grouped_logic = reduce(lambda l1, l2: l1 + ' & ' + l2, logics)
            return_str = f"({grouped_logic})"
        elif isinstance(logic, dict):
            if len(logic) > 1:
                raise ValueError(f"Dicts cannot have more than one keys. {logic} has.")
            logics = [UpgradesAnalyzer.reduce_logic(value, parent=key) for key, value in logic.items()]
            grouped_logic = reduce(lambda l1, l2: l1 + ' & ' + l2, logics)
            return_str = f"({grouped_logic})"

        if parent == 'not':
            return f"~{return_str}"
        return return_str

    def get_report(self):
        cfg = self.read_cfg()
        if 'upgrades' not in cfg:
            raise ValueError("The project yaml has no upgrades defined")
        records = []
        for indx, upgrade in enumerate(cfg['upgrades']):
            logger.info(f"Analyzing upgrade {indx + 1}")
            all_applied_bldgs = np.zeros((1, self.total_samples), dtype=bool)
            package_applied_bldgs = np.ones((1, self.total_samples), dtype=bool)
            if "package_apply_logic" in upgrade:
                package_flat_logic = UpgradesAnalyzer.flatten_lists(upgrade['package_apply_logic'])
                package_logic_str = UpgradesAnalyzer.reduce_logic(package_flat_logic, parent=None)
                package_applied_bldgs = self.buildstock_df.eval(package_logic_str, engine='python')

            for opt_index, option in enumerate(upgrade['options']):
                if 'apply_logic' in option:
                    flat_logic = UpgradesAnalyzer.flatten_lists(option['apply_logic'])
                    logic_str = UpgradesAnalyzer.reduce_logic(flat_logic, parent=None)
                    applied_bldgs = self.buildstock_df.eval(logic_str, engine='python')
                    applied_bldgs &= package_applied_bldgs
                    count = applied_bldgs.sum()
                    all_applied_bldgs |= applied_bldgs
                else:
                    count = self.total_samples
                    all_applied_bldgs = np.ones((1, self.total_samples), dtype=bool)
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

    def _print_options_combination_report(self, conds_dict, comb_type='and'):
        n_options = len(conds_dict)
        assert comb_type in ['and', 'or']
        comb_sym = '&' if comb_type == 'and' else '|'
        if n_options >= 2:
            header = f"Options '{comb_type}' combination report"
            print("-"*len(header))
            print(header)
        else:
            return
        for combination_size in range(2, n_options + 1):
            print("-"*len(header))
            for group in combinations(list(range(n_options)), combination_size):
                candidates = [opt_indx for opt_indx in group if conds_dict.get(opt_indx)]
                if comb_type == 'or' and len(candidates) < len(group):
                    count = self.total_samples
                else:
                    combined_logic = f' {comb_sym} '.join([conds_dict.get(opt_indx) for opt_indx in candidates])
                    count = self.buildstock_df.eval(combined_logic, engine='python').sum() if combined_logic else self.total_samples
                text = f" {comb_type} ". join([f"Option {opt_indx + 1}" for opt_indx in group])
                print(f"{text}: {count} ({self.to_pct(count)}%)")
        print("-"*len(header))
        return

    def print_detailed_report(self, upgrade_num, option_num=None):
        cfg = self.read_cfg()
        if upgrade_num == 0 or option_num == 0:
            raise ValueError(f"Upgrades and options are 1-indexed.Got {upgrade_num} {option_num}")

        if option_num is None:
            conds_dict = {}
            n_options = len(cfg['upgrades'][upgrade_num - 1]['options'])
            for option_indx in range(n_options):
                conds_dict[option_indx] = self.print_detailed_report(upgrade_num, option_indx + 1)
            self._print_options_combination_report(conds_dict, 'and')
            self._print_options_combination_report(conds_dict, 'or')
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
            cond, logic_str = self._get_logic_report(logic)
            footer_len = len(logic_str[-1])
            print("\n".join(logic_str))
            print("-"*footer_len)
        else:
            cond = None
        if "package_apply_logic" in upgrade:
            logic = UpgradesAnalyzer.flatten_lists(upgrade['package_apply_logic'])
            package_cond, logic_str = self._get_logic_report(logic)
            footer_len = len(logic_str[-1])
            print("Package Apply Logic Report")
            print("--------------------------")
            print("\n".join(logic_str))
            print("-"*footer_len)
            cond = cond + ' & ' + package_cond if cond else package_cond

        count = self.buildstock_df.eval(cond, engine='python').sum() if cond else self.total_samples
        footer_str = f"Overall applied to => {count} ({self.to_pct(count)}%)."
        #  print('-'*len(footer_str))
        print(footer_str)
        print('-'*len(footer_str))
        return cond

    def to_pct(self, count):
        return round(100 * count / self.total_samples, 1)

    def _get_logic_report(self, logic, parent=None):
        logic_condition = None
        if parent not in [None, 'and', 'or', 'not']:
            raise ValueError(f"Logic can only inlcude and, or, not blocks. {parent} found in {logic}.")
        if isinstance(logic, str):
            logic_condition = UpgradesAnalyzer._get_eq_str(logic)
            logic_str = [logic]
        elif isinstance(logic, list):
            sub_logics = [self._get_logic_report(entry) for entry in logic]
            sub_logic_conditions = [sl[0] for sl in sub_logics]
            logic_str = []
            [logic_str.extend(sl[1]) for sl in sub_logics]
            if parent in ['or']:
                grouped_logic = reduce(lambda l1, l2: l1 + ' | ' + l2, sub_logic_conditions)
            else:
                grouped_logic = reduce(lambda l1, l2: l1 + ' & ' + l2, sub_logic_conditions)
            logic_condition = f"({grouped_logic})"
        elif isinstance(logic, dict):
            if len(logic) > 1:
                raise ValueError(f"Dicts cannot have more than one keys. {logic} has.")
            key = list(logic.keys())[0]
            sub_logic = self._get_logic_report(logic[key], parent=key)
            sub_logic_conditions = sub_logic[0]
            sub_logic_str = sub_logic[1]
            logic_str = [key] + [f"  {ls}" for ls in sub_logic_str]
            logic_condition = f"({sub_logic_conditions})"

        if parent == 'not':
            logic_condition = f"~{logic_condition}"

        count = self.buildstock_df.eval(logic_condition, engine='python').sum()
        if parent is None and (not isinstance(logic, list) or len(logic) > 1):
            logic_str[0] = logic_str[0] + " => " + f"{count} ({self.to_pct(count)}%)"

        return logic_condition, logic_str
