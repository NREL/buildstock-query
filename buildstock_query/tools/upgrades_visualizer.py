"""
- - - - - - - - -
Upgrades Visualizer
Experimental Stage.
:author: Rajendra.Adhikari@nrel.gov
"""

from functools import reduce
from buildstock_query import BuildStockQuery, KWH2MBTU
import numpy as np
import re
from collections import defaultdict, Counter
import dash_bootstrap_components as dbc
from dash import html, ALL, dcc, ctx
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from dash_extensions.enrich import MultiplexerTransform, DashProxy
# import os
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import pandas as pd
from InquirerPy import inquirer
import plotly.express as px

# os.chdir("/Users/radhikar/Documents/eulpda/EULP-data-analysis/eulpda/smart_query/")
# from: https://github.com/thedirtyfew/dash-extensions/tree/1b8c6466b5b8522690442713eb421f622a1d7a59
# app = DashProxy(transforms=[
#     # TriggerTransform(),  # enable use of Trigger objects
#     MultiplexerTransform(),  # makes it possible to target an output multiple times in callbacks
#     # ServersideOutputTransform(),  # enable use of ServersideOutput objects
#     # NoOutputTransform(),  # enable callbacks without output
#     # BlockingCallbackTransform(),  # makes it possible to skip callback invocations while a callback is running
#     # LogTransform()  # makes it possible to write log messages to a Dash component
# ])
transforms = [MultiplexerTransform()]

# yaml_path = "/Users/radhikar/Documents/eulpda/EULP-data-analysis/notebooks/EUSS-project-file-example.yml"
yaml_path = "/Users/radhikar/Documents/largee/resstock/project_national/fact_sheets_category_1.yml"
opt_sat_path = "/Users/radhikar/Downloads/options_saturations.csv"
default_end_use = "fuel_use_electricity_total_m_btu"


def filter_cols(all_columns, prefixes=[], suffixes=[]):
        cols = []
        for col in all_columns:
            for prefix in prefixes:
                if col.startswith(prefix):
                    cols.append(col)
                    break
            else:
                for suffix in suffixes:
                    if col.endswith(suffix):
                        cols.append(col)
                        break
        return cols


def get_app(yaml_path: str, opt_sat_path: str, db_name: str = 'euss-tests',
            table_name: str = 'res_test_03_2018_10k_20220607',
            workgroup: str = 'largeee',
            buildstock_type: str = 'resstock'):
    if isinstance(table_name, tuple) or isinstance(table_name, list):
        baseline_run = BuildStockQuery(workgroup=workgroup,
                                       db_name=db_name,
                                       buildstock_type=buildstock_type,
                                       table_name=table_name[0],
                                       skip_reports=False)
        baseline_table_name = table_name[0] + "_baseline"
        upgrade_table_name = table_name[1] + "_upgrades"
        ts_table_name = table_name[1] + "_timeseries"
        tables = (baseline_table_name, ts_table_name, upgrade_table_name)
    else:
        baseline_run = None
        tables = table_name
    upgrade_run = BuildStockQuery(workgroup=workgroup,
                                  db_name=db_name,
                                  buildstock_type=buildstock_type,
                                  table_name=tables,
                                  skip_reports=False)
    upgrade2monthly_res = {}

    def save_monthly_result(upgrade, build_df):
        if upgrade == 0 and baseline_run is not None:
            run_obj = baseline_run
        else:
            run_obj = upgrade_run
        all_cols = [str(c.name) for c in run_obj.get_cols(table=run_obj.ts_table)]
        all_cols = filter_cols(all_cols, suffixes=['_kbtu', '_kwh', '_lb'])
        monthly_vals = run_obj.agg.aggregate_timeseries(enduses=all_cols,
                                                        group_by=[run_obj.bs_bldgid_column],
                                                        upgrade_id=0,
                                                        timestamp_grouping_func='month')
        monthly_vals = monthly_vals.set_index('building_id')
        monthly_vals['Month'] = monthly_vals['time'].dt.month_name()
        for col in monthly_vals.columns:
            if col.endswith('kwh'):
                monthly_vals[col] *= KWH2MBTU
        monthly_vals = monthly_vals.rename(columns=lambda col: col.replace('kwh', 'mbtu'))
        monthly_vals = build_df.join(monthly_vals)
        upgrade2monthly_res[upgrade] = monthly_vals

    report = upgrade_run.report.get_success_report()
    available_upgrades = list(report.index)
    available_upgrades.remove(0)
    euss_ua = upgrade_run.get_upgrades_analyzer(yaml_path, opt_sat_file=opt_sat_path)
    upgrade2name = {indx+1: f"Upgrade {indx+1}: {upgrade['upgrade_name']}" for indx,
                    upgrade in enumerate(euss_ua.get_cfg().get('upgrades', []))}
    upgrade2name[0] = "Upgrade 0: Baseline"
    upgrade2shortname = {indx+1: f"Upgrade {indx+1}" for indx,
                         upgrade in enumerate(euss_ua.get_cfg().get('upgrades', []))}
    # allupgrade2name = {0: "Upgrade 0: Baseline"} | upgrade2name
    change_types = ["any", "no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng"]
    chng2bldg = {}
    for chng in change_types:
        for upgrade in available_upgrades:
            print(f"Getting buildings for {upgrade} and {chng}")
            chng2bldg[(upgrade, chng)] = upgrade_run.report.get_buildings_by_change(upgrade_id=int(upgrade),
                                                                                    change_type=chng)
    download_csv_df = pd.DataFrame()
    resolution = 'annual'

    res_csv_df = upgrade_run.get_results_csv_full()
    upgrade_run.save_cache()
    res_csv_df = res_csv_df[res_csv_df['completed_status'] == 'Success']
    sample_weight = res_csv_df['build_existing_model.sample_weight'].iloc[0]
    res_csv_df['upgrade'] = 0
    build_cols = [c for c in res_csv_df.columns if c.startswith('build_existing_model.')]
    build_df = res_csv_df[build_cols]
    res_csv_df = res_csv_df.rename(columns={'upgrade_costs.upgrade_cost_usd': 'upgrade_cost_total_usd'})
    res_csv_df = res_csv_df.rename(columns=lambda x: x.split('.')[1] if '.' in x else x)
    res_csv_df = res_csv_df.drop(columns=['applicable', 'output_format'])  # These are useless columns
    # all_up_csvs = [res_csv_df]

    upgrade2res = {0: res_csv_df}
    save_monthly_result(0, build_df)
    for upgrade in available_upgrades:
        print(f"Getting up_csv for {upgrade}")
        up_csv = upgrade_run.get_upgrades_csv_full(upgrade_id=int(upgrade))
        save_monthly_result(upgrade, build_df)
        upgrade_run.save_cache()
        # print(list(up_csv.columns))
        # print(list(res_csv_df.columns))
        # print("upgrade", i, set(up_csv.columns)  - set(res_csv_df.columns))
        # print("upgrade", i, set(res_csv_df.columns)  - set(up_csv.columns))
        up_csv = up_csv.loc[res_csv_df.index]
        up_csv = up_csv.join(build_df)
        up_csv = up_csv.rename(columns={'upgrade_costs.upgrade_cost_usd': 'upgrade_cost_total_usd'})
        up_csv = up_csv.rename(columns=lambda x: x.split('.')[1] if '.' in x else x)
        up_csv = up_csv.drop(columns=['applicable', 'output_format'])
        up_csv['upgrade'] = up_csv['upgrade'].map(lambda x: int(x))
        invalid_rows_keys = up_csv['completed_status'] == 'Invalid'
        invalid_rows = up_csv[invalid_rows_keys].copy()
        if len(invalid_rows) > 0:
            invalid_rows.update(res_csv_df[invalid_rows_keys])
            invalid_rows['completed_status'] = 'Invalid'
            up_csv[invalid_rows_keys] = invalid_rows
        # up_csv = up_csv.reset_index().set_index(['upgrade'])
        upgrade2res[upgrade] = up_csv
    all_cols = res_csv_df.columns
    emissions_cols = filter_cols(all_cols, suffixes=['_lb'])
    end_use_cols = filter_cols(all_cols, ["end_use_", "energy_use__", "fuel_use_"])
    water_usage_cols = filter_cols(all_cols, suffixes=["_gal"])
    load_cols = filter_cols(all_cols, ["load_", "flow_rate_"])
    peak_cols = filter_cols(all_cols, ["peak_"])
    unmet_cols = filter_cols(all_cols, ["unmet_"])
    area_cols = filter_cols(all_cols, suffixes=["_ft_2", ])
    size_cols = filter_cols(all_cols, ["size_"])
    qoi_cols = filter_cols(all_cols, ["qoi_"])
    cost_cols = filter_cols(all_cols, ["upgrade_cost_"])
    char_cols = [c.removeprefix('build_existing_model.') for c in build_cols if 'applicable' not in c]
    fuels_types = ['electricity', 'natural_gas', 'propane', 'fuel_oil', 'coal', 'wood_cord', 'wood_pellets']

    def get_res(upgrade: int, enduses: list[str], group_by: list[str] | None = None, applied_only: bool = False):
        if upgrade == 0:
            res_df = upgrade2res[0].copy()
        elif applied_only:
            res = upgrade2res[int(upgrade)].copy()
            res = res[res['completed_status'] != 'Invalid']
            res_df = res
        else:
            res_df = upgrade2res[int(upgrade)].copy()
        group_by = group_by or []
        if resolution == 'monthly':
            if upgrade == 0 and baseline_run is not None:
                run_obj = baseline_run
            else:
                run_obj = upgrade_run
            monthly_vals = run_obj.agg.aggregate_timeseries(enduses=enduses,
                                                            group_by=[run_obj.ts_bldgid_column],
                                                            upgrade_id=upgrade,
                                                            timestamp_grouping_func='month')
            run_obj.save_cache()
            monthly_vals = monthly_vals.set_index('building_id')
            monthly_vals['Month'] = monthly_vals['time'].dt.month_name()
            for enduse in enduses:
                if enduse.endswith('kwh'):
                    monthly_vals[enduse] *= KWH2MBTU
            monthly_vals.loc[:, 'value'] = monthly_vals[enduses].sum(axis=1)
            monthly_vals = monthly_vals.join(res_df[group_by])
            return monthly_vals[group_by + ['Month', 'value']]
        else:
            res_df.loc[:, 'value'] = res_df[enduses].sum(axis=1)
            baseline_df = res_df[group_by + ['value']]
            return baseline_df

    def get_buildings(upgrade, applied_only=False):
        if upgrade == 0:
            return upgrade2res[0].index
        elif applied_only:
            res = upgrade2res[int(upgrade)]
            res = res[res['completed_status'] != 'Invalid']
            return res.index
        else:
            return upgrade2res[int(upgrade)].index

    def explode_str(input_str):
        input_str = str(input_str).lower()
        input_str = [
            int(x) if x and x[0] in "0123456789" else x
            for x in re.split(r"([\<\-])|([0-9]+)", input_str)
        ]
        return tuple("X" if x is None else x for x in input_str)

    def csv_generator(end_use, savings_type, applied_only, change_type, sync_upgrade, filter_bldg=None,
                      report_upgrade: int = 0, group_cols=None):

        base_vals = get_res(0, end_use)
        base_df = base_vals.loc[filter_bldg] if filter_bldg is not None else base_vals.copy()

        if group_cols:
            res_df = get_res(report_upgrade, end_use, group_by=group_cols, applied_only=applied_only)
            res_df = res_df.sort_values(group_cols, key=lambda series: [explode_str(x) for x in series])
            if len(group_cols) > 1:
                grouped_df = res_df.groupby(group_cols, sort=False)
                df_generator = ((', '.join(indx), df) for (indx, df) in grouped_df)
            else:
                df_generator = ((indx, df) for indx, df in res_df.groupby(group_cols[0], sort=False))
        else:
            df_generator = ((f"Upgrade {upgrade}", get_res(upgrade, end_use, applied_only=applied_only)) for upgrade in [report_upgrade])

        for indx, res_df in df_generator:
            if change_type:
                chng_upgrade = int(sync_upgrade) if sync_upgrade else int(report_upgrade) if report_upgrade else 0
                if chng_upgrade and chng_upgrade > 0:
                    change_bldg_list = chng2bldg[(chng_upgrade, change_type)]
                else:
                    change_bldg_list = []
                res_df = res_df.loc[res_df.index.intersection(change_bldg_list)]

            if filter_bldg is not None:
                res_df = res_df.loc[res_df.index.intersection(filter_bldg)]
            if len(res_df) == 0:
                continue

            sub_df = res_df['value']
            if savings_type == 'Savings':
                sub_df = base_df.loc[sub_df.index, 'value'] - sub_df
            elif savings_type == 'Percent Savings':
                sub_base_df = base_df.loc[sub_df.index, 'value']
                saving_df = 100 * (sub_base_df - sub_df) / sub_base_df
                saving_df[(sub_base_df == 0)] = -100  # If base is 0, and upgrade is not, assume -100% savings
                saving_df[(sub_df == 0) & (sub_base_df == 0)] = 0
                sub_df = saving_df
            yield indx, sub_df

    def get_ylabel(end_use):
        if len(end_use) == 1:
            return end_use[0]
        pure_end_use_name = end_use[0].removeprefix("end_use_")
        pure_end_use_name = pure_end_use_name.removeprefix("fuel_use_")
        pure_end_use_name = "_".join(pure_end_use_name.split("_")[1:])
        return f"{len(end_use)}_fuels_{pure_end_use_name}"

    def get_scatter(end_use, savings_type='', applied_only=False, change_type='', show_all_points=False,
                    sync_upgrade=None, filter_bldg=None, group_cols=None, report_upgrade=0):
        fig = go.Figure()
        report_upgrade = report_upgrade or 0
        res_df = get_res(report_upgrade, end_use, group_by=group_cols, applied_only=applied_only)
        if filter_bldg is not None:
            res_df = res_df.loc[res_df.index.intersection(filter_bldg)]

        sub_df = res_df['value'].copy()
        base_df = get_res(0, end_use, group_by=group_cols).copy()
        base_df['baseline_vals'] = base_df['value']
        base_df = base_df.loc[filter_bldg] if filter_bldg is not None else base_df
        base_df = base_df.loc[sub_df.index]
        ytitle = f"Upgrade {savings_type} values"
        xtitle = "Baseline absolute values"
        if savings_type == 'Absolute':
            base_df['upgrade_vals'] = sub_df
        elif savings_type == 'Savings':
            base_df['upgrade_vals'] = base_df['baseline_vals'] - sub_df
        elif savings_type == 'Percent Savings':
            base_df['upgrade_vals'] = 100 * (base_df['baseline_vals'] - sub_df) / base_df['baseline_vals']
            # If base is 0, and upgrade is not, assume -100% savings
            base_df.loc[(base_df['baseline_vals'] == 0), "upgrade_vals"] = -100
            base_df.loc[(sub_df == 0) & (base_df['baseline_vals'] == 0), 'upgrade_vals'] = 0

        base_df = base_df.reset_index()
        base_df['hovertext'] = base_df['building_id'].apply(lambda bid: f'{upgrade2name.get(int(report_upgrade))}'
                                                            f'<br>Building: {bid}<br>Sample Count: {len(base_df)}')
        report_df = base_df.rename(columns={'baseline_vals': xtitle, 'upgrade_vals': ytitle, 'hovertext': 'info'})
        if group_cols:
            if len(group_cols) == 1:
                fig = px.scatter(base_df, hover_name='hovertext', x='baseline_vals', y="upgrade_vals",
                                 facet_col=group_cols[0],
                                 labels={'baseline_vals': '', 'upgrade_vals': ''})
                report_df = report_df[['building_id', xtitle, ytitle,  group_cols[0], 'info']]
            elif len(group_cols) > 1:
                fig = px.scatter(base_df, hover_name='hovertext', x='baseline_vals', y="upgrade_vals",
                                 facet_col=group_cols[0], facet_row=group_cols[1],
                                 labels={'baseline_vals': '', 'upgrade_vals': ''})
                report_df = report_df[['building_id', xtitle, ytitle,  group_cols[0], group_cols[1], 'info']]
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        else:
            fig = px.scatter(base_df, hover_name='hovertext', x='baseline_vals', y="upgrade_vals",
                             labels={'baseline_vals': '', 'upgrade_vals': ''})
            report_df = report_df[['building_id', xtitle, ytitle, 'info']]
        title = f"{get_ylabel(end_use)}"
        fig.update_layout(boxmode="group",
                          title=f"{title} for {change_type} buildings" if change_type else f'{title}',
                          clickmode='event+select')
        fig.add_annotation(x=-0.02, y=0.5,
                           text=ytitle, textangle=-90,
                           arrowcolor="white",
                           xref="paper", yref="paper")
        fig.add_annotation(x=0.5, y=-0.17,
                           text=xtitle, textangle=0,
                           arrowcolor="white",
                           xref="paper", yref="paper")
        return fig, report_df

    def get_distribution(end_use, savings_type='', applied_only=False, change_type='', show_all_points=False,
                         sync_upgrade=None, filter_bldg=None, group_cols=None, report_upgrade=0):
        fig = go.Figure()
        counter = 0
        report_dfs = []
        if show_all_points:
            points = 'all'
            upgrades_to_plot = [report_upgrade] if report_upgrade else [0]
        else:
            points = 'suspectedoutliers'
            upgrades_to_plot = [0] + available_upgrades
        ytitle = f"{get_ylabel(end_use)}"
        xtitle = ", ".join(group_cols) if group_cols else 'Upgrade'
        for upgrade in upgrades_to_plot:
            yvals = []
            xvals = []
            sample_counts = []
            hovervals = []
            all_building_ids = []
            for indx, sub_df in csv_generator(end_use, savings_type, applied_only, change_type, sync_upgrade,
                                              filter_bldg, upgrade, group_cols):
                building_ids = list(sub_df.index)
                count = sum(sub_df < float('inf'))
                if counter >= 200:
                    sub_df = pd.DataFrame()
                    fig.add_trace(go.Box(
                        y=[],
                        name="Too many groups"
                    ))
                    break
                else:
                    xvals.extend([indx]*len(sub_df))
                    yvals.extend(sub_df.values)
                    all_building_ids.extend(building_ids)
                    sample_counts.extend([count] * len(sub_df))
                    hovertext = [f'{upgrade2name.get(upgrade)}<br>{indx}<br> Building: {bid}<br>Sample Count: {count}'
                                 for bid in building_ids]
                    hovervals.extend(hovertext)
                counter += 1

            fig.add_trace(go.Box(
                y=yvals,
                x=xvals,
                name=f'Upgrade {upgrade}',
                boxpoints=points,
                boxmean=True,  # represent mean
                hovertext=hovervals,
                hoverinfo="all"
            ))
            try:
                df = pd.DataFrame({'building_ids': all_building_ids, xtitle: xvals, ytitle: yvals,
                                   'upgrade': f'Upgrade {upgrade}', 'sample_count': sample_counts, 'info': hovervals})
            except Exception as exp:
                print(exp)
                continue
            report_dfs.append(df)

        fig.update_layout(yaxis_title=ytitle,
                          boxmode="group",
                          xaxis_title=xtitle,
                          title=f"Distribution for {change_type} buildings" if change_type else 'Distribution',
                          clickmode='event+select')
        return fig, pd.concat(report_dfs)

    def get_bars(end_use, value_type='mean', savings_type='', applied_only=False, change_type='',
                 sync_upgrade=None, filter_bldg=None, group_cols=None):
        fig = go.Figure()
        counter = 0
        report_dfs = []
        xtitle = ", ".join(group_cols) if group_cols else 'Upgrade'
        ytitle = f"{get_ylabel(end_use)}_{value_type}"
        for upgrade in [0] + available_upgrades:
            yvals = []
            xvals = []
            sample_counts = []
            upgrades = []
            hovervals = []
            for indx, up_vals in csv_generator(end_use, savings_type, applied_only, change_type, sync_upgrade,
                                               filter_bldg, upgrade, group_cols):

                count = len(up_vals)
                if value_type.lower() == 'total':
                    val = up_vals.sum() * sample_weight
                elif value_type.lower() == 'count':
                    val = up_vals.count()
                else:
                    val = up_vals.mean()
                if counter >= 200:
                    yvals.append(0)
                    xvals.append("Too many groups")
                    sample_counts.append(0)
                    upgrades.append(upgrade)
                    hovervals.append("Too many groups")
                    break
                else:
                    yvals.append(val)
                    xvals.append(indx)
                    sample_counts.append(count)
                    upgrades.append(upgrade)
                    hovertext = f"{upgrade2name.get(upgrade)}<br>{indx}<br>Average {val}. <br>Sample Count: {count}."
                    f"<br>Units Count: {count * sample_weight}."
                    hovervals.append(hovertext)
                counter += 1

            fig.add_trace(go.Bar(
                y=yvals,
                x=xvals,
                hovertext=hovervals,
                name=f'Upgrade {upgrade}',
                hoverinfo="all"
            )).update_traces(
                marker={"line": {"width": 0.5, "color": "rgb(0,0,0)"}}
            )
            try:
                df = pd.DataFrame({xtitle: xvals, ytitle: yvals, 'upgrade': [f'Upgrade {upgrade}'] * len(xvals),
                                   'sample_count': sample_counts, 'info': hovervals})
            except Exception:
                continue
            report_dfs.append(df)

        fig.update_layout(yaxis_title=ytitle,
                          barmode='group',
                          xaxis_title=xtitle,
                          title=f"{value_type} for {change_type} buildings" if change_type else f'{value_type}')

        return fig, pd.concat(report_dfs)

    def get_all_cols():
        if resolution == "annual":
            all_cols = [str(c.name) for c in upgrade_run.get_cols(table=upgrade_run.bs_table)]
            all_cols = [col.split('.')[1] if '.' in col else col for col in all_cols]
        else:
            assert upgrade_run.ts_table is not None
            all_cols = [str(c.name) for c in upgrade_run.get_cols(table=upgrade_run.ts_table)]
        return all_cols

    def get_all_end_use_cols():
        all_cols = get_all_cols()
        all_end_use_cols = filter_cols(all_cols, ["end_use_", "energy_use_", "fuel_use_"])
        return all_end_use_cols

    def get_end_use_cols(fuel):
        cols = []
        all_end_use_cols = get_all_end_use_cols()
        sep = "_" if resolution == "annual" else "__"
        for c in all_end_use_cols:
            if fuel in c or fuel == 'All':
                c = c.removeprefix(f"end_use{sep}{fuel}{sep}")
                c = c.removeprefix(f"fuel_use{sep}{fuel}{sep}")
                if fuel == 'All':
                    for f in sorted(fuels_types):
                        c = c.removeprefix(f"end_use{sep}{f}{sep}")
                        c = c.removeprefix(f"fuel_use{sep}{f}{sep}")
                cols.append(c)
        no_dup_cols = {c: None for c in cols}
        return list(no_dup_cols.keys())

    def get_emissions_cols():
        all_cols = get_all_cols()
        all_emissions_cols = filter_cols(all_cols, ["emissions_"])
        return all_emissions_cols

    def get_energy_db_cols(fuel, end_use):
        all_enduses = get_all_end_use_cols()
        if not end_use:
            return all_enduses[0]
        valid_cols = []
        sep = "_" if resolution == "annual" else "__"
        prefix = "fuel_use" if end_use.startswith("total") else "end_use"
        if fuel == 'All':
            valid_cols.extend(f"{prefix}{sep}{f}{sep}{end_use}" for f in fuels_types
                              if f"{prefix}{sep}{f}{sep}{end_use}" in all_enduses)

        else:
            valid_cols.append(f"{prefix}{sep}{fuel}{sep}{end_use}")
        return valid_cols

    def get_opt_report(upgrade, bldg_id):
        applied_options = list(upgrade_run.report.get_applied_options(upgrade_id=int(upgrade), bldg_ids=[bldg_id])[0])
        applied_options = [val for key, val in upgrade2res[upgrade].loc[bldg_id].items() if
                           key.startswith("option_") and key.endswith("_name")
                           and not (isinstance(val, float) and np.isnan(val))]

        opt_vals = [(opt.split('|')[0], opt.split('|')[1]) for opt in applied_options]
        char_cols = ['_'.join(opt.lower().split('|')[0].split()) for opt in applied_options]
        baseline_vals = upgrade2res[0].loc[bldg_id][char_cols]
        option_report = [f"{opt}: {base_val} => {up_val}" for (opt, up_val), base_val in zip(opt_vals, baseline_vals)]
        return option_report

    def get_baseline_chars(bldg_id, char_types=None):
        baseline_vals = upgrade2res[0].loc[bldg_id][char_cols]
        char_types = char_types or []
        return_list = []
        for char_type in char_types:
            return_val = [f'{k}: {v}' for k, v in baseline_vals.items()
                          if char_type in k]
            return_list += return_val

        return return_list

    external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]

    app = DashProxy(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], transforms=transforms,
                    external_scripts=external_script)
    app.layout = html.Div([dbc.Container(html.Div([
        dcc.Download(id="download-dataframe-csv"),
        dbc.Row([dbc.Col(html.H1("Upgrades Visualizer"), width='auto'), dbc.Col(html.Sup("beta"))]),
        # Add a row for annual, vs monthly vs seasonal plot radio buttons
        dbc.Row([dbc.Col(dbc.Label("Resolution: "), width='auto'),
                 dbc.Col(dcc.RadioItems(["annual", "monthly"], "annual",
                                        inline=True, id="radio_resolution"))]),

        dbc.Row([dbc.Col(dbc.Label("Visualization Type: "), width='auto'),
                 dbc.Col(dcc.RadioItems(["Mean", "Total", "Count", "Distribution", "Scatter"], "Mean",
                                        id="radio_graph_type",
                                        inline=True,
                                        labelClassName="pr-2"), width='auto'),
                #  dbc.Col(dbc.Collapse(children=[dcc.Checklist(['Show all points'], [],
                 #                                               inline=True, id='check_all_points')
                 #                                 ],
                 #                       id="collapse_points", is_open=True), width='auto'),
                 dbc.Col(children=[dcc.Checklist(['Show all points'], [],
                                                 inline=True, id='check_all_points')
                                   ],),
                 dbc.Col(dbc.Label("Value Type: "), width='auto'),
                 dbc.Col(dcc.RadioItems(["Absolute", "Savings", "Percent Savings"], "Absolute",  inline=True,
                                        id='radio_savings', labelClassName="pr-2"), width='auto'),
                 dbc.Col(dcc.Checklist(options=['Applied Only'], value=['Applied Only'],
                                       inline=True, id='chk_applied_only'), width='auto')
                 ]),
        dbc.Row([dbc.Col(html.Br())]),
        dbc.Row([dbc.Col(dcc.Loading(id='graph-loader', children=[html.Div(id='loader_label')]))]),
        dbc.Row([dbc.Col(dcc.Graph(id='graph'))]),
        dbc.Row([dbc.Col(dbc.Button("Download", id='csv-download'))], justify='end'),
        dcc.Store(id='graph-data-store'),
    ])),
        dbc.Row([dbc.Col(
            dcc.Tabs(id='tab_view_type', value='energy', children=[
                dcc.Tab(id='energy_tab', label='Energy', value='energy', children=[
                    dcc.RadioItems(fuels_types + ['All'], "electricity", id='radio_fuel',  inline=True,
                                   labelClassName="pr-2")]
                        ),
                dcc.Tab(label='Water Usage', value='water', children=[]
                        ),
                dcc.Tab(label='Load', value='load', children=[]
                        ),
                dcc.Tab(label='Peak', value='peak', children=[]
                        ),
                dcc.Tab(label='Unmet Hours', value='unmet_hours', children=[]
                        ),
                dcc.Tab(label='Area', value='area', children=[]
                        ),
                dcc.Tab(label='Size', value='size', children=[]
                        ),
                dcc.Tab(label='QOI', value='qoi', children=[]
                        ),
                dcc.Tab(label='emissions', value='emissions', children=[]
                        ),
                dcc.Tab(label='Upgrade Cost', value='upgrade_cost', children=[]
                        ),
            ])
        )
        ], className="mx-5 mt-5"),
        dbc.Row([dbc.Col(dcc.Dropdown(id='dropdown_enduse'))], className="mx-5 my-1"),
        dbc.Row(
            dbc.Col([
                dbc.Row([dbc.Col(html.Div("Restrict to buildings that have "), width='auto'),
                         dbc.Col(dcc.Dropdown(change_types, "", placeholder="Select change type...",
                                              id='dropdown_chng_type'), width='2'),
                         dbc.Col(html.Div(" in "), width='auto'),
                         dbc.Col(dcc.Dropdown(id='sync_upgrade', value='',
                                              options={}))
                         ],
                        className="flex items-center"),
                dbc.Row([dbc.Col(html.Div("Select:"), style={"padding-left": "12px", "padding-right": "0px"},
                                 width='auto'),
                         dbc.Col(dcc.Dropdown(id='input_building'), width=1),
                         dbc.Col(html.Div("("), width='auto',
                                 style={"padding-left": "0px", "padding-right": "0px"}),
                         dbc.Col(dcc.Checklist(['Lock)'], [],
                                               inline=True, id='chk-lock'),
                                 width='auto', style={"padding-left": "0px", "padding-right": "0px"}),
                         dbc.Col(html.Div(" in "), style={"padding-right": "0px"}, width='auto'),
                         dbc.Col(dcc.Dropdown(id='report_upgrade', value='', placeholder="Upgrade ...",
                                              options=upgrade2shortname), width=1),
                         dbc.Col(html.Div("grouped by:"), style={"padding-right": "0px"}, width='auto'),
                         dbc.Col(dcc.Dropdown(id='drp-group-by', options=char_cols, value=None,
                                              multi=True, placeholder="Select characteristics..."),
                                 width=3),
                         dbc.Col(dbc.Button("<= Copy", id="btn-copy", color="primary", size="sm",
                                            outline=True), class_name="centered-col", style={"padding-right": "0px"}),
                         dbc.Col(html.Div("Extra restriction: "), style={"padding-right": "0px"}, width='auto'),
                         dbc.Col(dcc.Dropdown(id='input_building2', disabled=False), width=1),
                         dbc.Col(dcc.Checklist(id='chk-graph', options=['Graph'], value=[],
                                               inline=True), width='auto'),
                         dbc.Col(dcc.Checklist(id='chk-options', options=['Options'], value=[],
                                               inline=True), width='auto'),
                         dbc.Col(dcc.Checklist(id='chk-enduses', options=['Enduses'], value=[],
                                               inline=True), width='auto'),
                         dbc.Col(dcc.Checklist(id='chk-chars', options=['Chars'], value=[],
                                               inline=True), width='auto'),
                         dbc.Col(dbc.Button("Reset", id="btn-reset", color="primary", size="sm", outline=True),
                                 width='auto'),
                         ]),
                dbc.Row([dbc.Col([
                    dbc.Row(html.Div(id="options_report_header")),
                    dbc.Row(dcc.Loading(id='opt-loader',
                                        children=html.Div(id="options_report"))),
                    dcc.Store("opt_report_store")
                ], width=5),
                    dbc.Col([
                        dbc.Row([dbc.Col(html.Div("View enduse that: "), width='auto'),
                                 dbc.Col(dcc.Dropdown(id='input_enduse_type',
                                                      options=['changed', 'increased', 'decreased', 'are almost zero'],
                                                      value='changed', clearable=False),
                                         width=2),
                                 dbc.Col(html.Div(), width='auto'),
                                 dbc.Col(html.Div("Charecteristics Report:"), width='auto'),
                                 dbc.Col(dcc.Dropdown(id='drp-char-report', options=char_cols, value=None,
                                                      multi=True, placeholder="Select characteristics...")),
                                 ]),
                        dbc.Row([dbc.Col(dcc.Loading(id='enduse_loader',
                                                     children=html.Div(id="enduse_report"))
                                         ),
                                 dbc.Col(dcc.Loading(id='char_loader',
                                                     children=html.Div(id="char_report"))
                                         ),
                                 ]),
                        dcc.Store("enduse_report_store"),
                        dcc.Store("char_report_store")
                    ], width=7)

                ]),
                dbc.Row([
                    dbc.Col(width=5),
                    dbc.Col(
                        dbc.Row([
                            dbc.Col(),
                            dbc.Col(dbc.Button("Download All Characteristics", id="btn-dwn-chars",
                                               color="primary",
                                               size="sm", outline=True), width='auto'),
                        ]), width=7)
                ])
            ]), className="mx-5 my-1"),
        html.Div(id="status_bar"),
        dcc.Download(id="download-chars-csv"),
        dcc.Store("uirevision")
        # dbc.Button("Kill me", id="button110")
    ])

    # download data with button click
    @app.callback(
        Output("download-dataframe-csv", "data"),
        Input("csv-download", "n_clicks"),
        prevent_initial_call=True)
    def download_csv(n_clicks):
        if not n_clicks:
            raise PreventUpdate()
        nonlocal download_csv_df
        return dcc.send_data_frame(download_csv_df.to_csv, "graph_data.csv")

    @app.callback(
        Output("download-chars-csv", "data"),
        Input("btn-dwn-chars", "n_clicks"),
        State("input_building", "value"),
        State("input_building", "options"),
        State("input_building2", "options"),
        State('chk-chars', 'value'),
    )
    def download_char(n_clicks, bldg_id, bldg_options, bldg_options2, chk_chars):
        if not n_clicks:
            raise PreventUpdate()

        if "Chars" in chk_chars and bldg_options2:
            bldg_ids = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options2]
        else:
            bldg_ids = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options]
        bldg_ids = [int(b) for b in bldg_ids]
        bdf = res_csv_df[char_cols].loc[bldg_ids]
        return dcc.send_data_frame(bdf.to_csv, f"chars_{n_clicks}.csv")

    def get_elligible_output_columns(category, fuel):
        if category == 'energy':
            elligible_cols = get_end_use_cols(fuel)
        elif category == 'water':
            elligible_cols = water_usage_cols if resolution == 'annual' else []
        elif category == 'load':
            elligible_cols = load_cols if resolution == 'annual' else []
        elif category == 'peak':
            elligible_cols = peak_cols if resolution == 'annual' else []
        elif category == 'unmet_hours':
            elligible_cols = unmet_cols if resolution == 'annual' else []
        elif category == 'area':
            elligible_cols = area_cols if resolution == 'annual' else []
        elif category == 'size':
            elligible_cols = size_cols if resolution == 'annual' else []
        elif category == 'qoi':
            elligible_cols = qoi_cols if resolution == 'annual' else []
        elif category == 'emissions':
            elligible_cols = emissions_cols if resolution == 'annual' else get_emissions_cols()
        elif category == 'upgrade_cost':
            elligible_cols = cost_cols if resolution == 'annual' else []
        else:
            raise ValueError(f"Invalid tab {category}")
        return elligible_cols

    @app.callback(
            Output('radio_resolution', 'options'),
            Input('radio_resolution', 'value'),
    )
    def update_resolution(res):
        nonlocal resolution
        resolution = res
        return ['annual', 'monthly']

    @app.callback(
        Output('dropdown_enduse', "options"),
        Output('dropdown_enduse', "value"),
        Input('tab_view_type', "value"),
        Input('radio_fuel', "value"),
        Input('dropdown_enduse', "value"),
        Input('radio_resolution', 'value')
    )
    def update_enduse_options(view_tab, fuel_type, current_enduse, resolution):
        elligible_cols = get_elligible_output_columns(view_tab, fuel_type)
        enduse = current_enduse if current_enduse in elligible_cols else elligible_cols[0]
        return sorted(elligible_cols), enduse

    # @app.callback(
    #     Output("collapse_points", 'is_open'),
    #     Input('radio_graph_type', "value")
    # )
    # def disable_showpoints(graph_type):
    #     print(f"Graph type: {graph_type.lower() == 'distribution'}")
    #     return True
    #     # return graph_type.lower() == "distribution"

    @app.callback(
        Output("sync_upgrade", 'value'),
        Output("sync_upgrade", 'options'),
        Output("sync_upgrade", 'placeholder'),
        Input('dropdown_chng_type', "value"),
        State("sync_upgrade", "value"),
        State("sync_upgrade", "options")
    )
    def update_sync_upgrade(chng_type, current_sync_upgrade, sync_upgrade_options):
        # print(chng_type, current_sync_upgrade, sync_upgrade_options)
        if chng_type:
            return current_sync_upgrade, upgrade2name, 'respective upgrades. (Click to restrict to specific upgrade)'
        else:
            return '', {}, ' <select a change type on the left first>'

    @app.callback(
        Output('input_building', 'placeholder'),
        Input('input_building', 'options')
    )
    def update_building_placeholder(options):
        return f"{len(options)} Buidlings" if options else "0 buildings."

    @app.callback(
        Output('input_building2', 'placeholder'),
        Output('input_building2', 'value'),
        Input('input_building2', 'value'),
        Input('input_building2', 'options')
    )
    def update_building_placeholder2(value, options):
        return f"{len(options)} Buidlings" if options else "No restriction", None

    @app.callback(
        Output('input_building', 'value'),
        Output('input_building', 'options'),
        Output('report_upgrade', 'value'),
        Output('check_all_points', "value"),
        Input('graph', "selectedData"),
        State('input_building', 'options'),
        State('report_upgrade', 'value')
    )
    def graph_click(selection_data, current_options, current_upgrade):
        if not selection_data or 'points' not in selection_data or len(selection_data['points']) < 1:
            raise PreventUpdate()

        selected_buildings = []
        selected_upgrades = []
        for point in selection_data['points']:
            if not (match := re.search(r"Building: (\d*)", point.get('hovertext', ''))):
                continue
            if bldg := match.groups()[0]:
                upgrade_match = re.search(r"Upgrade (\d*)", point.get('hovertext', ''))
                upgrade = upgrade_match.groups()[0] if upgrade_match else ''
                selected_buildings.append(bldg)
                selected_upgrades.append(upgrade)

        if not selected_buildings:
            raise PreventUpdate()

        selected_upgrade = selected_upgrades[0] or current_upgrade
        if len(selected_buildings) != 1:
            selected_buildings = list(set(selected_buildings))
            return '', selected_buildings, selected_upgrade, ['Show all points']
        current_options = current_options or selected_buildings
        return selected_buildings[0], current_options, selected_upgrade, ['Show all points']

    @app.callback(
        Output('check_all_points', 'value'),
        Input('input_building', 'value'),
        State('input_building', 'options'),
        State('check_all_points', 'value'))
    def uncheck_all_points(bldg_selection, bldg_options, current_val):
        if not bldg_selection and bldg_options:
            return [''] if len(bldg_options) > 30000 else current_val
        raise PreventUpdate()

    @app.callback(
        Output('chk-graph', 'value'),
        Output('chk-options', 'value'),
        Output('chk-enduses', 'value'),
        Output('input_building2', 'options'),
        Output("uirevision", "data"),
        Input('btn-reset', "n_clicks")
    )
    def reset(n_clicks):
        return [], [], [], [], n_clicks

    @app.callback(
        Output('input_building', 'options'),
        Input('btn-copy', "n_clicks"),
        State('input_building2', 'options'),
    )
    def copy(n_clicks, bldg_options2):
        return bldg_options2 or dash.no_update

    @app.callback(
        Output('input_building', 'value'),
        Output('input_building', 'options'),
        Output('input_building2', 'options'),
        State('input_building', 'value'),
        State('input_building', 'options'),
        State('input_building2', 'options'),
        Input('chk-lock', 'value'),
        Input('dropdown_chng_type', "value"),
        Input('sync_upgrade', 'value'),
        Input('report_upgrade', 'value'),
        Input('btn-reset', "n_clicks")
    )
    def bldg_selection(current_bldg, current_options, current_options2, chk_lock,
                       change_type, sync_upgrade, report_upgrade,
                       reset_click):

        if sync_upgrade and change_type:
            valid_bldgs = list(sorted(chng2bldg[(int(sync_upgrade), change_type)]))
        elif report_upgrade and change_type:
            valid_bldgs = list(sorted(chng2bldg[(int(report_upgrade), change_type)]))
            buildings = get_buildings(report_upgrade, applied_only=True)
            valid_bldgs = list(buildings.intersection(valid_bldgs))
        elif report_upgrade:
            buildings = get_buildings(report_upgrade, applied_only=True)
            valid_bldgs = list(buildings)
        else:
            valid_bldgs = list(upgrade2res[0].index)

        base_res = upgrade2res[0].loc[valid_bldgs]
        valid_bldgs = [str(b) for b in list(base_res.index)]

        if "btn-reset" != ctx.triggered_id and current_options and len(current_options) > 0 and chk_lock:
            current_options_set = set(current_options)
            valid_bldgs = [b for b in valid_bldgs if b in current_options_set]

        valid_bldgs2 = []

        if current_bldg and current_bldg not in valid_bldgs:
            current_bldg = valid_bldgs[0] if valid_bldgs else ''
        return current_bldg, valid_bldgs, valid_bldgs2

    def get_char_choices(char):
        if char:
            res0 = upgrade2res[0]
            unique_choices = sorted(list(res0[char].unique()))
            return unique_choices, unique_choices[0]
        else:
            return [], None

    def get_action_button_pairs(id, bldg_list_dict, report_type='opt'):
        buttons = []
        bldg_str = '' if len(bldg_list_dict[id]) > 10 else " [" + ','.join([str(b) for b in bldg_list_dict[id]]) + "]"
        for type in ['check', 'cross']:
            icon_name = "akar-icons:circle-check-fill" if type == 'check' else "gridicons:cross-circle"
            button = html.Div(dmc.ActionIcon(DashIconify(icon=icon_name,
                                                         width=20 if type == "check" else 22,
                                                         height=20 if type == "check" else 22,
                                                         ),
                                             id={'index': id, 'type': f'btn-{type}', 'report_type': report_type},
                                             variant="light"),
                              id=f"div-tooltip-target-{type}-{id}")
            if type == "check":
                tooltip = dbc.Tooltip(f"Select these buildings.{bldg_str}",
                                      target=f"div-tooltip-target-{type}-{id}", delay={'show': 1000})

                col = dbc.Col(html.Div([button, tooltip]),  width='auto', class_name="col-btn-cross",
                              style={"padding-left": "0px", "padding-right": "0px"})
            else:
                tooltip = dbc.Tooltip(f"Select all except these buildings.{bldg_str}",
                                      target=f"div-tooltip-target-{type}-{id}", delay={'show': 1000})
                col = dbc.Col(html.Div([button, tooltip]),  width='auto',  class_name="col-btn-check",
                              style={"padding-left": "0px", "padding-right": "0px"})
            buttons.append(col)
        return buttons

    @app.callback(
        Output('status_bar', "children"),
        Output('input_building2', "options"),
        Input({"type": "btn-check", "index": ALL, "report_type": "opt"}, "n_clicks"),
        Input({"type": "btn-cross", "index": ALL, "report_type": "opt"}, "n_clicks"),
        State("input_building", "options"),
        State("input_building2", "options"),
        State("chk-options", "value"),
        State("opt_report_store", "data"),
    )
    def opt_check_button_click(check_clicks, cross_clicks, bldg_options, bldg_options2, chk_options, opt_report):
        triggers = dash.callback_context.triggered_prop_ids
        if len(triggers) != 1:
            raise PreventUpdate()

        if "Options" in chk_options and bldg_options2:
            bldg_list = [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(b) for b in bldg_options]

        trigger_val = next(iter(triggers.values()))
        buildings = opt_report.get(trigger_val['index'], [])
        if trigger_val['type'] == 'btn-check':
            return '', [str(b) for b in buildings]

        bldg_set = set(buildings)
        except_buildings = [str(v) for v in bldg_list if int(v) not in bldg_set]
        return '', except_buildings

    @app.callback(
        Output('status_bar', "children"),
        Output('input_building2', "options"),
        Input({"type": "btn-check", "index": ALL, "report_type": "enduse"}, "n_clicks"),
        Input({"type": "btn-cross", "index": ALL, "report_type": "enduse"}, "n_clicks"),
        State("input_building", "options"),
        State("input_building2", "options"),
        State("chk-enduses", "value"),
        State("enduse_report_store", "data"),
    )
    def enduse_button_click(check_clicks, cross_clicks, bldg_options, bldg_options2, chk_enduses, opt_report):
        triggers = dash.callback_context.triggered_prop_ids
        if len(triggers) != 1:
            raise PreventUpdate()

        if "Enduses" in chk_enduses and bldg_options2:
            bldg_list = [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(b) for b in bldg_options]

        trigger_val = next(iter(triggers.values()))
        buildings = opt_report.get(trigger_val['index'], [])
        if trigger_val['type'] == 'btn-check':
            return '', [str(b) for b in buildings]

        bldg_set = set(buildings)
        except_buildings = [str(v) for v in bldg_list if int(v) not in bldg_set]
        return '', except_buildings

    @app.callback(
        Output('status_bar', "children"),
        Output('input_building2', "options"),
        Input({"type": "btn-check", "index": ALL, "report_type": "char"}, "n_clicks"),
        Input({"type": "btn-cross", "index": ALL, "report_type": "char"}, "n_clicks"),
        State("input_building", "options"),
        State("input_building2", "options"),
        State('chk-chars', 'value'),
        State("char_report_store", "data"),
    )
    def char_report_button_click(check_clicks, cross_clicks, bldg_options, bldg_options2, chk_char, char_report):
        triggers = dash.callback_context.triggered_prop_ids
        if len(triggers) != 1:
            raise PreventUpdate()

        if "Chars" in chk_char and bldg_options2:
            bldg_list = [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(b) for b in bldg_options]

        trigger_val = next(iter(triggers.values()))
        buildings = char_report.get(trigger_val['index'], [])
        if trigger_val['type'] == 'btn-check':
            return '', [str(b) for b in buildings]

        bldg_set = set(buildings)
        except_buildings = [str(v) for v in bldg_list if int(v) not in bldg_set]
        return '', except_buildings

    @app.callback(
        Output("options_report_header", "children"),
        Output("options_report", 'children'),
        Output("opt_report_store", "data"),
        Input('input_building', "value"),
        Input('input_building', "options"),
        Input('input_building2', "options"),
        State('report_upgrade', 'value'),
        Input('chk-options', 'value'),
    )
    def show_opt_report(bldg_id, bldg_options, bldg_options2, report_upgrade, chk_options):
        if not report_upgrade or not bldg_options:
            return "Select an upgrade to see options applied in that upgrade", [''], {}

        if dash.callback_context.triggered_id == 'input_building2' and "Options" not in chk_options:
            raise PreventUpdate()

        if "Options" in chk_options and bldg_options2:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options]

        applied_options = upgrade_run.report.get_applied_options(upgrade_id=int(report_upgrade), bldg_ids=bldg_list,
                                                                 include_base_opt=True)
        opt_only = [{entry.split('|')[0] for entry in opt.keys()} for opt in applied_options]
        reduced_set = list(reduce(set.union, opt_only))

        nested_dict = defaultdict(lambda: defaultdict(Counter))
        bldg_list_dict = defaultdict(list)

        for bldg_id, opt_dict in zip(bldg_list, applied_options):
            for opt_para, value in opt_dict.items():
                opt = opt_para.split('|')[0]
                para = opt_para.split('|')[1]
                nested_dict[opt][para][value] += 1
                bldg_list_dict[opt].append(bldg_id)
                bldg_list_dict[opt + "|" + para].append(bldg_id)
                bldg_list_dict[opt + "|" + para + "<-" + value].append(bldg_id)

        def get_accord_item(opt_name):
            total_count = sum(counter.total() for counter in nested_dict[opt_name].values())
            children = []
            for parameter, counter in nested_dict[opt_name].items():
                contents = []
                new_counter = Counter({"All": counter.total()})
                new_counter.update(counter)
                for from_val, count in new_counter.items():
                    if from_val == "All":
                        but_ids = f"{opt_name}|{parameter}"
                    else:
                        but_ids = f"{opt_name}|{parameter}<-{from_val}"
                    entry = dbc.Row([dbc.Col(f"<-{from_val} ({count})", width="auto"),
                                     *get_action_button_pairs(but_ids, bldg_list_dict)])
                    contents.append(entry)
                children.append(dmc.AccordionItem(contents, label=f"{parameter} ({counter.total()})"))

            accordian = dmc.Accordion(children, multiple=True)
            first_row = dbc.Row([dbc.Col(f"All ({total_count})", width="auto"),
                                 *get_action_button_pairs(opt_name, bldg_list_dict)])
            return dmc.AccordionItem([first_row, accordian], label=f"{opt_name} ({total_count})")
        if reduced_set:
            final_report = dmc.Accordion([get_accord_item(opt_name) for opt_name in reduced_set], multiple=True)
        else:
            final_report = ["No option got applied to the selected building(s)."]
        up_name = upgrade2name[int(report_upgrade)]
        return f"Options applied in {up_name}", final_report, dict(bldg_list_dict)

    @app.callback(
        Output("enduse_report", "children"),
        Output("enduse_report_store", "data"),
        State('report_upgrade', 'value'),
        Input('input_building', "value"),
        Input('input_building', "options"),
        Input('input_building2', "options"),
        Input('input_enduse_type', 'value'),
        Input('chk-enduses', 'value'),
    )
    def show_enduse_report(report_upgrade, bldg_id, bldg_options, bldg_options2, enduse_change_type, chk_enduse):
        if not report_upgrade or not bldg_options:
            return ["Select an upgrade to see enuse report."], {}

        if dash.callback_context.triggered_id == 'input_building2' and "Enduses" not in chk_enduse:
            raise PreventUpdate()

        if "Enduses" in chk_enduse and bldg_options2:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options]

        # print(bldg_list)
        dict_changed_enduses = upgrade_run.report.get_enduses_buildings_map_by_change(upgrade_id=int(report_upgrade),
                                                                                      change_type=enduse_change_type,
                                                                                      bldg_list=bldg_list)
        # print(changed_enduses)

        all_changed_enduses = list(dict_changed_enduses.keys())
        if not all_changed_enduses:
            if bldg_id:
                return f'No enduse has {enduse_change_type} in building {bldg_id} ', {}
            else:
                return f'No enduse has {enduse_change_type} in any of the buildings', {}

        enduses2bldgs = defaultdict(list)
        for end_use, bldgs in dict_changed_enduses.items():
            if end_use in all_changed_enduses:
                enduses2bldgs[end_use].extend([int(bldg_id) for bldg_id in bldgs])

        fuel2bldgs = defaultdict(set)
        fuel2enduses = defaultdict(list)
        for enduse, bldgs in enduses2bldgs.items():
            for fuel in ['all_fuel'] + fuels_types:
                if fuel in enduse:
                    fuel2bldgs[fuel] |= set(bldgs)
                    fuel2enduses[fuel].append(enduse)
                    break
            else:
                fuel2bldgs['other'] |= set(bldgs)
                fuel2enduses['other'].append(enduse)

        for key, val in fuel2bldgs.items():
            fuel2bldgs[key] = list(val)

        enduses2bldgs.update(fuel2bldgs)

        def get_accord_item(fuel):
            total_count = len(enduses2bldgs[fuel])
            contents = [dbc.Row([dbc.Col(f"All {fuel} ({total_count})", width="auto"),
                                 *get_action_button_pairs(fuel, enduses2bldgs, "enduse")])]
            for enduse in fuel2enduses[fuel]:
                count = len(enduses2bldgs[enduse])
                row = dbc.Row([dbc.Col(f"{enduse} ({count})", width="auto"),
                               *get_action_button_pairs(enduse, enduses2bldgs, "enduse")])
                contents.append(row)
            return dmc.AccordionItem(contents, label=f"{fuel} ({total_count})")

        report = dmc.Accordion([get_accord_item(fuel) for fuel in fuel2enduses.keys()],  multiple=True)
        storedict = dict(enduses2bldgs)
        return report, storedict

    @app.callback(
        Output("char_report", "children"),
        Output("char_report_store", "data"),
        Input('input_building', "value"),
        Input('input_building', "options"),
        Input('input_building2', "options"),
        Input('drp-char-report', 'value'),
        Input('chk-chars', 'value'),
    )
    def show_char_report(bldg_id, bldg_options, bldg_options2, inp_char, chk_chars):
        if not (bldg_options or bldg_options2 or bldg_id):
            return [""], {}
        if not inp_char:
            return ["Select a characteristics to see its report."], {}

        if dash.callback_context.triggered_id == 'input_building2' and "Chars" not in chk_chars:
            raise PreventUpdate()

        if "Chars" in chk_chars and bldg_options2:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options]

        chars_df = res_csv_df.loc[bldg_list][inp_char].reset_index()
        char2bldgs = chars_df.groupby(inp_char)['building_id'].agg(list).to_dict()
        if (total_len := len(char2bldgs)) > 200:
            return [f"Sorry, this would create more than 200 ({total_len}) rows."], {}
        char_dict = {}
        total_count = 0
        contents = []
        for char_vals, bldglist in char2bldgs.items():
            but_ids = "+".join(char_vals) if isinstance(char_vals, tuple) else char_vals
            char_dict[but_ids] = [int(b) for b in bldglist]
            count = len(bldglist)
            total_count += count
            contents.append(dbc.Row([dbc.Col(f"{char_vals} ({count})", width="auto"),
                                     *get_action_button_pairs(but_ids, char_dict, "char")]))

        report = dmc.Accordion([dmc.AccordionItem(contents, label=f"{inp_char} ({total_count})")])
        return report, char_dict

    @app.callback(
        Output('graph', 'figure'),
        Output('loader_label', "children"),
        State('tab_view_type', "value"),
        Input('drp-group-by', 'value'),
        Input('radio_fuel', "value"),
        Input('dropdown_enduse', "value"),
        Input('radio_graph_type', "value"),
        Input('radio_savings', "value"),
        Input('chk_applied_only', "value"),
        Input('dropdown_chng_type', "value"),
        Input('check_all_points', "value"),
        Input('sync_upgrade', 'value'),
        Input('input_building', 'value'),
        Input('input_building', 'options'),
        Input('input_building2', 'options'),
        Input('chk-graph', 'value'),
        State("uirevision", "data"),
        State('report_upgrade', 'value')
    )
    def update_figure(view_tab, grp_by, fuel, enduse, graph_type, savings_type, chk_applied_only, chng_type,
                      show_all_points, sync_upgrade, selected_bldg, bldg_options, bldg_options2, chk_graph, uirevision,
                      report_upgrade):
        nonlocal download_csv_df
        if dash.callback_context.triggered_id == 'input_building2' and "Graph" not in chk_graph:
            raise PreventUpdate()

        if "Graph" in chk_graph and bldg_options2:
            bldg_options = bldg_options2

        bldg_options = bldg_options or []
        if not enduse:
            full_name = []
        if view_tab == 'energy':
            full_name = get_energy_db_cols(fuel, enduse)
        else:
            full_name = [enduse]

        applied_only = "Applied Only" in chk_applied_only

        if selected_bldg:
            filter_bldg = [int(selected_bldg)]
        else:
            filter_bldg = [int(b) for b in bldg_options]

        # print(f"Sync upgrade is {sync_upgrade}. {sync_upgrade is None}")
        if graph_type in ['Mean', 'Total', 'Count']:
            new_figure, report_df = get_bars(full_name, graph_type, savings_type, applied_only,
                                             chng_type, sync_upgrade, filter_bldg, grp_by)
        elif graph_type in ["Distribution"]:
            new_figure, report_df = get_distribution(full_name, savings_type, applied_only,
                                                     chng_type, 'Show all points' in show_all_points, sync_upgrade,
                                                     filter_bldg, grp_by, report_upgrade)
        elif graph_type in ["Scatter"]:
            new_figure, report_df = get_scatter(full_name, savings_type, applied_only,
                                                chng_type, 'Show all points' in show_all_points, sync_upgrade,
                                                filter_bldg, grp_by, report_upgrade)
        else:
            raise ValueError(f"Invalid graph type {graph_type}")
        uirevision = uirevision or "default"
        new_figure.update_layout(uirevision=uirevision)
        download_csv_df = report_df.reset_index(drop=True)
        return new_figure, ""

    upgrade_run.save_cache()
    return app


def main():
    print("Welcome to Upgrades Visualizer.")
    yaml_path = inquirer.text(message="Please enter path to the buildstock configuration yml file: ",
                              default="/Users/radhikar/Downloads/fact_sheets_category_6.yml").execute()
    opt_sat_path = inquirer.text(message="Please enter path to the options saturation csv file: ",
                                 default="/Users/radhikar/Downloads/options_saturations.csv").execute()
    db_name = inquirer.text(message="Please enter database_name "
                            "(found in postprocessing:aws:athena in the buildstock configuration file)",
                            default='largeee_test_runs').execute()
    table_name = inquirer.text(message="Please enter table name (same as output folder name; found under "
                               "output_directory in the buildstock configuration file)",
                               default="medium_run_baseline_20230622,medium_run_category_6_20230707"
                               ).execute()

    if ',' in table_name:
        table_name = table_name.split(',')
    app = get_app(yaml_path, opt_sat_path, db_name=db_name, table_name=table_name)
    app.run_server(debug=False, port=8005)


if __name__ == '__main__':
    main()
