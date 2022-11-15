"""
- - - - - - - - -
Upgrades Visualizer
Experimental Stage.
:author: Rajendra.Adhikari@nrel.gov
"""

from functools import reduce
from buildstock_query import BuildStockQuery
import numpy as np
import re
from collections import defaultdict, Counter
import dash_bootstrap_components as dbc
from dash import html, ALL, dcc
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
yaml_path = "EUSS-project-file-example.yml"
default_end_use = "fuel_use_electricity_total_m_btu"


def get_app(db_name: str = 'euss-tests', table_name: str = 'res_test_03_2018_10k_20220607', workgroup: str = 'eulp',
            buildstock_type: str = 'resstock'):
    euss_athena = BuildStockQuery(workgroup=workgroup,
                                  db_name=db_name,
                                  buildstock_type=buildstock_type,
                                  table_name=table_name,
                                  skip_reports=False)

    report = euss_athena.report.get_success_report()
    available_upgrades = list(report.index)
    available_upgrades.remove(0)
    euss_ua = euss_athena.get_upgrades_analyzer(yaml_path)
    upgrade2name = {indx+1: f"Upgrade {indx+1}: {upgrade['upgrade_name']}" for indx,
                    upgrade in enumerate(euss_ua.get_cfg().get('upgrades', []))}
    upgrade2shortname = {indx+1: f"Upgrade {indx+1}" for indx,
                         upgrade in enumerate(euss_ua.get_cfg().get('upgrades', []))}
    # allupgrade2name = {0: "Upgrade 0: Baseline"} | upgrade2name
    change_types = ["any", "no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng"]
    chng2bldg = {}
    for chng in change_types:
        for upgrade in available_upgrades:
            chng2bldg[(upgrade, chng)] = euss_athena.report.get_buildings_by_change(upgrade, chng)

    def get_cols(df, prefixes=[], suffixes=[]):
        cols = []
        for col in df.columns:
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

    res_csv_df = euss_athena.get_results_csv()
    euss_athena.save_cache()
    res_csv_df = res_csv_df[res_csv_df['completed_status'] == 'Success']
    sample_weight = res_csv_df['build_existing_model.sample_weight'].iloc[0]
    res_csv_df['upgrade'] = 0
    build_cols = [c for c in res_csv_df.columns if c.startswith('build_existing_model.')]
    build_df = res_csv_df[build_cols]

    res_csv_df = res_csv_df.rename(columns=lambda x: x.split('.')[1] if '.' in x else x)
    res_csv_df = res_csv_df.drop(columns=['applicable'])  # These are useless columns
    # all_up_csvs = [res_csv_df]

    upgrade2res = {0: res_csv_df}
    for upgrade in available_upgrades:
        print(f"Getting up_csv for {upgrade}")
        up_csv = euss_athena.get_upgrades_csv(upgrade)
        # print(list(up_csv.columns))
        # print(list(res_csv_df.columns))
        # print("upgrade", i, set(up_csv.columns)  - set(res_csv_df.columns))
        # print("upgrade", i, set(res_csv_df.columns)  - set(up_csv.columns))
        up_csv = up_csv.loc[res_csv_df.index]
        up_csv = up_csv.join(build_df)
        up_csv = up_csv.rename(columns=lambda x: x.split('.')[1] if '.' in x else x)
        up_csv = up_csv.drop(columns=['applicable'])
        up_csv['upgrade'] = up_csv['upgrade'].map(lambda x: int(x))
        invalid_rows_keys = up_csv['completed_status'] == 'Invalid'
        invalid_rows = up_csv[invalid_rows_keys].copy()
        invalid_rows.update(res_csv_df[invalid_rows_keys])
        invalid_rows['completed_status'] = 'Invalid'
        up_csv[invalid_rows_keys] = invalid_rows
        # up_csv = up_csv.reset_index().set_index(['upgrade'])
        upgrade2res[upgrade] = up_csv

    emissions_cols = get_cols(res_csv_df, suffixes=['_lb'])
    end_use_cols = get_cols(res_csv_df, ["end_use_", "energy_use__", "fuel_use_"])
    water_usage_cols = get_cols(res_csv_df, suffixes=["_gal"])
    load_cols = get_cols(res_csv_df, ["load_", "flow_rate_"])
    peak_cols = get_cols(res_csv_df, ["peak_"])
    unmet_cols = get_cols(res_csv_df, ["unmet_"])
    area_cols = get_cols(res_csv_df, suffixes=["_ft_2", ])
    size_cols = get_cols(res_csv_df, ["size_"])
    qoi_cols = get_cols(res_csv_df, ["qoi_"])
    char_cols = [c.removeprefix('build_existing_model.') for c in build_cols if 'applicable' not in c]
    fuels_types = ['electricity', 'natural_gas', 'propane', 'fuel_oil', 'coal', 'wood_cord', 'wood_pellets']

    def get_res(upgrade, applied_only=False):
        if upgrade == 0:
            return upgrade2res[0]
        elif applied_only:
            res = upgrade2res[int(upgrade)]
            res = res[res['completed_status'] != 'Invalid']
            return res
        else:
            return upgrade2res[int(upgrade)]

    def explode_str(input_str):
        input_str = str(input_str).lower()
        input_str = [
            int(x) if x and x[0] in "0123456789" else x
            for x in re.split(r"([\<\-])|([0-9]+)", input_str)
        ]
        return tuple("X" if x is None else x for x in input_str)

    def csv_generator(end_use, savings_type, applied_only, change_type, sync_upgrade, filter_bldg=None,
                      report_upgrade=None, group_cols=None):

        base_vals = get_res(0)[end_use].sum(axis=1, skipna=False)
        base_df = base_vals.loc[filter_bldg] if filter_bldg is not None else base_vals.copy()

        if group_cols:
            res_df = get_res(report_upgrade, applied_only)
            res_df = res_df.sort_values(group_cols, key=lambda series: [explode_str(x) for x in series])
            grouped_df = res_df.groupby(group_cols, sort=False)
            df_generator = ((', '.join(indx) if isinstance(indx, tuple) else indx, df) for (indx, df) in grouped_df)
            df_type = "groups"
        else:
            df_type = "upgrades"
            df_generator = ((f"Upgrade {upgrade}", get_res(upgrade, applied_only)) for upgrade in [report_upgrade])

        for indx, res_df in df_generator:
            if change_type:
                upgrade = indx if df_type == "upgrades" else report_upgrade
                chng_upgrade = int(sync_upgrade) if sync_upgrade else upgrade
                if chng_upgrade and chng_upgrade > 0:
                    change_bldg_list = chng2bldg[(chng_upgrade, change_type)]
                else:
                    change_bldg_list = []
                res_df = res_df.loc[res_df.index.intersection(change_bldg_list)]

            if filter_bldg is not None:
                res_df = res_df.loc[res_df.index.intersection(filter_bldg)]
            if len(res_df) == 0:
                continue

            sub_df = res_df[end_use].sum(axis=1, skipna=False)
            if savings_type == 'Savings':
                sub_df = base_df[sub_df.index] - sub_df
            elif savings_type == 'Percent Savings':
                sub_base_df = base_df[sub_df.index]
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

    def get_distribution(end_use, savings_type='', applied_only=False, change_type='', show_all_points=False,
                         sync_upgrade=None, filter_bldg=None, group_cols=None):
        fig = go.Figure()
        counter = 0
        for upgrade in [0] + available_upgrades:
            yvals = []
            xvals = []
            hovervals = []
            points = 'all' if show_all_points else 'suspectedoutliers'
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
                    hovertext = [f'{indx}<br> Building: {bid}<br>Sample Count: {count}'
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

        fig.update_layout(yaxis_title=f"{get_ylabel(end_use)}",
                          boxmode="group",
                          xaxis_title=", ".join(group_cols) if group_cols else 'Upgrade',
                          title=f"Distribution for {change_type} buildings" if change_type else 'Distribution',
                          clickmode='event+select')
        return fig

    def get_bars(end_use, value_type='mean', savings_type='', applied_only=False, change_type='',
                 sync_upgrade=None, filter_bldg=None, group_cols=None):
        fig = go.Figure()
        counter = 0
        for upgrade in [0] + available_upgrades:
            yvals = []
            xvals = []
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
                    break
                else:
                    yvals.append(val)
                    xvals.append(indx)
                    hovertext = f"Upgrade {upgrade}: {indx}<br> Average {val}. <br> Sample Count: {count}."
                    f"<br> Units Count: {count * sample_weight}."
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

        fig.update_layout(yaxis_title=f"{get_ylabel(end_use)}_{value_type}",
                          barmode='group',
                          xaxis_title=", ".join(group_cols) if group_cols else 'Upgrade',
                          title=f"{value_type} for {change_type} buildings" if change_type else f'{value_type}')

        return fig

    def get_end_use_cols(fuel):
        cols = []
        for c in end_use_cols:
            if fuel in c or fuel == 'All':
                c = c.removeprefix(f"end_use_{fuel}_")
                c = c.removeprefix(f"fuel_use_{fuel}_")
                if fuel == 'All':
                    for f in sorted(fuels_types):
                        c = c.removeprefix(f"end_use_{f}_")
                        c = c.removeprefix(f"fuel_use_{f}_")
                cols.append(c)
        no_dup_cols = {c: None for c in cols}
        return list(no_dup_cols.keys())

    def get_all_end_use_cols(fuel, end_use):
        all_enduses_set = set(end_use_cols)
        valid_cols = []
        prefix = "fuel_use" if end_use.startswith("total") else "end_use"
        if fuel == 'All':
            valid_cols.extend(f"{prefix}_{f}_{end_use}" for f in fuels_types
                              if f"{prefix}_{f}_{end_use}" in all_enduses_set)

        else:
            valid_cols.append(f"{prefix}_{fuel}_{end_use}")
        return valid_cols

    def get_opt_report(upgrade, bldg_id):
        applied_options = list(euss_athena.report.get_applied_options(upgrade, [bldg_id])[0])
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
        dbc.Row([dbc.Col(html.H1("Upgrades Visualizer"), width='auto'), dbc.Col(html.Sup("beta"))]),
        dbc.Row([dbc.Col(dbc.Label("Visualization Type: "), width='auto'),
                 dbc.Col(dcc.RadioItems(["Mean", "Total", "Count", "Distribution"], "Mean",
                                        id="radio_graph_type",
                                        labelClassName="pr-2"), width='auto'),
                #  dbc.Col(dbc.Collapse(children=[dcc.Checklist(['Show all points'], [],
                 #                                               inline=True, id='check_all_points')
                 #                                 ],
                 #                       id="collapse_points", is_open=True), width='auto'),
                 dbc.Col(children=[dcc.Checklist(['Show all points'], [],
                                                 inline=True, id='check_all_points')
                                   ],),
                 dbc.Col(dbc.Label("Value Type: "), width='auto'),
                 dbc.Col(dcc.RadioItems(["Absolute", "Savings", "Percent Savings"], "Absolute",
                                        id='radio_savings', labelClassName="pr-2"), width='auto'),
                 dbc.Col(dcc.Checklist(options=['Applied Only'], value=[],
                                       inline=True, id='chk_applied_only'), width='auto')
                 ]),
        dbc.Row([dbc.Col(html.Br())]),
        dbc.Row([dbc.Col(dcc.Loading(id='graph-loader', children=[html.Div(id='loader_label')]))]),
        dbc.Row([dbc.Col(dcc.Graph(id='graph'))]),


    ])),
        dbc.Row([dbc.Col(
            dcc.Tabs(id='tab_view_type', value='energy', children=[
                dcc.Tab(id='energy_tab', label='Energy', value='energy', children=[
                    dcc.RadioItems(fuels_types + ['All'], "electricity", id='radio_fuel',
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
            ])
        )
        ], className="mx-5 mt-5"),
        dbc.Row([dbc.Col(dcc.Dropdown(id='dropdown_enduse'))], className="mx-5 my-1"),
        dbc.Row([
            dbc.Col(
                dcc.Tabs(id='tab_view_filter', value='change', children=[
                    dcc.Tab(label='Change', value='change', children=[
                        dbc.Row([dbc.Col(html.Div("Restrict to buildings that have "), width='auto'),
                                 dbc.Col(dcc.Dropdown(change_types, "", placeholder="Select change type...",
                                                      id='dropdown_chng_type'), width='2'),
                                 dbc.Col(html.Div(" in "), width='auto'),
                                 dbc.Col(dcc.Dropdown(id='sync_upgrade', value='',
                                                      options={}))
                                 ],
                                className="flex items-center")
                    ]),
                    dcc.Tab(label='Characteristics', value='char', children=[
                        html.Div(" ... and have these characteristics"),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='drp_chr_1', options=char_cols, value=None), width=3),
                            dbc.Col(html.Div(" = "), width='auto'),
                            dbc.Col(dcc.Dropdown(id='drp_chr_1_choices'))
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='drp_chr_2', options=char_cols, value=None), width=3),
                            dbc.Col(html.Div(" = "), width='auto'),
                            dbc.Col(dcc.Dropdown(id='drp_chr_2_choices'))
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='drp_chr_3', options=char_cols, value=None), width=3),
                            dbc.Col(html.Div(" = "), width='auto'),
                            dbc.Col(dcc.Dropdown(id='drp_chr_3_choices'))
                        ])
                    ]
                    ),
                    dcc.Tab(label='Option', value='option', children=[
                        html.Div("... and which got applied these options "),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='upgrade_1', options=[], value=None, placeholder="coming soon"),
                                    width=3),
                            dbc.Col(html.Div(" "), width='auto'),
                            dbc.Col(dcc.Dropdown(id='upgrade_1_options', placeholder="coming soon"))
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='upgrade_2', options=[], value=None, placeholder="coming soon"),
                                    width=3),
                            dbc.Col(html.Div(" "), width='auto'),
                            dbc.Col(dcc.Dropdown(id='upgrade_2_options', placeholder="coming soon"))
                        ]),
                    ]
                    ),
                    dcc.Tab(label='Building', value='building', children=[
                        dbc.Row([dbc.Col(html.Div("Select:"), width='auto'),
                                 dbc.Col(dcc.Dropdown(id='input_building'), width=1),
                                 dbc.Col(html.Div(" in "), width='auto'),
                                 dbc.Col(dcc.Dropdown(id='report_upgrade', value='', placeholder="Upgrade ...",
                                                      options=upgrade2shortname), width=1),
                                 dbc.Col(html.Div("grouped by:"), width='auto'),
                                 dbc.Col(dcc.Dropdown(id='drp-group-by', options=char_cols, value=None,
                                                      multi=True, placeholder="Select characteristics..."),
                                         width=3),
                                 dbc.Col(dbc.Button("<= Copy", id="btn-copy", color="primary", size="sm",
                                                    outline=True), class_name="centered-col"),
                                 dbc.Col(html.Div("Extra restriction: "), width='auto'),
                                 dbc.Col(dcc.Dropdown(id='input_building2', disabled=False), width=2),
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
                                                              options=['changed', 'increased', 'decreased'],
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
                    ]
                    ),
                ]))
        ], className="mx-5 my-1"),
        html.Div(id="status_bar"),
        dcc.Download(id="download-chars-csv"),
        dcc.Store("uirevision")
        # dbc.Button("Kill me", id="button110")
    ])

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

    @app.callback(
        Output('dropdown_enduse', "options"),
        Output('dropdown_enduse', "value"),
        Input('tab_view_type', "value"),
        Input('radio_fuel', "value"),
        Input('dropdown_enduse', "value")
    )
    def update_enduse_options(view_tab, fuel_type, current_enduse):
        if view_tab == 'energy':
            available_endues = get_end_use_cols(fuel_type)
        elif view_tab == 'water':
            available_endues = water_usage_cols
        elif view_tab == 'load':
            available_endues = load_cols
        elif view_tab == 'peak':
            available_endues = peak_cols
        elif view_tab == 'unmet_hours':
            available_endues = unmet_cols
        elif view_tab == 'area':
            available_endues = area_cols
        elif view_tab == 'size':
            available_endues = size_cols
        elif view_tab == 'qoi':
            available_endues = qoi_cols
        elif view_tab == 'emissions':
            available_endues = emissions_cols
        else:
            raise ValueError(f"Invalid tab {view_tab}")

        enduse = current_enduse or available_endues[0]
        if fuel_type == 'All':
            return available_endues, enduse

        if enduse not in available_endues:
            # print(f"Bad enduse {enduse}")
            return sorted(available_endues), available_endues[0]
    #     print(fuel_type, f"Update enduse",  available_endues, enduse)
        return sorted(available_endues), enduse

    @app.callback(
        Output("collapse_points", 'is_open'),
        Input('radio_graph_type', "value")
    )
    def disable_showpoints(graph_type):
        print(f"Graph type: {graph_type.lower() == 'distribution'}")
        return True
        # return graph_type.lower() == "distribution"

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
        Output('tab_view_filter', 'value'),
        Output('input_building', 'value'),
        Output('input_building', 'options'),
        Output('report_upgrade', 'value'),
        Output('check_all_points', "value"),
        Input('graph', "selectedData"),
        State('input_building', 'options')
    )
    def graph_click(selection_data, current_options):
        if not selection_data or 'points' not in selection_data or len(selection_data['points']) < 1:
            raise PreventUpdate()

        selected_buildings = []
        selected_upgrades = []
        for point in selection_data['points']:
            if not (match := re.search(r"Building: (\d*)", point.get('hovertext', ''))):
                continue
            if bldg := match.groups()[0]:
                upgrade_match = re.search(r"Upgrade (\d*):", point.get('hovertext', ''))
                upgrade = upgrade_match.groups()[0] if upgrade_match else ''
                selected_buildings.append(bldg)
                selected_upgrades.append(upgrade)

        if not selected_buildings:
            raise PreventUpdate()

        if len(selected_buildings) != 1:
            selected_buildings = list(set(selected_buildings))
            return 'building', '', selected_buildings, selected_upgrades[0], ['Show all points']
        current_options = current_options or selected_buildings
        return 'building', selected_buildings[0], current_options, selected_upgrades[0], ['Show all points']

    @app.callback(
        Output('check_all_points', 'value'),
        Input('input_building', 'value'),
        State('input_building', 'options'),
        State('check_all_points', 'value'))
    def uncheck_all_points(bldg_selection, bldg_options, current_val):
        if not bldg_selection and bldg_options:
            return [''] if len(bldg_options) > 1000 else current_val
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
        Input('dropdown_chng_type', "value"),
        Input('sync_upgrade', 'value'),
        Input('report_upgrade', 'value'),
        State('drp_chr_1', 'value'),
        Input('drp_chr_1_choices', 'value'),
        State('drp_chr_2', 'value'),
        Input('drp_chr_2_choices', 'value'),
        State('drp_chr_3', 'value'),
        Input('drp_chr_3_choices', 'value'),
        Input('btn-reset', "n_clicks")
    )
    def bldg_selection(current_bldg, change_type, sync_upgrade, report_upgrade,
                       char1, char_val1, char2, char_val2, char3, char_val3, reset_click):

        if sync_upgrade and change_type:
            valid_bldgs = list(sorted(chng2bldg[(int(sync_upgrade), change_type)]))
        elif report_upgrade and change_type:
            valid_bldgs = list(sorted(chng2bldg[(int(report_upgrade), change_type)]))
            res = get_res(report_upgrade, applied_only=True)
            valid_bldgs = list(res.index.intersection(valid_bldgs))
        elif report_upgrade:
            res = get_res(report_upgrade, applied_only=True)
            valid_bldgs = list(res.index)
        else:
            valid_bldgs = list(upgrade2res[0].index)

        chars = [char1, char2, char3]
        char_vals = [char_val1, char_val2, char_val3]
        base_res = upgrade2res[0].loc[valid_bldgs]
        condition = np.ones(len(base_res), dtype=bool)
        for char, char_val in zip(chars, char_vals):
            if char and char_val:
                condition &= base_res[char] == char_val
        valid_bldgs = [str(b) for b in list(base_res[condition].index)]

        if current_bldg and current_bldg not in valid_bldgs:
            current_bldg = valid_bldgs[0] if valid_bldgs else ''
        return current_bldg, valid_bldgs, []

    def get_char_choices(char):
        if char:
            res0 = upgrade2res[0]
            unique_choices = sorted(list(res0[char].unique()))
            return unique_choices, unique_choices[0]
        else:
            return [], None

    @app.callback(
        Output('drp_chr_1_choices', 'options'),
        Output('drp_chr_1_choices', 'value'),
        Input('drp_chr_1', 'value'),
    )
    def update_char_options1(char):
        return get_char_choices(char)

    @app.callback(
        Output('drp_chr_2_choices', 'options'),
        Output('drp_chr_2_choices', 'value'),
        Input('drp_chr_2', 'value'),
    )
    def update_char_options2(char):
        return get_char_choices(char)

    @app.callback(
        Output('drp_chr_3_choices', 'options'),
        Output('drp_chr_3_choices', 'value'),
        Input('drp_chr_3', 'value'),
    )
    def update_char_options3(char):
        return get_char_choices(char)

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

                col = dbc.Col(html.Div([button, tooltip]),  width='auto', class_name="col-btn-cross")
            else:
                tooltip = dbc.Tooltip(f"Select all except these buildings.{bldg_str}",
                                      target=f"div-tooltip-target-{type}-{id}", delay={'show': 1000})
                col = dbc.Col(html.Div([button, tooltip]),  width='auto', class_name="col-btn-check")
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

        applied_options = euss_athena.report.get_applied_options(int(report_upgrade), bldg_list, include_base_opt=True)
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
        dict_changed_enduses = euss_athena.report.get_enduses_by_change(int(report_upgrade), enduse_change_type,
                                                                        bldg_list)
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
        State("uirevision", "data")
    )
    def update_figure(view_tab, grp_by, fuel, enduse, graph_type, savings_type, chk_applied_only, chng_type,
                      show_all_points, sync_upgrade, selected_bldg, bldg_options, bldg_options2, chk_graph, uirevision):

        if dash.callback_context.triggered_id == 'input_building2' and "Graph" not in chk_graph:
            raise PreventUpdate()

        if "Graph" in chk_graph and bldg_options2:
            bldg_options = bldg_options2

        bldg_options = bldg_options or []
        if not enduse:
            full_name = []
        if view_tab == 'energy':
            full_name = get_all_end_use_cols(fuel, enduse)
        else:
            full_name = [enduse]

        applied_only = "Applied Only" in chk_applied_only

        if selected_bldg:
            filter_bldg = [int(selected_bldg)]
        else:
            filter_bldg = [int(b) for b in bldg_options]

        # print(f"Sync upgrade is {sync_upgrade}. {sync_upgrade is None}")
        if graph_type in ['Mean', 'Total', 'Count']:
            new_figure = get_bars(full_name, graph_type, savings_type, applied_only,
                                  chng_type, sync_upgrade, filter_bldg, grp_by)
        elif graph_type in ["Distribution"]:
            new_figure = get_distribution(full_name, savings_type, applied_only,
                                          chng_type, 'Show all points' in show_all_points, sync_upgrade,
                                          filter_bldg, grp_by)
        uirevision = uirevision or "default"
        new_figure.update_layout(uirevision=uirevision)
        return new_figure, ""

    euss_athena.save_cache()
    return app


def main():
    print("Welcome to Upgrades Visualizer.")
    db_name = inquirer.text(message="Please enter database_name "
                            "(found in postprocessing:aws:athena in the buildstock configuration file)",
                            default='euss-tests').execute()
    table_name = inquirer.text(message="Please enter table name (same as output folder name; found under "
                               "output_directory in the buildstock configuration file)",
                               default='res_test_03_2018_10k_20220607').execute()
    app = get_app(db_name=db_name, table_name=table_name)
    app.run_server(debug=False)


if __name__ == '__main__':
    main()
