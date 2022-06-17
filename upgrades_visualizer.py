"""
- - - - - - - - -
Upgrades Visualizer
Experimental Stage.
:author: Rajendra.Adhikari@nrel.gov
"""

from functools import reduce
from eulpda.smart_query.resstock_savings import ResStockSavings
import numpy as np
import re
from collections import defaultdict, Counter
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from dash_extensions.enrich import MultiplexerTransform, DashProxy
# import os
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


# yaml_path = "/Users/radhikar/Documents/eulpda/EULP-data-analysis/notebooks/EUSS-project-file-example.yml"
yaml_path = "notebooks/EUSS-project-file-example.yml"
default_end_use = "fuel_use_electricity_total_m_btu"

euss_athena = ResStockSavings(workgroup='eulp',
                              db_name='euss-tests',
                              buildstock_type='resstock',
                              table_name='res_test_03_2018_10k_20220607',
                              skip_reports=False)

report = euss_athena.get_success_report()
available_upgrades = list(report.index)
available_upgrades.remove(0)
euss_ua = euss_athena.get_upgrades_analyzer(yaml_path)
upgrade2name = {indx+1: f"Upgrade {indx+1}: {upgrade['upgrade_name']}" for indx,
                upgrade in enumerate(euss_ua.get_cfg()['upgrades'])}
allupgrade2name = {0: "Upgrade 0: Baseline"} | upgrade2name
change_types = ["any", "no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng"]
chng2bldg = {}
for chng in change_types:
    for upgrade in available_upgrades:
        chng2bldg[(upgrade, chng)] = euss_athena.get_buildings_by_change(upgrade, chng)


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
res_csv_df = res_csv_df[res_csv_df['completed_status'] == 'Success']
sample_weight = res_csv_df['build_existing_model.sample_weight'].iloc[0]
res_csv_df['upgrade'] = 0
build_cols = [c for c in res_csv_df.columns if c.startswith('build_existing_model.')]
build_df = res_csv_df[build_cols]

res_csv_df = res_csv_df.rename(columns=lambda x: x.split('.')[1] if '.' in x else x)
res_csv_df = res_csv_df.drop(columns=['applicable'])  # These are useless columns
all_up_csvs = [res_csv_df]

upgrade2res = {0: res_csv_df}
for i in range(1, 11):
    print(f"Getting up_csv for {i}")
    up_csv = euss_athena.get_upgrades_csv(i)
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
    upgrade2res[i] = up_csv

emissions_cols = get_cols(res_csv_df, suffixes=['_lb'])
end_use_cols = get_cols(res_csv_df, ["end_use_", "energy_use__", "fuel_use_"])
water_usage_cols = get_cols(res_csv_df, suffixes=["_gal"])
load_cols = get_cols(res_csv_df, ["load_", "flow_rate_"])
peak_cols = get_cols(res_csv_df, ["peak_"])
unmet_cols = get_cols(res_csv_df, ["unmet_"])
area_cols = get_cols(res_csv_df, suffixes=["_ft_2", ])
size_cols = get_cols(res_csv_df, ["size_"])
qoi_cols = get_cols(res_csv_df, ["qoi_"])
char_cols = [c.removeprefix('build_existing_model.') for c in build_cols]
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


def upgrade_csv_generator(end_use, savings_type, applied_only, change_type, sync_upgrade, filter_bldg=None):
    base_vals = get_res(0)[end_use].sum(axis=1, skipna=False)
#     building_ids = up_csv_df.loc[0]['building_id'].map(lambda x: f"Building: {x}")
    base_df = base_vals.loc[filter_bldg] if filter_bldg is not None else base_vals.copy()
    upgrade_list = list(available_upgrades)
    if savings_type not in ['Savings', 'Percent Savings'] and not change_type:
        upgrade_list.insert(0, 0)

    for upgrade in upgrade_list:
        # sub_df = up_csv_df.loc[upgrade].set_index('building_id')[end_use].sum(axis=1, skipna=False)
        res_df = get_res(upgrade, applied_only)
        # print(f"res_df for {upgrade} is {len(res_df)}")
        if change_type:
            chng_upgrade = int(sync_upgrade) if sync_upgrade else upgrade
            change_bldg_list = chng2bldg[(chng_upgrade, change_type)]
            res_df = res_df.loc[res_df.index.intersection(change_bldg_list)]
            # print(f"res_df for {upgrade} downselected to {len(res_df)} due to chng of {len(change_bldg_list)}")

        if filter_bldg is not None:
            res_df = res_df.loc[res_df.index.intersection(filter_bldg)]
            # print(f"Futher Down selected to {len(res_df)} due to {len(filter_bldg)}")

        sub_df = res_df[end_use].sum(axis=1, skipna=False)
        if savings_type == 'Savings':
            sub_df = base_df[sub_df.index] - sub_df
        elif savings_type == 'Percent Savings':
            sub_df = 100 * (base_df[sub_df.index] - sub_df) / base_df[sub_df.index]
        yield upgrade, sub_df


def get_ylabel(end_use):
    if len(end_use) == 1:
        return end_use[0]
    pure_end_use_name = end_use[0].removeprefix("end_use_")
    pure_end_use_name = pure_end_use_name.removeprefix("fuel_use_")
    pure_end_use_name = "_".join(pure_end_use_name.split("_")[1:])
    return f"{len(end_use)}_fuels_{pure_end_use_name}"


def get_distribution(end_use, savings_type='', applied_only=False, change_type='any', show_all_points=False,
                     sync_upgrade=None, filter_bldg=None):
    # print("Get dist got:", end_use, savings_type, applied_only, change_type, show_all_points,
    # sync_upgrade, len(filter_bldg) if filter_bldg else 'None')
    fig = go.Figure()
    # base_df = up_csv_df.loc[0].set_index('building_id')[end_use].sum(axis=1, skipna=False)
    for upgrade, sub_df in upgrade_csv_generator(end_use, savings_type, applied_only, change_type, sync_upgrade,
                                                 filter_bldg):
        building_ids = list(sub_df.index)
        count = sum(sub_df < float('inf'))
        points = 'all' if show_all_points else 'suspectedoutliers'
        fig.add_trace(go.Box(
            y=sub_df,
            name=f'Upgrade {upgrade}',
            boxpoints=points,
            boxmean=True,  # represent mean
            hovertext=[f'{allupgrade2name[upgrade]}<br> Building: {bid}<br>Sample Count: {count}'
                       for bid in building_ids],
            hoverinfo="all"
        ))
    fig.update_layout(yaxis_title=f"{get_ylabel(end_use)}",
                      title='Distribution')
    return fig


def get_bars(end_use, value_type='mean', savings_type='', applied_only=False, change_type='any',
             sync_upgrade=None, filter_bldg=None):
    fig = go.Figure()
#     end_use = end_use or "fuel_use_electricity_total_m_btu"
    for upgrade, up_vals in upgrade_csv_generator(end_use, savings_type, applied_only, change_type, sync_upgrade,
                                                  filter_bldg):
        count = len(up_vals)
        if value_type.lower() == 'total':
            val = up_vals.sum() * sample_weight
        elif value_type.lower() == 'count':
            val = up_vals.count()
        else:
            val = up_vals.mean()
        fig.add_trace(go.Bar(
            y=[val],
            x=[f"Upgrade {upgrade}"],
            name=f'Upgrade {upgrade}',
            hovertext=f"{allupgrade2name[upgrade]}<br> Average {val}. <br> Sample Count: {count}."
            f"<br> Units Count: {count * sample_weight}.",
            hoverinfo="all"
        ))

    fig.update_layout(yaxis_title=f"{get_ylabel(end_use)}_{value_type}", title=f'{value_type}')

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
    applied_options = list(euss_athena.get_applied_options(upgrade, [bldg_id])[0])
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


app = DashProxy(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], transforms=[MultiplexerTransform()])
app.layout = html.Div([dbc.Container(html.Div([
    dbc.Row([dbc.Col(html.H1("Upgrades Visualizer"), width='auto'), dbc.Col(html.Sup("beta"))]),
    dbc.Row([dbc.Col(dbc.Label("Visualization Type: "), width='auto'),
             dbc.Col(dcc.RadioItems(["Mean", "Total", "Count", "Distribution"], "Mean",
                                    id="radio_graph_type"), width='auto'),
             dbc.Col(dbc.Collapse(children=[dcc.Checklist(['Show all points'], [],
                                                          inline=True, id='check_all_points')
                                            ],
                                  id="collapse_points", is_open=False), width='auto'),
             dbc.Col(dbc.Label("Value Type: "), width='auto'),
             dbc.Col(dcc.RadioItems(["Absolute", "Savings", "Percent Savings"], "Absolute",
                                    id='radio_savings'), width='auto'),
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
                dcc.RadioItems(fuels_types + ['All'], "electricity", id='radio_fuel')]
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
    ]),
    dbc.Row([dbc.Col(dcc.Dropdown(id='dropdown_enduse'))]),
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
                             ]
                            )
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
                    dbc.Row([dbc.Col(html.Div("Further restrict to building:"), width='auto'),
                             dbc.Col(dcc.Dropdown(id='input_building'), width=2),
                             dbc.Col(html.Div(" in "), width='auto'),
                             dbc.Col(dcc.Dropdown(id='report_upgrade', value='',
                                                  options=upgrade2name), width=4),
                             dbc.Col(html.Div("View enduse that: "), width='auto'),
                             dbc.Col(dcc.Dropdown(id='input_enduse_type', options=['changed', 'increased',
                                                                                   'decreased'], value='changed'),
                                     width=1),
                             dbc.Col(dcc.Checklist(id='chk_intersect', options=[
                                 'Intersect'], value=[], inline=True), width=2)
                             ]),
                    dbc.Row([dbc.Col([
                        dbc.Row(html.Div(id="options_report_header")),
                        dbc.Row(dcc.Loading(id='opt-loader', children=[html.Div(id="options_report")]))
                    ]),
                        dbc.Col([
                            dbc.Row(html.Div(id='enduse_report_header')),
                            dbc.Row(dcc.Loading(id='enduse_loader', children=html.Div(id="enduse_report")))
                        ])

                    ])
                ]
                ),
            ]))
    ]),
    html.Div(id="status_bar"),
])


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
    return graph_type.lower() == "distribution"


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
    return f"{len(options)} Buidlings" if options else "No buildings available."


@app.callback(
    Output('tab_view_filter', 'value'),
    Output('input_building', 'value'),
    Output('input_building', 'options'),
    Output('report_upgrade', 'value'),
    Output('check_all_points', "value"),
    Input('graph', "clickData"),
    State('input_building', 'options')
)
def graph_click(click_data, current_options):
    if not click_data or 'points' not in click_data or len(click_data['points']) != 1:
        raise PreventUpdate()

    if not (match := re.search(r"Building: (\d*)", click_data['points'][0].get('hovertext', ''))):
        raise PreventUpdate()

    if bldg := match.groups()[0]:
        current_options = current_options or [bldg]
        upgrade_match = re.search(r"Upgrade (\d*):", click_data['points'][0].get('hovertext', ''))
        upgrade = upgrade_match.groups()[0] if upgrade_match else ''
        return 'building', bldg, current_options, upgrade, ['Show all points']
    raise PreventUpdate()


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
    Output('input_building', 'value'),
    Output('input_building', 'options'),
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
)
def bldg_selection(current_bldg, change_type, sync_upgrade, report_upgrade,
                   char1, char_val1, char2, char_val2, char3, char_val3):
    if sync_upgrade and change_type:
        valid_bldgs = list(sorted(chng2bldg[(int(sync_upgrade), change_type)]))
    elif report_upgrade and change_type:
        valid_bldgs = list(sorted(chng2bldg[(int(report_upgrade), change_type)]))
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
    return current_bldg, valid_bldgs


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


@app.callback(
    Output("options_report_header", "children"),
    Output("options_report", 'children'),
    Input('input_building', "value"),
    Input('input_building', "options"),
    State('report_upgrade', 'value'),
    Input('chk_intersect', 'value'),
)
def show_opt_report(bldg_id, bldg_options, report_upgrade, intersect):
    if not report_upgrade or not bldg_options:
        return "Select an upgrade to see options applied in that upgrade", [""]

    bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options]

    applied_options = euss_athena.get_applied_options(int(report_upgrade), bldg_list, include_base_opt=True)
    opt_only = [set(opt.keys()) for opt in applied_options]
    if 'Intersect' in intersect:
        reduced_set = list(reduce(set.intersection, opt_only))
    else:
        reduced_set = list(reduce(set.union, opt_only))

    merged_options = defaultdict(list)
    for opt_dict in applied_options:
        for key, value in opt_dict.items():
            merged_options[key].append(value)
    for key, val in merged_options.items():
        if len(val) > 1:
            merged_options[key] = Counter(val)

    final_list = [f"{option} from {merged_options[option]}" for option in reduced_set]
    final_list = final_list or ["No option is applied to the building(s)"]
    up_name = upgrade2name[int(report_upgrade)]
    return f"Options applied in {up_name}", [html.Div(opt) for opt in final_list]


@app.callback(
    Output("enduse_report_header", "children"),
    Output("enduse_report", "children"),
    State('report_upgrade', 'value'),
    Input('input_building', "value"),
    Input('input_building', "options"),
    Input('input_enduse_type', 'value'),
    Input('chk_intersect', 'value'),
)
def show_enduse_report(report_upgrade, bldg_id, bldg_options, enduse_change_type, intersect):
    if not report_upgrade or not bldg_options:
        return "Select an upgrade to see enduses that changed in that upgrade", ""

    bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options]

    # print(bldg_list)
    changed_enduses = euss_athena.get_enduses_by_change(int(report_upgrade), enduse_change_type, bldg_list)
    # print(changed_enduses)
    if 'Intersect' in intersect:
        changed_enduses = reduce(set.intersection, changed_enduses.values())
    else:
        changed_enduses = reduce(set.union, changed_enduses.values())

    if changed_enduses:
        return f"The following enduses {enduse_change_type}.", [', '.join(sorted(changed_enduses))]
    if 'Intersect' in intersect:
        return f'No enduses has {enduse_change_type} in all buildings', []
    else:
        return f'No enduses has {enduse_change_type} in Building {bldg_id}', []


@app.callback(
    Output('graph', 'figure'),
    Output('loader_label', "children"),
    State('tab_view_type', "value"),
    Input('radio_fuel', "value"),
    Input('dropdown_enduse', "value"),
    Input('radio_graph_type', "value"),
    Input('radio_savings', "value"),
    Input('chk_applied_only', "value"),
    Input('dropdown_chng_type', "value"),
    Input('check_all_points', "value"),
    Input('sync_upgrade', 'value'),
    Input('input_building', 'value'),
    State('input_building', 'options')
)
def update_figure(view_tab, fuel, enduse, graph_type, savings_type, chk_applied_only, chng_type,
                  show_all_points, sync_upgrade, selected_bldg, bldg_options):
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
                              chng_type, sync_upgrade, filter_bldg)
    elif graph_type in ["Distribution"]:
        new_figure = get_distribution(full_name, savings_type, applied_only,
                                      chng_type, 'Show all points' in show_all_points, sync_upgrade,
                                      filter_bldg)
    new_figure.update_layout(uirevision="Same")
    return new_figure, ""


euss_athena.save_cache()


if __name__ == '__main__':
    app.run_server(debug=False)
