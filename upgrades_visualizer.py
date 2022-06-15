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
from jupyter_dash import JupyterDash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash
import plotly.graph_objects as go

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
all_up_csvs = [res_csv_df]

upgrade2res = {0: res_csv_df}
for i in range(1, 11):
    print(f"Getting up_csv for {i}")
    up_csv = euss_athena.get_upgrades_csv(i)
    up_csv = up_csv.loc[res_csv_df.index]
    sub_build_df = build_df.loc[up_csv.index]
    up_csv = up_csv.join(build_df)
    up_csv = up_csv.rename(columns=lambda x: x.split('.')[1] if '.' in x else x)
    up_csv['upgrade'] = up_csv['upgrade'].map(lambda x: int(x))
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


def get_distribution(end_use, savings_type='', applied_only=False, change_type='any', show_all_points=False,
                     comp_upgrade=None, filter_bldg=None):
    fig = go.Figure()
    # base_df = up_csv_df.loc[0].set_index('building_id')[end_use].sum(axis=1, skipna=False)
    base_vals = upgrade2res[0][end_use].sum(axis=1, skipna=False)
#     building_ids = up_csv_df.loc[0]['building_id'].map(lambda x: f"Building: {x}")
    for upgrade in [0] + available_upgrades:
        # sub_df = up_csv_df.loc[upgrade].set_index('building_id')[end_use].sum(axis=1, skipna=False)
        if upgrade >= 1 and change_type != 'any':
            bldg_list = filter_bldg or chng2bldg[(upgrade, change_type)]
            res_df = upgrade2res[upgrade].loc[bldg_list]
            base_df = base_vals.loc[bldg_list]
        elif change_type == 'any':
            res_df = upgrade2res[upgrade]
            base_df = base_vals

        else:
            sync_bldg_list = filter_bldg or chng2bldg[(int(comp_upgrade), change_type)]
            res_df = upgrade2res[upgrade].loc[sync_bldg_list]
            base_df = base_vals.loc[sync_bldg_list]
        sub_df = res_df[end_use].sum(axis=1, skipna=False)
        if savings_type == 'Savings':
            sub_df = base_df - sub_df
            if not applied_only:
                sub_df = sub_df.fillna(0)
        elif savings_type == 'Percent Savings':
            sub_df = 100 * (base_df - sub_df) / base_df
            if not applied_only:
                sub_df = sub_df.fillna(0)
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
    pure_end_use_name = end_use[0].removeprefix("end_use_")
    pure_end_use_name = pure_end_use_name.removeprefix("fuel_use_")
    ylabel = end_use[0] if len(end_use) == 1 else f"All_fuel_{pure_end_use_name}"
    fig.update_layout(yaxis_title=ylabel,
                      title='Distribution by upgrade')
    return fig


def get_bars(end_use, value_type='mean', savings_type='', applied_only=False, change_type='any',
             comp_upgrade=None, filter_bldg=None):
    fig = go.Figure()
#     end_use = end_use or "fuel_use_electricity_total_m_btu"
    base_vals = upgrade2res[0][end_use].sum(axis=1, skipna=False)

    for upgrade in range(11):
        if upgrade >= 1 and change_type != 'any':
            bldg_list = filter_bldg or chng2bldg[(upgrade, change_type)]
            res_df = upgrade2res[upgrade].loc[bldg_list]
            base_df = base_vals.loc[bldg_list]
        elif change_type == 'any':
            res_df = upgrade2res[upgrade]
            base_df = base_vals

        else:
            sync_bldg_list = filter_bldg or chng2bldg[(int(comp_upgrade), change_type)]
            res_df = upgrade2res[upgrade].loc[sync_bldg_list]
            base_df = base_vals.loc[sync_bldg_list]
        if savings_type == 'Savings':
            up_vals = base_df - res_df[end_use].sum(axis=1, skipna=False)
        elif savings_type == 'Percent Savings':
            up_vals = 100 * (base_df - res_df[end_use].sum(axis=1, skipna=False)) / base_df
        else:
            up_vals = res_df[end_use].sum(axis=1, skipna=False)

        if not applied_only and savings_type:
            up_vals = up_vals.fillna(0)

        count = sum(up_vals < float('inf'))  # number of not na

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
    pure_end_use_name = end_use[0].removeprefix("end_use_")
    pure_end_use_name = pure_end_use_name.removeprefix("fuel_use_")
    ylabel = end_use[0] if len(end_use) == 1 else f"All_fuel_{pure_end_use_name}"
    fig.update_layout(yaxis_title=f"{ylabel}" + f"_{value_type}",
                      title=f'{value_type} value by upgrade')
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


app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([dbc.Container(html.Div([
    dbc.Row([dbc.Col(html.H1("Upgrades Visualizer"), width='auto'), dbc.Col(html.Sup("beta"))]),
    dbc.Row([dbc.Col(dbc.Label("Visualization Type: "), width='auto'),
             dbc.Col(radio_graph_type := dcc.RadioItems(["Mean", "Total", "Count", "Distribution"], "Mean",
                                                        id="radio_graph_type"), width='auto'),
             dbc.Col(dbc.Collapse(children=[check_all_points := dcc.Checklist(['Show all points'], [],
                                                                              inline=True, id='check_all_points')
                                            ],
                                  id="collapse_points", is_open=False), width='auto'),
             dbc.Col(dbc.Label("Value Type: "), width='auto'),
             dbc.Col(radio_savings := dcc.RadioItems(["Absolute", "Savings", "Percent Savings"], "Absolute",
                                                     id='radio_savings'), width='auto'),
             dbc.Col(dbc.Collapse(id="collapse_applied_only",
                                  children=[radio_applied_only := dcc.RadioItems(options=['All', 'Applied Only'],
                                                                                 value='All',
                                                                                 id='radio_applied'), ],
                                  is_open=False), width='auto')
             ]),
    dbc.Row([dbc.Col(html.Br())]),
    dbc.Row([dbc.Col(dcc.Loading(id='graph-loader', children=[loader_label := html.Div(id='loader_label')]))]),
    dbc.Row([dbc.Col(dcc.Graph(id='graph'))]),


])),
    dbc.Row([dbc.Col(
        tab_view_type := dcc.Tabs(id='tab_view_type', value='energy', children=[
            energy_tab := dcc.Tab(id='energy_tab', label='Energy', value='energy', children=[
                radio_fuel := dcc.RadioItems(fuels_types + ['All'], "electricity")]
            ),
            water_tab := dcc.Tab(label='Water Usage', value='water', children=[]
                                 ),
            load_tab := dcc.Tab(label='Load', value='load', children=[]
                                ),
            extra_tab := dcc.Tab(label='Peak', value='peak', children=[]
                                 ),
            extra_tab := dcc.Tab(label='Unmet Hours', value='unmet_hours', children=[]
                                 ),
            area_tab := dcc.Tab(label='Area', value='area', children=[]
                                ),
            size_tab := dcc.Tab(label='Size', value='size', children=[]
                                ),
            qoi_tab := dcc.Tab(label='QOI', value='qoi', children=[]
                               ),
            emissions_tab := dcc.Tab(label='emissions', value='emissions', children=[]
                                     ),
        ])
    )
    ]),
    dbc.Row([dbc.Col(dropdown_enduse := dcc.Dropdown(id='dropdown_enduses'))]),
    dbc.Row([
        dbc.Col(
            tab_view_filter := dcc.Tabs(id='tab_view_filter', value='change', children=[
                dcc.Tab(label='Change', value='change', children=[
                    dbc.Row([dbc.Col(radio_chng_type := dcc.RadioItems(change_types, "any",
                                                                       id='radio_chng_type'), width='auto'),
                             dbc.Collapse(id="collapse_focus_on", is_open=False, children=[
                                 dbc.Row([
                                     dbc.Col(html.Div("Basline Sync To: "), width='auto'),
                                     dbc.Col(dcc.Dropdown(id='comp_upgrade', value='1',
                                                          options=upgrade2name))
                                 ])
                             ]
                    )
                    ])
                ]
                ),
                dcc.Tab(label='Characteristics', value='char', children=[
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
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='upgrade_1', options=char_cols, value=None), width=3),
                        dbc.Col(html.Div(" "), width='auto'),
                        dbc.Col(dcc.Dropdown(id='upgrade_1_options'))
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='upgrade_2', options=char_cols, value=None), width=3),
                        dbc.Col(html.Div(" "), width='auto'),
                        dbc.Col(dcc.Dropdown(id='upgrade_2_options'))
                    ]),
                ]
                ),
                dcc.Tab(label='Building', value='building', children=[
                    dbc.Row([dbc.Col(html.Div("Building: "), width='auto'),
                             dbc.Col(dcc.Dropdown(id='input_building'), width=2),
                             dbc.Col(html.Div("Enduse Type: "), width='auto'),
                             dbc.Col(dcc.Dropdown(id='input_enduse_type', options=['changed', 'increased',
                                                                                   'decreased'], value='changed'),
                                     width=2),
                             dbc.Col(dcc.Checklist(id='chk_all_bldgs', options=[
                                 'All Buildings'], value=[], inline=True), width=2),
                             dbc.Col(dcc.Checklist(id='chk_intersect', options=[
                                 'Intersect'], value=[], inline=True), width=2)
                             ]),
                    dbc.Row([dbc.Col(dbc.Row([
                        dbc.Row([dbc.Col(html.Div(id="opt_applied_for_text")),
                                 ]),
                        dbc.Row([dbc.Col(dcc.Loading(id='opt-loader',
                                                     children=[html.Div(id="options_report")]))])
                    ])
                    ),
                        dbc.Col(dbc.Row([dbc.Col(
                            dcc.Loading(id='enduse_loader',
                                         children=[html.Div(id="enduse_report")])
                        )]
                        )
                    )

                    ]
                    )
                ]
                ),
            ]))
    ]),
    html.Div(id="status_bar"),
])


@app.callback(
    [Output(dropdown_enduse, "options"), Output(dropdown_enduse, "value")],
    [Input(tab_view_type, "value"), Input(radio_fuel, "value"), Input(dropdown_enduse, "value")]
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
    Output('collapse_applied_only', 'is_open'),
    [Input(radio_savings, "value"),
     ]
)
def disable_applied_only(savings):
    return savings.lower() == "savings"


@app.callback(
    Output("collapse_points", 'is_open'),
    [Input(radio_graph_type, "value")]
)
def disable_showpoints(graph_type):
    return graph_type.lower() == "distribution"


@app.callback(
    Output("collapse_focus_on", 'is_open'),
    [Input(radio_chng_type, "value")]
)
def disable_focus_on(chng_type):
    return chng_type.lower() != "any"
# Run app and display result inline in the notebook


@app.callback(
    [Output(tab_view_filter, 'value'),
     Output('input_building', 'value'),
     Output('input_building', 'options')
     ],
    [State(tab_view_filter, 'value'),
     State('input_building', 'value'),
     State('input_building', 'options'),
     Input('graph', "clickData"),
     Input(radio_chng_type, "value"),
     Input('comp_upgrade', 'value')],
    [State('drp_chr_1', 'value'),
     Input('drp_chr_1_choices', 'value'),
     State('drp_chr_2', 'value'),
     Input('drp_chr_2_choices', 'value'),
     State('drp_chr_3', 'value'),
     Input('drp_chr_3_choices', 'value'),
     ]
)
def bldg_selection(current_view, current_bldg, current_options,
                   click_data, change_type, comp_upgrade,
                   char1, char_val1, char2, char_val2, char3, char_val3
                   ):
    trigger_comp_id = dash.callback_context.triggered_id
    # print("Current", current_view, current_bldg, current_options)
    if trigger_comp_id == 'graph':
        if click_data and 'points' in click_data and len(click_data['points']) == 1:
            if match := re.search(r"Building: (\d*)", click_data['points'][0].get('hovertext', '')):
                if bldg := match.groups()[0]:
                    # print ("building", str(bldg), current_options)
                    return "building", str(bldg), current_options
        # print("Triggered from graph. Not valid point. Ignoring")
        raise PreventUpdate()
    else:
        if change_type == 'any':
            raise PreventUpdate()

        valid_bldgs = list(sorted(chng2bldg[(int(comp_upgrade), change_type)]))

        chars = [char1, char2, char3]
        char_vals = [char_val1, char_val2, char_val3]
        base_res = upgrade2res[0].loc[valid_bldgs]
        condition = np.ones(len(base_res), dtype=bool)
        for char, char_val in zip(chars, char_vals):
            if char and char_val:
                condition &= base_res[char] == char_val
        valid_bldgs = [str(b) for b in list(base_res[condition].index)]
        if valid_bldgs:
            bldg = current_bldg if current_bldg in valid_bldgs else valid_bldgs[0]
        else:
            bldg = ''
        # print("Updating differently")
        return current_view, str(bldg), valid_bldgs


def get_char_choices(char, bldg_options):
    if char:
        if bldg_options:
            bldg_options = [int(b) for b in bldg_options]
            res0 = upgrade2res[0].loc[bldg_options]
        else:
            res0 = upgrade2res[0]
        unique_choices = sorted(list(res0[char].unique()))
        return unique_choices, unique_choices[0]
    else:
        return [], None


@app.callback(
    [Output('drp_chr_1_choices', 'options'),
     Output('drp_chr_1_choices', 'value')],
    [Input('drp_chr_1', 'value'),
     State('input_building', "options")
     ]
)
def update_char_options1(char, bldg_options):
    return get_char_choices(char, bldg_options)


@app.callback(
    [Output('drp_chr_2_choices', 'options'),
     Output('drp_chr_2_choices', 'value')],
    [Input('drp_chr_2', 'value'),
     State('input_building', "options")
     ]
)
def update_char_options2(char, bldg_options):
    return get_char_choices(char, bldg_options)


@app.callback(
    [Output('drp_chr_3_choices', 'options'),
     Output('drp_chr_3_choices', 'value')],
    [Input('drp_chr_3', 'value'),
     State('input_building', "options")
     ]
)
def update_char_options3(char, bldg_options):
    return get_char_choices(char, bldg_options)


@app.callback(
    [Output("opt_applied_for_text", "children"),
     Output("options_report", 'children')],
    [Input('comp_upgrade', 'value'),
     Input('input_building', "value"),
     Input('input_building', "options"),
     Input('chk_intersect', 'value'),
     Input('chk_all_bldgs', 'value'),
     ],
)
def show_opt_report(cmp_upgrade, bldg_id, bldg_options, intersect, all_bldgs):
    if cmp_upgrade and ((('All Buildings' in all_bldgs) and bldg_options) or bldg_id):
        if 'All Buildings' in all_bldgs:
            bldg_list = [int(b) for b in bldg_options]
        else:
            bldg_list = [int(bldg_id)]

        applied_options = euss_athena.get_applied_options(int(cmp_upgrade), bldg_list, include_base_opt=True)
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

        final_list = []
        for option in reduced_set:
            final_list.append(f"{option} from {merged_options[option]}")

        up_name = upgrade2name[int(cmp_upgrade)]
        return f"Options applied in {up_name}", [html.Div(opt) for opt in final_list]
    else:

        raise PreventUpdate()


@app.callback(
    [Output("enduse_report", "children")],
    [Input('comp_upgrade', 'value'),
     Input('input_building', "value"),
     Input('input_building', "options"),
     Input('input_enduse_type', 'value'),
     Input('chk_intersect', 'value'),
     Input('chk_all_bldgs', 'value')
     ]
)
def show_enduse_report(cmp_upgrade, bldg_id, bldg_options, enduse_change_type, intersect, all_bldgs):
    if not cmp_upgrade or not bldg_id or not bldg_options:
        raise PreventUpdate()

    bldg_list = [int(b) for b in bldg_options] if 'All Buildings' in all_bldgs else [int(bldg_id)]

    # print(bldg_list)
    changed_enduses = euss_athena.get_enduses_by_change(int(cmp_upgrade), enduse_change_type, bldg_list)
    # print(changed_enduses)
    if 'Intersect' in intersect:
        changed_enduses = reduce(set.intersection, changed_enduses.values())
    else:
        changed_enduses = reduce(set.union, changed_enduses.values())

    if changed_enduses:
        return [', '.join(sorted(changed_enduses))]
    if 'Intersect' in intersect and 'All Buildings' in all_bldgs:
        return [f'No enduses has {enduse_change_type} in all buildings']
    else:
        return [f'No enduses has {enduse_change_type} in Building {bldg_id}']


@app.callback(
    [Output('graph', 'figure'),
     Output(loader_label, "children")],
    [State(tab_view_type, "value"),
     Input(radio_fuel, "value"),
     Input(dropdown_enduse, "value"),
     Input(radio_graph_type, "value"),
     Input(radio_savings, "value"),
     Input('radio_applied', "value"),
     Input(radio_chng_type, "value"),
     Input(check_all_points, "value"),
     Input('comp_upgrade', 'value'),
     Input(tab_view_filter, 'value'),
     Input('input_building', 'value'),
     Input('input_building', 'options')
     ]
)
def update_figure(view_tab, fuel, enduse, graph_type, savings_type, applied_only, chng_type,
                  show_all_points, comp_upgrade, filter_view, selected_bldg, bldg_options):
    if not enduse:
        raise PreventUpdate()
    if view_tab == 'energy':
        full_name = get_all_end_use_cols(fuel, enduse)
    else:
        full_name = [enduse]

    if filter_view == 'building' and selected_bldg:
        filter_bldg = [int(selected_bldg)]
    elif filter_view == 'building' and bldg_options:
        filter_bldg = [int(b) for b in bldg_options]
    else:
        filter_bldg = None
    if graph_type in ['Mean', 'Total', 'Count']:
        new_figure = get_bars(full_name, graph_type, savings_type, applied_only == 'Applied Only',
                              chng_type, comp_upgrade, filter_bldg)
    elif graph_type in ["Distribution"]:
        new_figure = get_distribution(full_name, savings_type, applied_only == 'Applied Only',
                                      chng_type, 'Show all points' in show_all_points, comp_upgrade,
                                      filter_bldg)
    new_figure.update_layout(uirevision="Same")
    return new_figure, ""


euss_athena.save_cache()


if __name__ == '__main__':
    app.run_server(mode='external', debug=False)
