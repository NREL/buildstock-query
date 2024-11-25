from functools import reduce
import re
import os
os.environ['POLARS_MAX_THREADS'] = '4'
from collections import defaultdict, Counter
import dash_bootstrap_components as dbc
from dash import html, ALL, dcc, ctx, Dash, dcc, html, Input, Output, ALL, MATCH, Patch, callback
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
from InquirerPy import inquirer
from buildstock_query.tools.visualizer.viz_data import VizData
from buildstock_query.tools.visualizer.plot_utils import PlotParams, ValueTypes, SavingsTypes
from buildstock_query.tools.visualizer.figure import UpgradesPlot
from buildstock_query.helpers import load_script_defaults, save_script_defaults
import polars as pl
import pandas as pd
from typing import Literal
from buildstock_query.tools.visualizer.viz_utils import filter_cols, get_viz_data
import json
from sqlalchemy.sql import sqltypes
import datetime
import time
import diskcache
cache = diskcache.Cache("./diskcache")
from dash import Dash, DiskcacheManager, CeleryManager, Input, Output, html, callback, set_props

background_callback_manager = DiskcacheManager(cache)

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
DAYS = [str(i) for i in range(1, 32)]

# Add these styles at the top after imports
CARD_STYLE = {
    'padding': '20px',
    'margin': '10px 0',
    'border-radius': '8px',
    'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
    'background-color': 'white'
}

SECTION_STYLE = {
    'margin-bottom': '20px'
}

def get_app(viz_data: VizData):
    """Creates and returns the Dash application.
    
    Args:
        timeseries_data: Data source for timeseries values
        char_data: Data source for building characteristics
    """
    app = Dash(__name__, 
                   external_stylesheets=[dbc.themes.BOOTSTRAP],
                   )
    viz_data.init_metadata_df()
    timeseries_columns = [col.name for col in viz_data.main_run.ts_table.columns 
                          ]
    build_cols = viz_data.metadata_df.columns
    char_cols = [c.removeprefix(viz_data.main_run._char_prefix) for c in build_cols if 'applicable' not in c]
    app.layout = dbc.Container([
        # Header with spinner in the middle
        dbc.Row([
            dbc.Col(html.H2("Timeseries Visualizer", className="display-20"), width=4),
            dbc.Col(
                dcc.Loading(
                    id="loading-spinner",
                    children=html.Div(id="loading-output"),
                    fullscreen=False,
                ), 
                width=2,
                className="d-flex justify-content-center align-items-center"
            ),
            dbc.Col(html.Plaintext(id='update-status', className="text-muted"), width=2),
            dbc.Col(
                html.Div(
                    id='cost-display',
                    className="text-muted",
                    style={'textAlign': 'right', 'color': '#999'}
                ),
                width=4
            ),
        ], className="mb-0 align-items-center"),        # Main Graph with card styling
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(id='main-graph')
            ])
        ], style=CARD_STYLE),

        # Controls Section
        dbc.Card([
            dbc.CardBody([
                # Top row - Plot Controls with Update button
                dbc.Row([
                    dbc.Col([
                        html.Label("Upgrade:", className="mb-1", style={'fontSize': '14px'}),
                        dcc.Dropdown(
                            id='upgrade-dropdown',
                            options=list(viz_data.upgrade2shortname.keys()),
                            value=list(viz_data.upgrade2shortname.keys())[0] if viz_data.upgrade2shortname else None,
                            multi=True,
                            placeholder="Which Upgrade to Plot"
                        )
                    ], width=2),
                    dbc.Col([
                        html.Label("Columns to plot:", className="mb-1", style={'fontSize': '14px'}),
                        dcc.Dropdown(
                            id='plot-dropdown',
                            options=timeseries_columns,
                            value=timeseries_columns[20] if timeseries_columns else None,
                            multi=True,
                            placeholder="Select metrics to plot"
                        )
                    ], width=True),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button(
                                "Update",
                                id="update-plot-button",
                                color="primary",
                                className="w-100",
                                style={'minWidth': '120px'}
                            ), width="auto"),
                            dbc.Col(dbc.Button(
                                "Cancel",
                                id="cancel-button",
                                color="danger",
                                className="w-100",
                                style={'minWidth': '120px'},
                                disabled=True
                            ), width="auto")
                        ], className="g-2 h-100 align-items-end justify-content-end")
                    ], width="auto", className="d-flex align-items-end")
                ], className="mb-3 g-2 align-items-end"),

                # New split layout for aggregation controls
                dbc.Row([
                    # Left Column - Building Aggregation
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Aggregate across buildings", className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Aggregation type:", className="mb-1", style={'fontSize': '14px'}),
                                        dcc.Dropdown(
                                            id='building-agg-dropdown',
                                            options=[
                                                {'label': x.capitalize(), 'value': x} 
                                                for x in ['sum', 'avg', 'max', 'min']
                                            ],
                                            value=['avg'],
                                            multi=True
                                        )
                                    ])
                                ], className="mb-3"),
                                
                                # Filters section
                                html.Label("Filters:", className="mb-1", style={'fontSize': '14px'}),
                                html.Div(id='restrictions-container', children=[], className="mb-2"),
                                dbc.Button(
                                    [DashIconify(icon="mdi:plus", className="me-1"), "Add Filter"],
                                    id="add-restriction-btn",
                                    color="primary",
                                    size="sm"
                                )
                            ])
                        ], className="h-100")
                    ], width=6),

                    # Right Column - Time Aggregation
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Aggregate across time", className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Shape type:", className="mb-1", style={'fontSize': '14px'}),
                                        dcc.Dropdown(
                                            id='shape-type-dropdown',
                                            options=[
                                                {'label': "Daily Shape", "value": "daily_shape"},
                                                {'label': "Weekend Shape", "value": "weekend_shape"},
                                                {'label': "Weekday Shape", "value": "weekday_shape"}
                                            ],
                                            value=[],
                                            multi=True,
                                            placeholder="Full year (no aggregation)"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Aggregation type:", className="mb-1", style={'fontSize': '14px'}),
                                        dcc.Dropdown(
                                            id='time-agg-dropdown',
                                            options=[
                                                {'label': x.capitalize(), 'value': x} 
                                                for x in ['avg', 'max', 'min']
                                            ],
                                            value=['avg'],
                                            multi=True
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Resolution:", className="mb-1", style={'fontSize': '14px'}),
                                        dcc.Dropdown(
                                            id='resolution-dropdown',
                                            options=[],  # Will be populated by callback
                                            value=None
                                        )
                                    ], width=4)
                                ], className="mb-3"),
                                
                                # Time range controls
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Start:", className="mb-1", style={'fontSize': '14px'}),
                                        dbc.Row([
                                            dbc.Col(dcc.Dropdown(
                                                id='start-month-dropdown',
                                                options=[{'label': m, 'value': m} for m in MONTHS],
                                                value='Jan'
                                            ), width=6),
                                            dbc.Col(dcc.Dropdown(
                                                id='start-day-dropdown',
                                                options=[{'label': d, 'value': d} for d in DAYS],
                                                value='1'
                                            ), width=6)
                                        ], className="g-1")
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("End:", className="mb-1", style={'fontSize': '14px'}),
                                        dbc.Row([
                                            dbc.Col(dcc.Dropdown(
                                                id='end-month-dropdown',
                                                options=[{'label': m, 'value': m} for m in MONTHS],
                                                value='Dec'
                                            ), width=6),
                                            dbc.Col(dcc.Dropdown(
                                                id='end-day-dropdown',
                                                options=[{'label': d, 'value': d} for d in DAYS],
                                                value='31'
                                            ), width=6)
                                        ], className="g-1")
                                    ], width=6)
                                ], className="g-2")
                            ])
                        ], className="h-100")
                    ], width=6)
                ])
            ], className="p-3")
        ], style=CARD_STYLE),
        # Keep the restrictions store
        dcc.Store(id='restrictions-store', data={'count': 0}),
        dcc.Store(id='cost-store', data={'dollars': 0, 'gb': 0}),

    ], fluid=True, className="py-4", style={'background-color': '#f8f9fa'})

    @app.callback(
    Output('restrictions-container', 'children'),
    Output('restrictions-store', 'data'),
    Input('add-restriction-btn', 'n_clicks'),
    Input({'type': 'remove-restriction', 'index': ALL}, 'n_clicks'),
    State('restrictions-store', 'data'),
    State('restrictions-container', 'children'),
    prevent_initial_call=True
    )
    def manage_restrictions(add_clicks, remove_clicks, store_data, current_children):
        if not dash.callback_context.triggered:
            raise PreventUpdate
        triggered_id = dash.callback_context.triggered[0]['prop_id']
        # Initialize current_children and store_data if None
        if current_children is None:
            current_children = []
        if store_data is None:
            store_data = {'count': 0}

        if triggered_id == 'add-restriction-btn.n_clicks':
            new_index = store_data['count']
            store_data['count'] += 1
            new_restriction = get_restriction_row(new_index)
            current_children.append(new_restriction)

        elif 'remove-restriction' in triggered_id:
            triggered_index = json.loads(triggered_id.split('.')[0])['index']
            current_children = [
                child for child in current_children
                if child['props']['id'].get('index') != triggered_index
            ]

        return current_children, store_data

    def get_restriction_row(new_index):
        new_restriction = dbc.Row(
                id={'type': 'restriction-row', 'index': new_index},
                children=[
                dbc.Col(
                    [
                    dcc.Dropdown(
                        id={'type': 'char-name', 'index': new_index},
                        options=char_cols,
                        placeholder="Select characteristic"
                    )
                ], width=5),
                dbc.Col([
                    dcc.Dropdown(
                        id={'type': 'char-value', 'index': new_index},
                        multi=True,
                        placeholder="Select values"
                    )
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        DashIconify(icon="mdi:close"),
                        id={'type': 'remove-restriction', 'index': new_index},
                        color="danger",
                        size="sm",
                        style={'border-radius': '50%'}
                    )
                ], width=1, className="d-flex align-items-center")
            ], className="mb-2 g-2")
        
        return new_restriction


    @app.callback(
        Output({'type': 'char-value', 'index': ALL}, 'options'),
        Output({'type': 'char-value', 'index': ALL}, 'value'),
        Input({'type': 'char-name', 'index': ALL}, 'value'),
        Input({'type': 'char-value', 'index': ALL}, 'value'),
        State({'type': 'char-value', 'index': ALL}, 'id'),
        prevent_initial_call=False
    )
    def update_char_values(all_char_names, all_char_values, all_ids):
        """Update characteristic value options and selections based on filter chain."""
        if not is_valid_input(all_ids, all_char_names, viz_data.metadata_df):
            return [[]] * len(all_ids), [None] * len(all_ids)
        
        try:
            return process_all_rows(all_char_names, all_char_values, all_ids, viz_data.metadata_df)
        except Exception as e:
            print(f"Unexpected error in update_char_values: {str(e)}")
            return [[]] * len(all_ids), [None] * len(all_ids)

    def is_valid_input(all_ids, all_char_names, metadata_df):
        """Validate input parameters and data availability."""
        if not all_ids:
            return False
        if not all_char_names:
            return False
        if metadata_df is None or metadata_df.is_empty():
            print("Warning: Metadata DataFrame is empty or not initialized")
            return False
        return True

    def process_all_rows(all_char_names, all_char_values, all_ids, metadata_df):
        """Process each filter row to generate options and validate values."""
        all_options = []
        new_values = []
        
        for i, (current_char_name, current_values, _) in enumerate(zip(all_char_names, all_char_values, all_ids)):
            if not current_char_name:
                all_options.append([])
                new_values.append(None)
                continue
            
            try:
                filtered_df = apply_preceding_filters(metadata_df, all_char_names, all_char_values, i)
                if filtered_df.is_empty():
                    all_options.append([])
                    new_values.append(None)
                    continue
                
                options, new_value = process_single_row(filtered_df, current_char_name, current_values)
                all_options.append(options)
                new_values.append(new_value)
                
            except Exception as e:
                print(f"Error processing row {i}: {str(e)}")
                all_options.append([])
                new_values.append(None)
        
        return all_options, new_values

    def apply_preceding_filters(df, all_char_names, all_char_values, current_index):
        """Apply filters from rows preceding the current row."""
        filtered_df = df
        for j, (char_name, char_values) in enumerate(zip(all_char_names, all_char_values)):
            if (char_name and char_values and j < current_index and char_name in filtered_df.columns):
                filtered_df = filtered_df.filter(
                    pl.col(char_name).cast(pl.Utf8).is_in([str(v) for v in char_values])
                )
        return filtered_df

    def process_single_row(filtered_df, current_char_name, current_values):
        """Process a single row to generate options and validate current values."""
        value_counts = (filtered_df.select(pl.col(current_char_name))
                        .filter(pl.col(current_char_name).is_not_null())
                        .group_by(current_char_name)
                        .count()
                        .sort(current_char_name)
                        .to_dict(as_series=False))
        available_values = {str(v) for v in value_counts[current_char_name] if v is not None}
        
        options = [
            {
                'label': f"{str(value)} ({count:,})",
                'value': str(value),
                'title': f"{str(value)} - {count:,} buildings"
            }
            for value, count in zip(value_counts[current_char_name], value_counts['count'])
        ]
        
        new_value = None
        if current_values:
            current_values_str = [str(v) for v in current_values]
            valid_values = [v for v in current_values_str if v in available_values]
            new_value = valid_values if valid_values else None
        
        return options, new_value


    @app.callback(
        Output('resolution-dropdown', 'options'),
        Output('resolution-dropdown', 'value'),
        Output('time-agg-dropdown', 'disabled'),
        Input('shape-type-dropdown', 'value')
    )
    def update_resolution_options(shape_type):       
        shape_types = ['daily_shape', 'weekend_shape', 'weekday_shape']
        has_shape_type = any(shape in shape_type for shape in shape_types)
        
        if not shape_type:
            return ['15min', 'hour', 'day', 'week', 'month'], '15min', True
        else:
            return ['15min', 'hour'], '15min', False

    @app.callback(
        Output('loading-output', 'children'),
        Input('update-plot-button', 'n_clicks'),
        State('upgrade-dropdown', 'value'),
        State('plot-dropdown', 'value'),
        State('building-agg-dropdown', 'value'),
        State('time-agg-dropdown', 'value'),
        State('shape-type-dropdown', 'value'),
        State('resolution-dropdown', 'value'),
        State('start-month-dropdown', 'value'),
        State('start-day-dropdown', 'value'),
        State('end-month-dropdown', 'value'),
        State('end-day-dropdown', 'value'),
        State({'type': 'char-name', 'index': ALL}, 'value'),
        State({'type': 'char-value', 'index': ALL}, 'value'),
        State('cost-store', 'data'),
        prevent_initial_call=True,
        background=True,
        manager=background_callback_manager,
        running=[
            (Output('update-plot-button', 'disabled'), True, False),
            (Output('cancel-button', 'disabled'), False, True),
        ],
        cancel=[Input('cancel-button', 'n_clicks')]
    )
    def update_plot(n_clicks, upgrades, plot_cols, building_aggs, time_aggs, 
                    shape_types, resolution, start_month, start_day, 
                    end_month, end_day, filter_names, filter_values, current_cost):
        print(f"update_plot called with n_clicks: {n_clicks}")
        if not n_clicks or not plot_cols or upgrades is None:
            raise PreventUpdate
        
        # Ensure all inputs are lists
        upgrades = [upgrades] if not isinstance(upgrades, list) else upgrades
        plot_cols = [plot_cols] if not isinstance(plot_cols, list) else plot_cols
        building_aggs = [building_aggs] if not isinstance(building_aggs, list) else building_aggs
        time_aggs = [time_aggs] if not isinstance(time_aggs, list) else time_aggs
        shape_types = shape_types or [None]
        shape_types = [shape_types] if not isinstance(shape_types, list) else shape_types

        traces = []
        total_plots = len(upgrades) * len(plot_cols) * len(building_aggs) * len(time_aggs) * len(shape_types)
        completed = 0
        
        # Initialize the figure structure
        figure = {
            'data': [],
            'layout': {
                'title': 'Energy Usage Over Time',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Energy Usage'},
                'showlegend': True,
                'legend': {'orientation': 'h', 'y': -0.2},
                'margin': {'l': 60, 'r': 20, 't': 40, 'b': 60}
            }
        }
        
        total_added_cost = 0
        total_added_gb = 0
        month_to_num = {month: idx for idx, month in enumerate(MONTHS, 1)}
        for upgrade in upgrades:
            for enduse in plot_cols:
                for building_agg in building_aggs:
                    for time_agg in time_aggs:
                        for shape_type in shape_types:
                            # Update status
                            shape_agg_str = '' if not shape_type else f"-{shape_type}_{time_agg}"
                            name = f"Upgrade {upgrade} - {enduse} ({building_agg}{shape_agg_str})"
                            status = f"Processing {completed + 1}/{total_plots}: {name}"
                            print(status)
                            run_obj = viz_data.run_obj(upgrade)
                            run_obj.load_cache()
                            db_filter_names = [f"{run_obj._char_prefix}{name}" for name in filter_names if name is not None]
                            restrict = list(zip(db_filter_names, filter_values))

                            pre_dollars = run_obj.execution_cost.get('Dollars', 0)
                            pre_gb = run_obj.execution_cost.get('GB', 0)

                            df =run_obj.agg.aggregate_timeseries(
                                enduses=[enduse],
                                upgrade_id=upgrade,
                                restrict=restrict,
                                agg_func=building_agg,
                            )

                            # Capture costs after query and calculate difference
                            post_dollars = run_obj.execution_cost.get('Dollars', 0)
                            post_gb = run_obj.execution_cost.get('GB', 0)
                            total_added_cost += post_dollars - pre_dollars
                            total_added_gb += post_gb - pre_gb

                            new_cost_data = {
                                'dollars': current_cost['dollars'] + total_added_cost,
                                'gb': current_cost['gb'] + total_added_gb
                            }
                            set_props('cost-store', {"data": new_cost_data})

                            viz_data.run_obj(upgrade).save_cache()
                            if building_agg != 'sum':
                                new_col = f"{enduse}__{building_agg}"
                            else:
                                new_col = enduse                

                            df = df.sort_values('time')
                            if df.empty:
                                print(f"No data found for {upgrade} - {enduse}")
                                continue
                            sample_count = df['sample_count'].iloc[0] if 'sample_count' in df.columns else 'N/A'
                            first_date = df['time'].iloc[0]
                            first_offset = first_date - datetime.datetime(first_date.year, 1, 1)
                            df['time'] = df['time'].dt.tz_localize(None) - first_offset
                            
                            df = df[
                                (df['time'].dt.month > month_to_num[start_month]) & 
                                (df['time'].dt.month < month_to_num[end_month]) |
                                
                                ((df['time'].dt.month == month_to_num[start_month]) &
                                (df['time'].dt.day >= int(start_day))) |

                                ((df['time'].dt.month == month_to_num[end_month]) &
                                (df['time'].dt.day <= int(end_day)))
                            ]

                            freq_map = {
                                'day': '1D',
                                'week': '1W',
                                'hour': '1H', 
                                '15min': '15T',
                                'month': '1M'
                            }

                            agg_func = {
                                'avg': 'mean',
                                'max': 'max', 
                                'min': 'min',
                            }.get(time_agg, 'sum')

                            df = df.groupby(pd.Grouper(key='time', freq=freq_map[resolution])).agg({new_col: 'sum'})

                            if shape_type:
                                # Create a base date using the first date from the data
                                base_date = datetime.datetime(first_date.year, 1, 1)
                                if resolution == '15min':
                                    periods = 96  # 24 hours * 4 periods per hour
                                    freq = '15T'
                                else:  # hourly
                                    periods = 24
                                    freq = 'H'
                                time_index = pd.date_range(base_date, periods=periods, freq=freq)
                                if shape_type == 'weekend_shape':
                                    mask = df.index.weekday.isin([5, 6])
                                elif shape_type == 'weekday_shape': 
                                    mask = df.index.weekday.isin([0, 1, 2, 3, 4])
                                else: # daily_shape
                                    mask = slice(None)
                                df = df[mask]
                                group_cols = [df.index.hour, df.index.minute] if resolution == '15min' else [df.index.hour]
                                df = df.groupby(group_cols).agg({new_col: agg_func})
                                df = pd.DataFrame({new_col: df[new_col].values}, index=time_index)

                            new_trace = {
                                'x': df.index.tolist(),
                                'y': df[new_col].tolist(),
                                'name': name,
                                'type': 'scatter',
                                'mode': 'lines',
                                'hovertemplate': f'Value: %{{y:,.2f}}<br>Time: %{{x}}<br>Sample Count: {sample_count}'
                            }
                            
                            traces.append(new_trace)
                            completed += 1
                            
                            # Update the graph progressively
                            figure['data'] = traces
                            figure['layout']['title'] = f"Completed {completed}/{total_plots}"
                            set_props('main-graph', {"figure":figure})

        figure['layout']['title'] = f"Completed all {total_plots} plots."
        set_props('main-graph', {"figure":figure, "config": {'edits': {"titleText": True, "axisTitleText": True, 'legendText': True}}})
        return ""



    @app.callback(
        Output('cancel-button', 'n_clicks'),
        Input('cancel-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def cancel_queries(n_clicks):
        if not n_clicks:
            raise PreventUpdate
            
        # Stop all queries for each upgrade
        for upgrade in viz_data.upgrade2shortname.keys():
            viz_data.run_obj(str(upgrade)).stop_all_queries()
            
        return None

    @app.callback(
        Output('cost-display', 'children'),
        Input('cost-store', 'data'),
    )
    def update_cost_display(cost_data):
        if not cost_data:
            return "Query Cost: $0.000 (0.00 GB)"
        
        return f"Query Cost: ${cost_data['dollars']:.3f} ({cost_data['gb']:.2f} GB)"

    return app

def main():
    print("Welcome to Upgrades Visualizer.")
    defaults = load_script_defaults("project_info")
    # check if variable 'done' exists and is 1
    debug = True
    if debug:
        opt_sat_file = defaults.get("opt_sat_file", "")
        workgroup = defaults.get("workgroup", "")
        db_name = defaults.get("db_name", "")
        table_name = defaults.get("table_name", "")
        upgrades_selection = defaults.get("upgrades_selection", "")
    else:
        opt_sat_file = inquirer.text(message="Please enter path to the options saturation csv file:",
                                    default=defaults.get("opt_sat_file", "")).execute().strip()
        workgroup = inquirer.text(message="Please enter Athena workgroup name:",
                                default=defaults.get("workgroup", "")).execute().strip()
        db_name = inquirer.text(message="Please enter database name "
                                "(found in postprocessing:aws:athena in the buildstock configuration file):",
                                default=defaults.get("db_name", "")).execute().strip()
        table_name = inquirer.text(message="Please enter table name (same as output folder name; found under "
                                "output_directory in the buildstock configuration file). [Enter two names "
                                "separated by comma if baseline and upgrades are in different run]:",
                                default=defaults.get("table_name", "")
                                ).execute().strip()
        upgrades_selection = inquirer.text(message="Please enter upgrade ids separated by comma and dashes "
                                        "(example: `1-3,5,7,8-9`) or leave empty to include all upgrades.",
                                        default=defaults.get("upgrades_selection", "")).execute()
        defaults.update({"opt_sat_file": opt_sat_file, "workgroup": workgroup,
                        "db_name": db_name, "table_name": table_name,
                        "upgrades_selection": upgrades_selection})
        save_script_defaults("project_info", defaults)
    if ',' in table_name:
        table_name = table_name.split(',')
    viz_data = get_viz_data(opt_sat_path=opt_sat_file, db_name=db_name, table_name=table_name, workgroup=workgroup,
                            buildstock_type='resstock', include_monthly=False,
                            upgrades_selection_str=upgrades_selection, init_query=False)
    app = get_app(viz_data)
    # get debug from env variable
    import os
    debug = os.getenv("DEBUG", "True").lower() == "true"
    port = os.getenv("PORT", "8006")
    print(f"Debug mode: {debug}, port: {port}")
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    main()
