from buildstock_query.tools.upgrades_visualizer.plot_utils import PlotParams, ValueTypes
from buildstock_query.tools.upgrades_visualizer.viz_data import VizData
import plotly.graph_objects as go
import polars as pl


class UpgradesPlot:
    def __init__(self, viz_data: VizData) -> None:
        self.viz_data = viz_data

    def get_ylabel(self, end_use):
        if len(end_use) == 1:
            return end_use[0]
        pure_end_use_name = end_use[0].removeprefix("end_use_")
        pure_end_use_name = pure_end_use_name.removeprefix("fuel_use_")
        pure_end_use_name = "_".join(pure_end_use_name.split("_")[1:])
        return f"{len(end_use)}_fuels_{pure_end_use_name}"

    def get_plot(self, df, params: PlotParams):
        fig = go.Figure()
        counter = 0
        counter2 = 0
        report_dfs = [pl.DataFrame()]

        if params.value_type in [ValueTypes.mean, ValueTypes.total, ValueTypes.count]:
            xtitle = ", ".join(params.group_by[1:]) if len(params.group_by) > 1 else params.group_by[0]
            ytitle = f"{self.get_ylabel(params.enduses)}_{params.value_type.value}_{params.savings_type.value}"
        elif params.value_type in [ValueTypes.distribution]:
            xtitle = ", ".join(params.group_by[1:]) if len(params.group_by) > 1 else params.group_by[0]
            ytitle = f"{self.get_ylabel(params.enduses)}_{params.savings_type.value}"
        else:
            assert params.value_type in [ValueTypes.scatter]
            xtitle = "baseline_value"
            ytitle = f"{self.get_ylabel(params.enduses)}_{params.savings_type.value}"
        for grp0, sub_df in df.groupby(params.group_by[0], maintain_order=True):
            upgrade = grp0 if params.group_by[0] == 'upgrade' else params.upgrades_to_plot[0]
            yvals = []
            xvals = []
            second_groups = []
            sample_counts = []
            upgrades = []
            hovervals = []
            if len(params.group_by) > 1:
                second_plots = [(group_name, group_df) for group_name, group_df in
                                sub_df.groupby(params.group_by[1:], maintain_order=True)]
            else:
                second_plots = [(tuple(), sub_df)]
            for second_name, second_df in second_plots:
                name = ','.join(second_name) if second_name else str(grp0)
                count = len(second_df)
                mean = pl.mean(second_df['value'])
                if counter >= 500:
                    yvals.append(0.1)
                    xvals.append("Too many groups")
                    sample_counts.append(0)
                    upgrades.append(upgrade)
                    hovervals.append("Too many groups")
                    grp0 = "Too many groups"
                    break
                if params.value_type in [ValueTypes.total, ValueTypes.mean, ValueTypes.count]:
                    if params.value_type == ValueTypes.total:
                        val = pl.sum(second_df['value'])
                    elif params.value_type == ValueTypes.count:
                        val = second_df['building_id'].n_unique()
                    else:
                        val = pl.mean(second_df['value'])
                    val = float(val)
                    yvals.append(val)
                    xvals.append(name)
                    sample_counts.append(count)
                    upgrades.append(upgrade)
                    hovertext = f"{self.viz_data.upgrade2name.get(upgrade)}<br>{grp0}<br>{name}<br>Average {mean}."\
                        f"<br>Sample Count: {count}."
                    f"<br>Units Count: {count * self.viz_data.sample_weight}."
                    hovervals.append(hovertext)
                    second_groups.append(name)
                elif params.value_type in [ValueTypes.distribution, ValueTypes.scatter]:
                    hovertext = [f'{self.viz_data.upgrade2name.get(upgrade)}<br>{grp0}<br>{name}<br>Building:'
                                 f'{bid}<br>Sample Count: {count}'
                                 for bid in second_df['building_id'].to_list()]
                    if params.value_type == ValueTypes.distribution:
                        xvals.extend([name] * count)
                        yvals.extend(second_df['value'].to_list())
                        second_groups.extend([name] * count)
                        sample_counts.extend([count] * count)
                        upgrades.extend([upgrade] * count)
                        hovervals.extend(hovertext)
                    else:
                        xvals.append(second_df['baseline_value'].to_list())
                        yvals.append(second_df['value'].to_list())
                        second_groups.append(name)
                        sample_counts.append([count] * count)
                        upgrades.append([upgrade] * count)
                        hovervals.append(hovertext)
                counter += 1
                counter2 += 1

            self._add_plot(params, fig, grp0, yvals, xvals, second_groups, hovervals, len(params.group_by) > 1)
            try:
                sub_df = pl.DataFrame({xtitle: xvals, ytitle: yvals, 'upgrade': [f'Upgrade {grp0}'] * len(xvals),
                                      'sample_count': sample_counts, 'info': hovervals})
                # sub_df = sub_df.with_columns(pl.col(ytitle).cast(pl.Float32))
                report_dfs.append(sub_df)
            except Exception:
                continue

        if params.change_type:
            title = f"{params.value_type} - {params.savings_type} for {params.change_type} buildings"
        else:
            title = f'{params.value_type} - {params.savings_type} value'

        if params.group_by[0] != "upgrade":
            title = f"Upgrade {params.upgrades_to_plot[0]} - {title}"
        self._update_layout(params, fig, xtitle, ytitle, title, len(params.group_by) > 1)

        return fig, pl.concat(report_dfs)

    def _update_layout(self, params: PlotParams, fig, xtitle, ytitle, title, multi_group=False):
        if params.value_type in [ValueTypes.mean, ValueTypes.total, ValueTypes.count]:
            fig.update_layout(yaxis_title=ytitle,
                              barmode='group',
                              xaxis_title=xtitle,
                              legend={"title": params.group_by[0]},
                              title=title)
        elif params.value_type in [ValueTypes.distribution]:
            fig.update_layout(yaxis_title=ytitle,
                              boxmode="group" if multi_group else "overlay",
                              xaxis_title=xtitle,
                              title=title,
                              legend={"title": params.group_by[0]},
                              clickmode='event+select')
        elif params.value_type == ValueTypes.scatter:
            fig.update_layout(yaxis_title=ytitle,
                              title=title,
                              boxmode="group" if multi_group else "overlay",
                              legend={"title": params.group_by[0]},
                              clickmode='event+select')
            fig.add_annotation(
                dict(
                    x=0.5,
                    y=-0.20,
                    showarrow=False,
                    text=xtitle,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=14)
                )
            )

    def _add_plot(self, params: PlotParams, fig, grp0, yvals, xvals, second_groups, hovervals, multi_group=False):
        if params.value_type in [ValueTypes.mean, ValueTypes.total, ValueTypes.count]:
            fig.add_trace(go.Bar(
                    y=yvals,
                    x=xvals,
                    hovertext=hovervals,
                    name=f'{grp0}',
                    hoverinfo="all"
                )).update_traces(
                    marker={"line": {"width": 0.5, "color": "rgb(0,0,0)"}}
                )
        elif params.value_type == ValueTypes.distribution:
            fig.add_trace(go.Box(
                y=yvals,
                x=xvals,
                name=f'{grp0}',
                boxpoints='suspectedoutliers',
                boxmean=True,  # represent mean
                hovertext=hovervals,
                hoverinfo="all"
            ))
        elif params.value_type == ValueTypes.scatter:
            if multi_group:
                self._add_multi_scatter(fig, grp0, list(zip(second_groups, xvals, yvals, hovervals)))
            else:
                fig.add_trace(go.Scatter(
                    y=yvals[0],
                    x=xvals[0],
                    name=f'{grp0}',
                    mode='markers',
                    hovertext=hovervals[0],
                    hoverinfo="all"))

    def _add_multi_scatter(self, fig, grp0, plot_tuples):
        # Define the width of each subplot based on total plots and desired gaps
        num_plots = len(plot_tuples)
        gap_fraction = 0.05  # for a 5% gap
        subplot_width = (1.0 - (num_plots-1)*gap_fraction) / num_plots

        # Iterate over the tuples and add scatter plots
        for index, (name, xvals, yvals, hovervals) in enumerate(plot_tuples):
            domain_start = index * (subplot_width + gap_fraction)
            domain_end = domain_start + subplot_width

            # Constructing the appropriate xaxis and yaxis keys
            scatter_xaxis_key = 'x' if index == 0 else f'x{index+1}'
            scatter_yaxis_key = 'y' if index == 0 else f'y{index+1}'

            layout_xaxis_key = f'xaxis{"" if index == 0 else index+1}'
            layout_yaxis_key = f'yaxis{"" if index == 0 else index+1}'

            # Add scatter plot
            fig.add_trace(go.Scatter(x=xvals,
                                     y=yvals,
                                     hovertext=hovervals[0],
                                     name=grp0,
                                     legendgroup=grp0,
                                     mode='markers',
                                     showlegend=True if index == 0 else False,
                                     xaxis=scatter_xaxis_key,
                                     yaxis=scatter_yaxis_key,
                                     hoverinfo="all"))

            # Update the layout with the appropriate axis properties
            fig.update_layout({
                layout_xaxis_key: {
                    'domain': [domain_start, domain_end],
                    'anchor': scatter_yaxis_key,
                    'title': name
                },
                layout_yaxis_key: {
                    'anchor': scatter_xaxis_key
                }
            })
