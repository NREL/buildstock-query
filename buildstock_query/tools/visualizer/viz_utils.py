import re

from buildstock_query.tools.visualizer.viz_data import VizData


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


def get_int_set(input_str):
    """
        Convert "1,2,3-6,8,9" to [1, 2, 3, 4, 5, 6, 8, 9]
    """
    if not input_str:
        return set()

    pattern = r'^(\d+(-\d+)?,)*(\d+(-\d+)?)$'
    if not re.match(pattern, input_str):
        raise ValueError(f"{input_str} is not a valid pattern for list")

    result = set()
    segments = input_str.split(',')
    for segment in segments:
        if '-' in segment:
            start, end = map(int, segment.split('-'))
            result |= set(range(start, end + 1))
        else:
            result.add(int(segment))

    return result


def get_viz_data(opt_sat_path, db_name, db_schema, table_name, workgroup, buildstock_type, include_monthly, upgrades_selection_str, init_query):
    viz_data = VizData(opt_sat_path=opt_sat_path, db_name=db_name, db_schema=db_schema,
                       run=table_name, workgroup=workgroup, buildstock_type=buildstock_type,
                       upgrades_selection=get_int_set(upgrades_selection_str)
                       )
    if init_query:
        viz_data.init_change2bldgs()
        viz_data.init_annual_results()
        if include_monthly:
            viz_data.init_monthly_results()
    return viz_data