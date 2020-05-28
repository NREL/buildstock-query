smart_query
==================

This module provides a set of helper functions to standardize queries to download common results.
It leverages AWS boto3 Athena API to run queries in parallel, and tremendously speedup the query time for some of the
heavy time series queries.

For usage example, look at: [smart_query_example_resstock](../data_generation/smart_query_example_resstock.py) and 
[smart_query_example_comstock](../data_generation/smart_query_example_comstock.py)