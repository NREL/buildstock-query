buildstock-query (formerly known as smart_query)
==================

This module provides a set of helper functions to standardize queries to download common results.
It leverages AWS boto3 Athena API to run queries in parallel, and tremendously speedup the query time for some of the
heavy time series queries.
