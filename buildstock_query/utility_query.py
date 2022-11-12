"""
# EULPAthena
- - - - - - - - -
A class to run AWS Athena queries for the EULP project using built-in query functions. New query use cases for the
EULP project should be implemented as member function of this class.

:author: Rajendra.Adhikari@nrel.gov

:author: Anthony.Fontanini@nrel.gov
"""

import buildstock_query.main as main
import logging
from typing import List, Any, Tuple
import pandas as pd
import sqlalchemy as sa
from collections import defaultdict
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildStockUtility:
    def __init__(self, buildstock_query: 'main.BuildStockQuery',
                 eia_mapping_year: int = 2018, eia_mapping_version: int = 1):
        """
        A class to run AWS Athena queries for the EULP project using built-in query functions. Look up definition in \
        the ResStockAthena to understand other args and kwargs.
        Args:
            eia_mapping_year: The year of the EIA form 861 service territory map to use when mapping to utility \
            service territories. Currently, only 2018 and 2012 are valid years.
        """
        self.bsq = buildstock_query
        self.agg = buildstock_query.agg
        self.group_query_id = 0
        self.eia_mapping_year = eia_mapping_year
        self.eia_mapping_version = eia_mapping_version
        self.cache: dict = defaultdict()

    def _aggregate_ts_by_map(self,
                             map_table_name: str,
                             baseline_column: str,
                             map_column: str,
                             id_column: str,
                             id_list: List[Any],
                             enduses: List[str],
                             group_by: List[str] = None,
                             get_query_only: bool = False,
                             query_group_size: int = 1,
                             split_endues: bool = False):

        group_by = [] if group_by is None else group_by
        new_table = map_table_name
        join_list = [(new_table, baseline_column, map_column)]
        logger.info(f"Will submit request for {id_list}")
        GS = query_group_size
        id_list_batches = [id_list[i:i+GS] for i in range(0, len(id_list), GS)]
        results_array = []
        for current_ids in id_list_batches:
            if len(current_ids) == 1:
                current_ids = current_ids[0]
            logger.info(f"Submitting query for {current_ids}")
            if split_endues:
                logger.info("Splitting the query into separate queries for each enduse.")
                result_df = self.agg.aggregate_timeseries(enduses=enduses,
                                                          group_by=[id_column] + group_by,
                                                          join_list=join_list,
                                                          weights=['weight'],
                                                          sort=True,
                                                          restrict=[(id_column, current_ids)],
                                                          run_async=False,
                                                          get_query_only=get_query_only,
                                                          split_enduses=True)
                results_array.append(result_df)
            else:
                query = self.agg.aggregate_timeseries(enduses=enduses,
                                                      group_by=[id_column] + group_by,
                                                      join_list=join_list,
                                                      weights=['weight'],
                                                      sort=True,
                                                      restrict=[(id_column, current_ids)],
                                                      run_async=True,
                                                      get_query_only=True,
                                                      split_enduses=False)
                results_array.append(query)

        if get_query_only:
            return results_array

        if split_endues:
            # In this case, the resuls_array will contain the result dataframes
            logger.info("Concatenating the results from all IDs")
            all_dfs = pd.concat(results_array)
            return all_dfs
        else:
            # In this case, results_array will contain the queries
            batch_query_id = self.bsq.submit_batch_query(results_array)
            return self.bsq.get_batch_query_result(batch_id=batch_query_id)

    def get_eiaid_map(self) -> tuple[str, str, str]:
        if self.eia_mapping_version == 1:
            map_table_name = 'eiaid_weights'
            map_baseline_column = 'build_existing_model.county'
            map_eiaid_column = 'county'
        elif self.eia_mapping_version == 2:
            map_table_name = 'v2_eiaid_weights'
            map_baseline_column = 'build_existing_model.county'
            map_eiaid_column = 'county'
        elif self.eia_mapping_version == 3:
            map_table_name = 'v3_eiaid_weights_%d' % (self.eia_mapping_year)
            map_baseline_column = 'build_existing_model.county'
            map_eiaid_column = 'county'
        else:
            raise ValueError("Invalid mapping_version")

        return map_table_name, map_baseline_column, map_eiaid_column

    def aggregate_ts_by_eiaid(self, eiaid_list: List[str], enduses: List[str] = None, group_by: List[str] = None,
                              get_query_only: bool = False,
                              query_group_size: int = None,
                              split_endues: bool = False,
                              ):
        """
        Aggregates the timeseries result, grouping by utilities.
        Args:
            eiaid_list: The list of utility ids (EIAID) assigned by EIA.
            enduses: The list of enduses to aggregate
            group_by: Additional columns to group the aggregation by
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the list of queries to run instead of the result.
            query_group_size: The number of eiaids to be grouped together when running athena queries. This should be
                              used as large as possible that doesn't result in query timeout.
            split_endues: Query each enduses separately to spread load on Athena

        Returns:
            Pandas dataframe with the aggregated timeseries and the requested enduses grouped by utilities
        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        eiaid_list = [str(e) for e in eiaid_list]
        if not enduses:
            raise ValueError("Need to provide enduses")
        id_column = 'eiaid'

        if query_group_size is None:
            query_group_size = min(100, len(eiaid_list))

        return self._aggregate_ts_by_map(eiaid_map_table_name, map_baseline_column, map_eiaid_column, id_column,
                                         eiaid_list, enduses, group_by, get_query_only,
                                         query_group_size, split_endues)

    def aggregate_unit_counts_by_eiaid(self, eiaid_list: List[str] = None, group_by: List[str] = None,
                                       get_query_only: bool = False):
        """
        Returns the counts of the number of dwelling units, grouping by eiaid and other additional group_by columns if
        provided.
        Args:
            eiaid_list: The list of utility ids (EIAID) to aggregate for
            group_by: Additional columns to group by
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the query instead of the result

        Returns:
            Pandas dataframe with the units counts
        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        group_by = [] if group_by is None else group_by
        restrict = [('eiaid', eiaid_list)] if eiaid_list else None
        eiaid_col = self.bsq.get_column("eiaid", eiaid_map_table_name)
        result = self.agg.aggregate_annual([], group_by=[eiaid_col] + group_by,
                                           sort=True,
                                           join_list=[(eiaid_map_table_name, map_baseline_column, map_eiaid_column)],
                                           weights=['weight'],
                                           restrict=restrict,
                                           get_query_only=get_query_only)
        return result

    def aggregate_annual_by_eiaid(self, enduses: List[str], group_by: List[str] = None,
                                  get_query_only: bool = False):
        """
        Aggregates the annual consumption in the baseline table, grouping by all the utilities
        Args:
            enduses: The list of enduses to aggregate
            group_by: Additional columns to group the aggregation by
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the list of queries to run instead of the result.
        Returns:
            Pandas dataframe with the annual sum of the requested enduses, grouped by utilities
        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        join_list = [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)]
        group_by = [] if group_by is None else group_by
        eiaid_col = self.bsq.get_column("eiaid", eiaid_map_table_name)
        result = self.agg.aggregate_annual(enduses=enduses, group_by=[eiaid_col] + group_by,
                                           join_list=join_list,
                                           weights=['weight'],
                                           sort=True,
                                           get_query_only=get_query_only)
        return result

    def get_filtered_results_csv_by_eiaid(
            self, eiaids: List[str], get_query_only: bool = False):
        """
        Returns a portion of the results csvs, which belongs to given list of utilities
        Args:
            eiaids: The eiaid list of utitlies
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe that is a subset of the results csv, that belongs to provided list of utilities
        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        query = sa.select(['*']).select_from(self.bsq.bs_table)
        query = self.bsq._add_join(query, [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)])
        query = self.bsq._add_restrict(query, [("eiaid", eiaids)])
        query = query.where(self.bsq.get_column("weight") > 0)
        if get_query_only:
            return self.bsq._compile(query)
        res = self.bsq.execute(query)
        return res

    def get_eiaids(self, restrict: List[Tuple[str, List]] = None):
        """
        Returns the list of building
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
        Returns:
            Pandas dataframe consisting of the eiaids belonging to the provided list of locations.
        """
        restrict = list(restrict) if restrict else []
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        eiaid_col = self.bsq.get_column("eiaid", eiaid_map_table_name)
        if 'eiaids' in self.cache:
            if self.bsq.db_name + '/' + eiaid_map_table_name in self.cache['eiaids']:
                return self.cache['eiaids'][self.bsq.db_name + '/' + eiaid_map_table_name]
        else:
            self.cache['eiaids'] = {}

        join_list = [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)]
        annual_agg = self.agg.aggregate_annual(enduses=[], group_by=[eiaid_col],
                                               restrict=restrict,
                                               join_list=join_list,
                                               weights=['weight'],
                                               sort=True)
        self.cache['eiaids'] = list(annual_agg['eiaid'].values)
        return self.cache['eiaids']

    def get_buildings_by_eiaids(self, eiaids: List[str], get_query_only: bool = False):
        """
        Returns the list of buildings belonging to the given list of utilities.
        Args:
            eiaids: list of utility EIAIDs
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the building ids belonging to the provided list of utilities.

        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        query = sa.select([self.bsq.bs_bldgid_column.distinct()])
        query = self.bsq._add_join(query, [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)])
        query = self.bsq._add_restrict(query, [("eiaid", eiaids)])
        query = query.where(self.bsq.get_column("weight") > 0)
        if get_query_only:
            return self.bsq._compile(query)
        res = self.bsq.execute(query)
        return res

    def get_locations_by_eiaids(self, eiaids: List[str], get_query_only: bool = False):
        """
        Returns the list of locations/counties (depends on mapping version) belonging to a given list of utilities.
        Args:
            eiaids: list of utility EIAIDs
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the locations (for version 1) or counties (for version 2) belonging to the
            provided list of utilities.

        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        self.bsq.get_table(eiaid_map_table_name)
        query = sa.select([self.bsq.get_column(map_eiaid_column).distinct()])
        query = self.bsq._add_restrict(query, [("eiaid", eiaids)])
        query = query.where(self.bsq.get_column("weight") > 0)
        if get_query_only:
            return self.bsq._compile(query)
        res = self.bsq.execute(query)
        return res
