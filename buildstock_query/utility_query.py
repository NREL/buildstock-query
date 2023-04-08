import buildstock_query.main as main
import logging
from typing import List, Any, Tuple
import pandas as pd
import sqlalchemy as sa
from collections import defaultdict
from typing import Optional, Union, Sequence
from buildstock_query.schema import UtilityTSQuery
from buildstock_query.schema.helpers import gather_params
from buildstock_query.schema.query_params import AnyColType
import typing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildStockUtility:
    """
    Class to perform electric utility centric queries.
    """

    def __init__(self, buildstock_query: 'main.BuildStockQuery',
                 eia_mapping_year: int = 2018, eia_mapping_version: int = 1):
        """
        Class to perform electric utility centric queries
        Args:
            eia_mapping_year: The year of the EIA form 861 service territory map to use when mapping to utility \
                              service territories. Currently, only 2018 and 2012 are valid years.
            eia_mapping_version: The EIA mapping version to use.
        """
        self._bsq = buildstock_query
        self._agg = buildstock_query.agg
        self._group_query_id = 0
        self.eia_mapping_year = eia_mapping_year
        self.eia_mapping_version = eia_mapping_version
        self._cache: dict = defaultdict()

    def _aggregate_ts_by_map(self,
                             map_table_name: str,
                             baseline_column_name: str,
                             map_column_name: str,
                             id_column: str,
                             id_list: Sequence[Any],
                             params: UtilityTSQuery):
        new_table = self._bsq.get_table(map_table_name)
        new_column = self._bsq.get_column(map_column_name, table_name=map_table_name)
        baseline_column = self._bsq.get_column(baseline_column_name, self._bsq.bs_table)
        params.group_by = [id_column] + list(params.group_by)
        params.weights = list(params.weights) + ['weight']
        params.join_list = [(new_table, baseline_column, new_column)] + list(params.join_list)
        logger.info(f"Will submit request for {id_list}")
        GS = params.query_group_size
        id_list_batches = [id_list[i:i + GS] for i in range(0, len(id_list), GS)]
        results_array = []
        for current_ids in id_list_batches:
            new_params = params.copy(deep=True)
            if len(current_ids) == 1:
                current_ids = current_ids[0]
            new_params.restrict = [(id_column, current_ids)] + list(new_params.restrict)
            logger.info(f"Submitting query for {current_ids}")
            result = self._agg.aggregate_timeseries(params=new_params)
            results_array.append(result)

        if params.get_query_only:
            return results_array

        if params.split_enduses:
            # In this case, the resuls_array will contain the result dataframes
            logger.info("Concatenating the results from all IDs")
            all_dfs = pd.concat(results_array)
            return all_dfs
        else:
            # In this case, results_array will contain the queries
            batch_query_id = self._bsq.submit_batch_query(results_array)
            return self._bsq.get_batch_query_result(batch_id=batch_query_id)

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

    @gather_params(UtilityTSQuery)
    def aggregate_ts_by_eiaid(self, params: UtilityTSQuery):
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
        if not params.enduses:
            raise ValueError("Need to provide enduses")
        id_column = 'eiaid'

        if params.query_group_size is None:
            params.query_group_size = min(100, len(params.eiaid_list))

        return self._aggregate_ts_by_map(map_table_name=eiaid_map_table_name,
                                         baseline_column_name=map_baseline_column,
                                         map_column_name=map_eiaid_column,
                                         id_column=id_column,
                                         id_list=params.eiaid_list,
                                         params=params)

    def aggregate_unit_counts_by_eiaid(self, eiaid_list: list[str],
                                       group_by: list[Union[AnyColType, tuple[str, str]]] = [],
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
        logger.info("Aggregating unit counts by eiaid")
        group_by = group_by or []
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        group_by = [] if group_by is None else group_by
        restrict = [('eiaid', eiaid_list)]
        eiaid_col = self._bsq.get_column("eiaid", eiaid_map_table_name)
        result = self._agg.aggregate_annual(enduses=[], group_by=[eiaid_col] + group_by,
                                            sort=True,
                                            join_list=[(eiaid_map_table_name, map_baseline_column, map_eiaid_column)],
                                            weights=['weight'],
                                            restrict=restrict,
                                            get_query_only=get_query_only)
        return result

    def aggregate_annual_by_eiaid(self, enduses: List[str], group_by: Optional[List[str]] = None,
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
        eiaid_col = self._bsq.get_column("eiaid", eiaid_map_table_name)
        result = self._agg.aggregate_annual(enduses=enduses, group_by=[eiaid_col] + group_by,
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
        query = sa.select(['*']).select_from(self._bsq.bs_table)
        query = self._bsq._add_join(query, [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)])
        query = self._bsq._add_restrict(query, [("eiaid", eiaids)])
        query = query.where(self._bsq.get_column("weight") > 0)
        if get_query_only:
            return self._bsq._compile(query)
        res = self._bsq.execute(query)
        return res

    def get_eiaids(self, restrict: Optional[List[Tuple[str, List]]] = None):
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
        eiaid_col = self._bsq.get_column("eiaid", eiaid_map_table_name)
        if 'eiaids' in self._cache:
            if self._bsq.db_name + '/' + eiaid_map_table_name in self._cache['eiaids']:
                return self._cache['eiaids'][self._bsq.db_name + '/' + eiaid_map_table_name]
        else:
            self._cache['eiaids'] = {}

        join_list = [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)]
        annual_agg = self._agg.aggregate_annual(enduses=[], group_by=[eiaid_col],
                                                restrict=restrict,
                                                join_list=join_list,
                                                weights=['weight'],
                                                sort=True)
        self._cache['eiaids'] = list(annual_agg['eiaid'].values)
        return self._cache['eiaids']

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
        query = sa.select([self._bsq.bs_bldgid_column.distinct()])
        query = self._bsq._add_join(query, [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)])
        query = self._bsq._add_restrict(query, [("eiaid", eiaids)])
        query = query.where(self._bsq.get_column("weight") > 0)
        if get_query_only:
            return self._bsq._compile(query)
        res = self._bsq.execute(query)
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
        eiaid_map_table = self._bsq.get_table(eiaid_map_table_name)
        query = sa.select([eiaid_map_table.c[map_eiaid_column].distinct()])
        query = self._bsq._add_restrict(query, [("eiaid", eiaids)])
        query = query.where(eiaid_map_table.c["weight"] > 0)
        if get_query_only:
            return self._bsq._compile(query)
        res = self._bsq.execute(query)
        return list(res[map_eiaid_column].values)
