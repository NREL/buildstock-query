"""
# EULPAthena
- - - - - - - - -
A class to run AWS Athena queries for the EULP project using  built-in query functions. New query use cases for the
EULP project should be implemented as member function of this class.

:author: Rajendra.Adhikari@nrel.gov
"""

from eulpda.smart_query.ResStockAthena import ResStockAthena
import logging
from typing import List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EULPAthena(ResStockAthena):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_query_id = 0
        self.query_store = {}
        self.execution_id_list = []
        self.query_list = []
        self.batch_query_status_map = {}

    def _aggregate_ts_by_map(self,
                             map_table_name: str,
                             baseline_column: str,
                             map_column: str,
                             id_column: str,
                             id_list: List[Any],
                             enduses: List[str],
                             group_by: List[str],
                             get_query_only: bool = False):

        group_by = [] if group_by is None else group_by
        new_table = map_table_name
        join_list = [(new_table, baseline_column, map_column)]
        logger.info(f"Will submit request for {id_list}")
        batch_queries_to_submit = []
        for current_id in id_list:
            query = self.aggregate_timeseries(enduses=enduses,
                                              group_by=[id_column] + group_by + ['time'],
                                              join_list=join_list,
                                              weights=['weight'],
                                              order_by=[id_column, 'time'],
                                              restrict=[(id_column, current_id)],
                                              run_async=True,
                                              get_query_only=True)
            batch_queries_to_submit.append(query)

        if get_query_only:
            return batch_queries_to_submit

        batch_query_id = self.submit_batch_query(batch_queries_to_submit)
        return self.get_batch_query_result(batch_id=batch_query_id)

    @staticmethod
    def get_eiaid_map(mapping_version, year=2018):
        if mapping_version == 1:
            map_table_name = 'eiaid_weights'
            map_baseline_column = 'build_existing_model.location'
            map_eiaid_column = 'location'
        elif mapping_version == 2:
            map_table_name = 'v2_eiaid_weights'
            map_baseline_column = 'build_existing_model.county'
            map_eiaid_column = 'county'
        elif mapping_version == 3:
            map_table_name = 'v3_eiaid_weights_%d' % year
            map_baseline_column = 'build_existing_model.county'
            map_eiaid_column = 'county'
        else:
            raise ValueError("Invalid mapping_version")

        return map_table_name, map_baseline_column, map_eiaid_column

    def aggregate_ts_by_eiaid(self, eiaid_list: List[str], enduses: List[str] = None, group_by: List[str] = None,
                              mapping_version=3, year=2018, get_query_only: bool = False):
        """
        Aggregates the timeseries result, grouping by utilities.
        Args:
            eiaid_list: The list of utility ids (EIAID) assigned by EIA.
            enduses: The list of enduses to aggregate
            group_by: Additional columns to group the aggregation by
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            year: The year of the EIA form 861 service territory to use.
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe with the aggregated timeseries and the requested enduses grouped by utilities
        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map(mapping_version, year)
        eiaid_list = [str(e) for e in eiaid_list]
        if not enduses:
            enduses = ['total_site_electricity_kwh']
        id_column = 'eiaid'

        return self._aggregate_ts_by_map(eiaid_map_table_name, map_baseline_column, map_eiaid_column, id_column,
                                         eiaid_list, enduses, group_by, get_query_only)

    def aggregate_unit_counts_by_eiaid(self, eiaid_list: List[str] = None, group_by: List[str] = None,
                                       mapping_version=3, year=2018, get_query_only: bool = False):
        """
        Returns the counts of the number of dwelling units, grouping by eiaid and other additional group_by columns if
        provided.
        Args:
            eiaid_list: The list of utility ids (EIAID) to aggregate for
            group_by: Additional columns to group by
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            year: The year of the EIA form 861 service territory to use.
            get_query_only: If set to true, returns the query instead of the result

        Returns:
            Pandas dataframe with the units counts
        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map(mapping_version, year)
        group_by = [] if group_by is None else group_by
        restrict = [('eiaid', eiaid_list)] if eiaid_list else None

        result = self.aggregate_annual([], group_by=['eiaid'] + group_by,
                                       order_by=['eiaid'] + group_by,
                                       join_list=[(eiaid_map_table_name, map_baseline_column, map_eiaid_column)],
                                       weights=['weight'],
                                       restrict=restrict,
                                       get_query_only=get_query_only)
        return result

    def aggregate_annual_by_eiaid(self, enduses: List[str], group_by: List[str] = None,
                                  mapping_version=3, year=2018, get_query_only: bool = False):
        """
        Aggregates the annual consumption in the baseline table, grouping by all the utilities
        Args:
            enduses: The list of enduses to aggregate
            group_by: Additional columns to group the aggregation by
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            year: The year of the EIA form 861 service territory to use.
            get_query_only: If set to true, returns the list of queries to run instead of the result.
        Returns:
            Pandas dataframe with the annual sum of the requested enduses, grouped by utilities
        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map(mapping_version, year)
        join_list = [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)]
        group_by = [] if group_by is None else group_by
        result = self.aggregate_annual(enduses=enduses, group_by=['eiaid'] + group_by,
                                       join_list=join_list,
                                       weights=['weight'],
                                       order_by=['eiaid'],
                                       get_query_only=get_query_only)
        return result

    def get_filtered_results_csv_by_eiaid(
            self, eiaids: List[str], mapping_version=3, year=2018, get_query_only: bool = False):
        """
        Returns a portion of the results csvs, which belongs to given list of utilities
        Args:
            eiaids: The eiaid list of utitlies
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            year: The year of the EIA form 861 service territory to use.
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe that is a subset of the results csv, that belongs to provided list of utilities
        """
        C = ResStockAthena.make_column_string
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map(mapping_version, year)
        eiaid_str = ','.join([f"'{e}'" for e in eiaids])
        query = f'''select * from {C(self.baseline_table_name)} join {C(eiaid_map_table_name)} on '''\
                f'''{C(map_baseline_column)} = "{map_eiaid_column}" where eiaid in ({eiaid_str}) and weight > 0 '''\
                f'''order by 1'''
        if get_query_only:
            return query
        return self.execute(query)

    def get_buildings_by_locations(self, locations: List[str], get_query_only: bool = False):
        """
        Returns the list of buildings belonging to given list of locations.
        Args:
            locations: list of `build_existing_model.location' strings
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the building ids belonging to the provided list of locations.

        """
        C = ResStockAthena.make_column_string
        locations_str = ','.join([f"'{a}'" for a in locations])
        query = f'''select building_id from {C(self.baseline_table_name)} where "build_existing_model.location" in ''' \
                f'''({locations_str}) order by building_id'''
        if get_query_only:
            return query
        res = self.execute(query)
        return res

    def get_buildings_by_eiaids(self, eiaids: List[str], mapping_version=3, year=2018, get_query_only: bool = False):
        """
        Returns the list of buildings belonging to the given list of utilities.
        Args:
            eiaids: list of utility EIAIDs
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            year: The year of the EIA form 861 service territory to use.
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the building ids belonging to the provided list of utilities.

        """
        C = ResStockAthena.make_column_string
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map(mapping_version, year)
        eiaid_str = ','.join([f"'{e}'" for e in eiaids])
        query = f"select distinct(building_id) from {C(self.baseline_table_name)} join {C(eiaid_map_table_name)}" \
                f" on {C(map_baseline_column)} = {C(map_eiaid_column)} where eiaid in ({eiaid_str}) and " \
                f" weight > 0 order by 1 "
        if get_query_only:
            return query
        res = self.execute(query)
        return res

    def get_locations_by_eiaids(self, eiaids: List[str], mapping_version=3, year=2018, get_query_only: bool = False):
        """
        Returns the list of locations/counties (depends on mapping version) belonging to a given list of utilities.
        Args:
            eiaids: list of utility EIAIDs
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            year: The year of the EIA form 861 service territory to use.
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the locations (for version 1) or counties (for version 2) belonging to the
            provided list of utilities.

        """
        C = ResStockAthena.make_column_string
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map(mapping_version, year)
        eiaid_str = ','.join([f"'{e}'" for e in eiaids])
        query = f"select distinct({C(map_eiaid_column)}) from {C(eiaid_map_table_name)} where weight > 0 and eiaid in" \
                f" ({eiaid_str}) order by 1"
        if get_query_only:
            return query
        res = self.execute(query)
        return res
