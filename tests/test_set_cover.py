import pathlib
import numpy as np
from buildstock_query.tools.set_cover import SetCoverSolver
import pytest
import pandas as pd
from unittest.mock import patch

class TestSetCoverSolver:
    def test_greedy_hitting_set(self):

        # Test case: Basic functionality
        building_groups = [
            {1, 2, 3, 8},  # Group 1
            {2, 4, 5},  # Group 2
            {3, 5, 6},  # Group 3
            {6, 7, 8},  # Group 4
            {1, 8, 9},  # Group 5
        ]

        solver = SetCoverSolver(groups=building_groups)

        assert [8, 5] == solver.get_greedy_hitting_set()
        # The preffered items doesn't influence tie-breaking and the final result
        assert [8, 5] == solver.get_greedy_hitting_set(current_list=[1, 5, 7])

        # Test case: Empty input
        solver = SetCoverSolver(groups=[])
        assert solver.get_greedy_hitting_set() == []

        # Test case: Input with empty sets (empty sets are ignored)
        building_groups_with_empty = [set(), {1, 2}, set(), {3, 4}]
        solver = SetCoverSolver(groups=building_groups_with_empty)
        assert [4, 2] == solver.get_greedy_hitting_set()

        # Test case: Disjoint sets requiring multiple buildings
        disjoint_groups = [
            {1, 2},
            {3, 4},
            {5, 6},
            {7, 8},
            {9, 10},
        ]
        solver = SetCoverSolver(groups=disjoint_groups)
        assert [10, 8, 6, 4, 2] == solver.get_greedy_hitting_set()
        # Algorithm should honor the preferred items when feasible
        assert [10, 8, 6, 4, 2] == solver.get_greedy_hitting_set(current_list=[10, 8, 6, 4, 2])
        assert [10, 8, 6, 3, 1] == solver.get_greedy_hitting_set(current_list=[10, 8, 6, 3, 1])
        assert [1, 3, 5, 7, 9] == solver.get_greedy_hitting_set(current_list=[1, 3, 5, 7, 9])
        assert [1, 3, 5, 8, 10] == solver.get_greedy_hitting_set(current_list=[1, 3, 5, 8, 10])
        # Even if the preferred items are not feasible, the algorithm should still find a minimal hitting set
        assert [1, 3, 5, 10, 8] == solver.get_greedy_hitting_set(current_list=[1, 2, 3, 4, 5])

        # Test case where greedy algorithm is not optimal
        building_groups = [
            {1, 10, 4},
            {1, 9, 4},
            {1, 8},
            {2, 7},
            {2, 6, 4},
            {2, 5, 4},
        ]
        solver = SetCoverSolver(groups=building_groups)
        assert [4, 2, 1] == solver.get_greedy_hitting_set()  # [2, 1] would have been enough
    
    def test_refine_minimal_set(self):
        """"""
        groups = [
            {1, 2, 3, 8},  # Group 1
            {2, 4, 5},  # Group 2
            {3, 5, 6},  # Group 3
            {6, 7, 8},  # Group 4
            {1, 8, 9},  # Group 5
        ]
        solver = SetCoverSolver(groups=groups)
        # Starting list has extra items; algorithm should pare it down to the minimal hitting set
        assert [8, 5] == solver.refine_minimal_set(current_list=[8, 5, 4])
        
        # Starting list is already minimal – nothing should change
        assert [8, 5] == solver.refine_minimal_set(current_list=[8, 5])

        # Starting list covers only some groups; refine should add as few items as needed
        assert [8, 5] == solver.refine_minimal_set(current_list=[8])
        assert [2, 8, 6] == solver.refine_minimal_set(current_list=[2])
        assert [1, 7, 5] == solver.refine_minimal_set(current_list=[1, 7])
        assert [3, 8, 5] == solver.refine_minimal_set(current_list=[3, 8])
        assert [1, 2, 6] == solver.refine_minimal_set(current_list=[1, 2, 3, 6])

        # Empty starting list should raise an error
        with pytest.raises(ValueError):
            solver.refine_minimal_set([])
    
    def test_find_minimal_set(self):
        """Test finding minimal set using both greedy and refined algorithms."""
        # Test case where refine does better than greedy
        groups = [
            {1, 10, 4},
            {1, 9, 4},
            {1, 8},
            {2, 7},
            {2, 6, 4},
            {2, 5, 4},
        ]
        solver = SetCoverSolver(groups=groups)
        assert [4, 2, 1] == solver.get_greedy_hitting_set(current_list=[2])
        assert [2, 1] == solver.refine_minimal_set(current_list=[2])
        assert [4, 2, 1] == solver.find_minimal_set()
        assert [2, 1] == solver.find_minimal_set(current_list=[2])

        # Test case where greedy does better than refine
        # ------------------------------------------------
        # Here `current_list` starts with an item that does **not** appear in any group.  The
        # refine-based approach will therefore need to *add* every item found by the greedy
        # algorithm and will end up longer, so `find_minimal_set` must prefer the greedy
        # solution.
        groups = [{1}, {2}]
        solver = SetCoverSolver(groups=groups)
        # Sanity-check helpers return what we expect
        assert [2, 1] == solver.get_greedy_hitting_set()  # no preferred list
        assert [3, 2, 1] == solver.refine_minimal_set(current_list=[3])
        # `find_minimal_set` should therefore return the shorter greedy result
        assert [2, 1] == solver.find_minimal_set(current_list=[3])

        # Test case where both algorithms return the same *length* but different items
        # ---------------------------------------------------------------------------
        # There are multiple optimal 2-item covers.  Greedy picks one based on its own
        # heuristics; refine preserves the order of the supplied list.  Because the lengths
        # are the same, `find_minimal_set` should prefer the refined result.
        groups = [{1, 2}, {3, 4}]
        solver = SetCoverSolver(groups=groups)
        greedy_res = solver.get_greedy_hitting_set()             # expected [4, 2]
        refined_res = solver.refine_minimal_set(current_list=[1, 3])  # [1, 3]
        assert len(greedy_res) == len(refined_res) == 2  # both minimal-length covers
        # `find_minimal_set` should follow the refined result when lengths tie
        assert refined_res == solver.find_minimal_set(current_list=[1, 3])

        # Test case where no `current_list` is provided – defaults to greedy algorithm
        # ---------------------------------------------------------------------------
        groups = [{1, 2, 3}, {3, 4, 5}, {5, 6}]
        solver = SetCoverSolver(groups=groups)
        assert solver.find_minimal_set() == solver.get_greedy_hitting_set()


    # --- Additional edge-case tests ----------------------------------------------------

    def test_all_common_item(self):
        """All groups share a common item - minimal set should be that single item."""
        groups = [{1, 2}, {1, 3}, {1, 4}]
        solver = SetCoverSolver(groups=groups)
        assert solver.get_greedy_hitting_set() == [1]
        assert solver.refine_minimal_set(current_list=[1]) == [1]

    def test_duplicate_items_in_group(self):
        """Duplicates inside an input group must not affect the result."""
        groups = [{1, 1, 2}, {2, 3}]  # duplicates collapse when turned into set
        solver = SetCoverSolver(groups=groups)
        assert solver.get_greedy_hitting_set() == [2]

    def test_preferred_already_covers(self):
        """If preferred list already hits all groups, algorithm should immediately return from it."""
        groups = [{8, 4}, {5, 6}]
        solver = SetCoverSolver(groups=groups)
        assert solver.get_greedy_hitting_set(current_list=[8, 5]) == [8, 5]

    def test_preferred_with_nonexistent_items(self):
        """Non-existent preferred items should be ignored, algorithm still finds minimal set."""
        groups = [{8, 4}, {5, 6}]
        solver = SetCoverSolver(groups=groups)
        assert solver.get_greedy_hitting_set(current_list=[8, 44, 53]) == [8, 6]

    def test_single_large_group(self):
        """One large group should be solved with a single picked item."""
        universe = list(range(1, 101))
        groups = [set(universe)]
        solver = SetCoverSolver(groups=groups)
        res = solver.get_greedy_hitting_set()
        assert len(res) == 1 and res[0] in universe

    def test_minimal_set_empty_groups(self):
        """An empty list of groups should result in an empty minimal set regardless of method."""
        solver = SetCoverSolver(groups=[])
        assert solver.find_minimal_set() == []

    def test_minimal_set_duplicate_preferred(self):
        """`current_list` may contain duplicate items; they must be handled gracefully."""
        groups = [{1, 2}, {2, 3}]
        solver = SetCoverSolver(groups=groups)
        # The duplicates in preferred list should be ignored but 2 already covers both groups.
        assert [2] == solver.find_minimal_set(current_list=[2, 2, 2])

    def test_tie_break_stability(self):
        """When multiple items tie, algorithm still returns a single valid item deterministically."""
        groups = [[2, 1], [1, 2]]
        solver = SetCoverSolver(groups=groups)
        # Both 2 and 1 is valid, but due initial sorting of the groups, 2 should be returned
        assert [2] == solver.get_greedy_hitting_set()