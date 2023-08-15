from buildstock_query.helpers import load_pickle, save_pickle
import pathlib
from sqlalchemy.exc import NoSuchTableError


folder_path = pathlib.Path(__file__).parent.resolve()


def load_cache_from_pkl(name):
    full_path = folder_path / "reference_files" / f"{name}_query_cache.pkl"
    return load_pickle(full_path)


def load_tbl_from_pkl(name, *args, **kwargs):
    folder_path = pathlib.Path(__file__).parent.resolve()
    full_path = folder_path / "reference_files" / f"{name}.pkl"
    try:
        return load_pickle(full_path)
    except FileNotFoundError as e:
        raise NoSuchTableError(f"Table {name} cannot be loaded from pickle.") from e


def mock_get_tables(self, table_name):
    ts_table = load_tbl_from_pkl(f"{table_name}_timeseries")
    bs_table = load_tbl_from_pkl(f"{table_name}_baseline")
    return ts_table, bs_table


def assert_query_equal(query1, query2):
    query1 = query1.strip().lower().split()
    query2 = query2.strip().lower().split()
    wrong_matches = [(indx, word, query2[indx]) for indx, word in enumerate(query1)
                     if word != query2[indx]]
    assert not wrong_matches


def save_ref_pkl(name, obj):
    folder_path = pathlib.Path(__file__).parent.resolve()
    save_pickle(folder_path / f"reference_files/{name}.pkl", obj)
    print(f"Saved {name}.pkl")
