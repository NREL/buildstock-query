
from concurrent.futures import Future
from pyathena.sqlalchemy_athena import AthenaDialect
from pyathena.pandas.result_set import AthenaPandasResultSet
import datetime
import pickle
import os
import pandas as pd
from typing import Literal


KWH2MBTU = 0.003412141633127942
MBTU2KWH = 293.0710701722222


class CachedFutureDf(Future):
    def __init__(self, df: pd.DataFrame, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.df = df
        self.set_result(self.df)

    def running(self) -> Literal[False]:
        return False

    def done(self) -> Literal[True]:
        return True

    def cancelled(self) -> Literal[False]:
        return False

    def result(self, timeout=None) -> pd.DataFrame:
        return self.df

    def as_pandas(self) -> pd.DataFrame:
        return self.df


class AthenaFutureDf:
    def __init__(self, db_future: Future) -> None:
        self.future = db_future

    def cancel(self) -> bool:
        return self.future.cancel()

    def running(self) -> bool:
        return self.future.running()

    def done(self) -> bool:
        return self.future.done()

    def cancelled(self) -> bool:
        return self.future.cancelled()

    def result(self, timeout=None) -> AthenaPandasResultSet:
        return self.future.result()

    def as_pandas(self) -> pd.DataFrame:
        return self.future.as_pandas()


class COLOR:
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'


def print_r(text):  # print in Red
    print(f"{COLOR.RED}{text}{COLOR.END}")


def print_y(text):  # print in Yellow
    print(f"{COLOR.YELLOW}{text}{COLOR.END}")


def print_g(text):  # print in Green
    print(f"{COLOR.GREEN}{text}{COLOR.END}")


class CustomCompiler(AthenaDialect().statement_compiler):  # type: ignore
    def render_literal_value(self, value, type_):
        if isinstance(value, (datetime.datetime)):
            return "timestamp '%s'" % str(value).replace("'", "''")
        if isinstance(value, list):
            return f"ARRAY[{','.join([str(v) for v in value])}]"
        return super(CustomCompiler, self).render_literal_value(value, type_)


class DataExistsException(Exception):
    def __init__(self, message, existing_data=None):
        super(DataExistsException, self).__init__(message)
        self.existing_data = existing_data


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found for loading table")
    with open(path, "rb") as f:
        return pickle.load(f)
