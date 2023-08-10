from concurrent.futures import Future
from pyathena.sqlalchemy.base import AthenaDialect
from pyathena.pandas.result_set import AthenaPandasResultSet
import datetime
import pickle
import os
import pandas as pd
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from buildstock_query.schema.utilities import MappedColumn  # noqa: F401


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
        return self.future.as_pandas()  # type: ignore # mypy doesn't know about AthenaPandasResultSet


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


class UnSupportedTypeException(Exception):
    pass


class CustomCompiler(AthenaDialect().statement_compiler):  # type: ignore

    @staticmethod
    def render_literal(obj):
        from buildstock_query.schema.utilities import MappedColumn  # noqa: F811
        if isinstance(obj, (int, float)):
            return str(obj)
        elif isinstance(obj, str):
            return "'%s'" % obj.replace("'", "''")
        elif isinstance(obj, (datetime.datetime)):
            return "timestamp '%s'" % str(obj).replace("'", "''")
        elif isinstance(obj, list):
            return CustomCompiler.get_array_string(obj)
        elif isinstance(obj, tuple):
            return f"({', '.join([CustomCompiler.render_literal(v) for v in obj])})"
        elif isinstance(obj, MappedColumn):
            keys = list(obj.mapping_dict.keys())
            values = list(obj.mapping_dict.values())
            if isinstance(obj.key, tuple):
                indexing_str = f"({', '.join(tuple(obj.bsq._compile(source) for source in obj.key))})"
            else:
                indexing_str = obj.bsq._compile(obj.key)

            return f"MAP({CustomCompiler.render_literal(keys)}, " +\
                   f"{CustomCompiler.render_literal(values)})[{indexing_str}]"
        else:
            raise UnSupportedTypeException(f"Unsupported type {type(obj)} for literal {obj}")

    @staticmethod
    def get_array_string(array):
        # rewrite to break into multiple arrays joined by CONCAT if the number of elements is > 254
        if len(array) > 254:
            array_list = ["ARRAY[" + ', '.join([CustomCompiler.render_literal(v) for v in array[i:i+254]]) + "]"
                          for i in range(0, len(array), 254)]
            return "CONCAT(" + ', '.join(array_list) + ")"
        else:
            return f"ARRAY[{', '.join([CustomCompiler.render_literal(v) for v in array])}]"

    def render_literal_value(self, obj, type_):
        from buildstock_query.schema.utilities import MappedColumn  # noqa: F811
        if isinstance(obj, (datetime.datetime, list, tuple, MappedColumn)):
            return CustomCompiler.render_literal(obj)

        return super(CustomCompiler, self).render_literal_value(obj, type_)


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
