from pydantic import BaseModel, Field
from typing import Optional, Union
from functools import wraps
import sqlalchemy as sa
from typing import Literal


class AnnualQuery(BaseModel):
    enduses:  Optional[list[str]]
    group_by: Optional[list[sa.sql.elements.Label | sa.Column | str | tuple[str, str]]] = Field(default_factory=list)
    upgrade_id: Optional[int] = None
    sort: bool = False
    join_list: Optional[list[tuple[str, str, str]]] = Field(default_factory=list)
    restrict: Optional[list[tuple[str, str | int | list]]] = Field(default_factory=list)
    weights: Optional[list[Union[str, tuple]]] = Field(default_factory=list)
    get_quartiles: bool = False
    run_async: bool = False
    get_query_only: bool = False
    limit: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
        smart_union = True


class TSQuery(AnnualQuery):
    split_enduses: bool = False
    collapse_ts: bool = False
    timestamp_grouping_func: Optional[Literal["month", "day", "hour"]] = None


class SavingsQuery(TSQuery):
    annual_only: bool = True
    applied_only: bool = False
    unload_to: str = ''
    partition_by: Optional[list[str]] = None


def accept_query(querycls):
    def _accept_query(func):
        @wraps(func)
        def inner(self, **kwargs):
            query_obj = querycls.parse_obj(kwargs)
            return func(self, query_params=query_obj)
        return inner
    return _accept_query
