from pydantic import BaseModel, Field
from typing import Optional, Union, Sequence
from typing import Literal
from buildstock_query.schema.utilities import AnyTableType, AnyColType


class AnnualQuery(BaseModel):
    enduses:  Sequence[AnyColType]
    group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list)
    upgrade_id: str = '0'
    sort: bool = True
    join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list)
    restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list)
    weights: Sequence[Union[str, tuple, AnyColType]] = Field(default_factory=list)
    get_quartiles: bool = False
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
    partition_by: Sequence[str] = Field(default_factory=list)


class UtilityTSQuery(TSQuery):
    query_group_size: int = 20
    eiaid_list: Sequence[str]
