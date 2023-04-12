from pydantic import BaseModel, Field
from typing import Optional, Union, Sequence
import sqlalchemy as sa
from typing import Literal, Any

SACol = sa.Column[Any]
SALabel = sa.sql.expression.Label[Any]  # type: ignore
DBColType = Union[SALabel,  SACol]
DBTableType = sa.Table
AnyColType = Union[DBColType, str]
AnyTableType = Union[DBTableType, str]


class AnnualQuery(BaseModel):
    enduses:  Sequence[str]
    group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list)
    upgrade_id: str = '0'
    sort: bool = True
    join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list)
    restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list)
    weights: Sequence[Union[str, tuple]] = Field(default_factory=list)
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
    partition_by: Sequence[str] = Field(default_factory=list)


class UtilityTSQuery(TSQuery):
    query_group_size: int = 20
    eiaid_list: Sequence[str]
