from pydantic import BaseModel, Field, Extra
from typing import Optional, Union, Callable
from collections.abc import Sequence
from typing import Literal
from buildstock_query.schema.utilities import AnyTableType, AnyColType
from pydantic import validator


class BaseQuery(BaseModel):
    enduses: Sequence[AnyColType]
    group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list)
    upgrade_id: str = "0"
    sort: bool = True
    join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list)
    restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list)
    avoid: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list)
    weights: Sequence[Union[str, tuple, AnyColType]] = Field(default_factory=list)
    get_quartiles: bool = False
    get_nonzero_count: bool = False
    get_query_only: bool = False
    limit: Optional[int] = None
    agg_func: Optional[Union[str, Callable]] = "sum"

    class Config:
        arbitrary_types_allowed = True
        smart_union = True
        extra = Extra.forbid


class TSQuery(BaseQuery):
    split_enduses: bool = False
    collapse_ts: bool = False
    timestamp_grouping_func: Optional[Literal["month", "day", "hour"]] = None


class SavingsQuery(TSQuery):
    annual_only: bool = True
    unload_to: str = ""
    partition_by: Sequence[str] = Field(default_factory=list)


class UtilityTSQuery(TSQuery):
    query_group_size: int = 20
    eiaid_list: Sequence[str]


class Query(BaseQuery):
    annual_only: bool = True
    include_savings: bool = False
    include_baseline: bool = False
    timestamp_grouping_func: Optional[Literal["year", "month", "day", "hour"]] = None
    partition_by: Sequence[str] = Field(default_factory=list)
    applied_only: Optional[bool] = False
    unload_to: Optional[str] = None

    # validate that include_savings is False if upgrade_id is '0'
    @classmethod
    @validator("include_savings")
    def validate_include_savings(cls, include_savings, values):
        if include_savings and values.get("upgrade_id") == "0":
            raise ValueError("include_savings cannot be True when upgrade_id is '0'")
        return include_savings

    # validate that annual_only is False if timestamp_grouping_func is not None
    @classmethod
    @validator("annual_only")
    def validate_annual_only(cls, annual_only, values):
        if values.get("timestamp_grouping_func") is not None and annual_only:
            raise ValueError("annual_only must be False when timestamp_grouping_func is provided")
        return annual_only

    # validate that applied_only is False if upgrade_id is '0'
    @classmethod
    @validator("applied_only")
    def validate_applied_only(cls, applied_only, values):
        if applied_only and values.get("upgrade_id") == "0":
            raise ValueError("applied_only cannot be set when upgrade_id is '0'")
        return applied_only

    # validate that include_baseline is False if upgrade_id is '0'
    @classmethod
    @validator("include_baseline")
    def validate_include_baseline(cls, include_baseline, values):
        if include_baseline and values.get("upgrade_id") == "0":
            raise ValueError("include_baseline cannot be set when upgrade_id is '0'")
        return include_baseline
