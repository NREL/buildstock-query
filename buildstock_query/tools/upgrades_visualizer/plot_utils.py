from pydantic import BaseModel, Field
from enum import Enum


class SavingsTypes(str, Enum):
    absolute = "Absolute"
    savings = "Savings"
    percent_savings = "Percent Savings"


class ValueTypes(str, Enum):
    total = "total"
    count = "count"
    mean = "mean"
    distribution = "distribution"
    scatter = "scatter"


class PlotParams(BaseModel):
    enduses: list[str]
    savings_type: SavingsTypes
    change_type: str | None
    upgrade: int | None
    group_by: list[str] = Field(default_factory=list)
    filter_bldgs: list[int] = Field(default_factory=list)
    sync_upgrade: int | None = None
    resolution: str = "annual"
    value_type: ValueTypes
