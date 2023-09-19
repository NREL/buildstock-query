from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


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
    change_type: Optional[str]
    upgrade: Optional[int]
    group_by: list[str] = Field(default_factory=list)
    filter_bldgs: list[int] = Field(default_factory=list)
    sync_upgrade: Optional[int] = None
    resolution: str = "annual"
    value_type: ValueTypes
