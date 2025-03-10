from pydantic import BaseModel
from typing import Optional

class TableSuffix(BaseModel):
    baseline: str
    timeseries: str
    upgrades: str


class ColumnPrefix(BaseModel):
    characteristics: str
    output: str


class ColumnNames(BaseModel):
    building_id: str
    sample_weight: str
    sqft: str
    timestamp: str
    completed_status: str
    unmet_hours_cooling_hr: str
    unmet_hours_heating_hr: str
    map_eiaid_column: Optional[str] = None
    fuel_totals: list[str]


class CompletionValues(BaseModel):
    success: str
    fail: str
    unapplicable: str


class Structure(BaseModel):
    # whether the baseline timeseries is copied for unapplicable buildings in an upgrade
    unapplicables_have_ts: bool


class DBSchema(BaseModel):
    table_suffix: TableSuffix
    column_prefix: ColumnPrefix
    column_names: ColumnNames
    completion_values: CompletionValues
    structure: Structure
