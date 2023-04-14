from typing import Optional, Union
from typing import Literal
from pydantic import BaseModel


class RunParams(BaseModel):
    workgroup: str
    db_name: str
    table_name: Union[str, tuple[str, Optional[str], Optional[str]]]
    buildstock_type: Literal['resstock', 'comstock'] = 'resstock'
    timestamp_column_name: str = 'time'
    building_id_column_name: str = 'building_id'
    sample_weight: Union[str, int, float] = "build_existing_model.sample_weight"
    region_name: str = 'us-west-2'
    execution_history: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        smart_union = True


class BSQParams(RunParams):
    skip_reports: bool = False

    def get_run_params(self):
        return RunParams.parse_obj(self.dict(include=set(RunParams.__fields__.keys())))