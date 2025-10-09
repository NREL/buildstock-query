from typing import Optional, Union
from typing import Literal
from pydantic import BaseModel


class RunParams(BaseModel):
    workgroup: str
    db_name: str
    table_name: Union[str, tuple[str, Optional[str], Optional[str]]]
    buildstock_type: Literal['resstock', 'comstock'] = 'resstock'
    db_schema: Optional[str] = None
    sample_weight_override: Optional[Union[int, float]] = None
    region_name: str = 'us-west-2'
    execution_history: Optional[str] = None
    cache_folder: str = '.bsq_cache'
    athena_query_reuse: bool = True
    metadata_table_suffix: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        smart_union = True


class BSQParams(RunParams):
    skip_reports: bool = False

    def get_run_params(self):
        return RunParams.parse_obj(self.dict(include=set(RunParams.__fields__.keys())))
