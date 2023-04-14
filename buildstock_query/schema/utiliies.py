from pydantic import BaseModel
from buildstock_query.schema.query_params import AnyColType
from typing import Union
from buildstock_query import BuildStockQuery


class MappedColumn(BaseModel):
    bsq: BuildStockQuery
    name: str
    mapping_dict: dict
    key: Union[AnyColType, tuple[AnyColType, ...]]

    class Config:
        arbitrary_types_allowed = True
        smart_union = True
