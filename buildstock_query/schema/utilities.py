from __future__ import annotations
from typing import Union, Any, Sequence
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy.sql.elements import Label, ColumnElement
from sqlalchemy.sql.selectable import Subquery

# from buildstock_query import BuildStockQuery  # can't import due to circular import


SACol = Union[sa.Column, ColumnElement]
SALabel = Label
DBColType = Union[SALabel,  SACol]
DBTableType = sa.Table
AnyTableType = Union[DBTableType, str, Subquery]


class MappedColumn(BaseModel):
    bsq: Any  # BuildStockQuery
    name: str
    mapping_dict: dict
    key: Union[Union[DBColType, str], Sequence[Union[DBColType, str]]]

    class Config:
        arbitrary_types_allowed = True
        smart_union = True


AnyColType = Union[DBColType, str, MappedColumn]
