from __future__ import annotations
from typing import Union, Any
from collections.abc import Sequence
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy.sql.elements import Label, ColumnElement
from sqlalchemy.sql.selectable import Subquery

# from buildstock_query import BuildStockQuery  # can't import due to circular import


SACol = sa.Column | ColumnElement
SALabel = Label
DBColType = SALabel | SACol
DBTableType = sa.Table | Subquery
AnyTableType = Union[DBTableType, str]


class MappedColumn(BaseModel):
    bsq: Any  # BuildStockQuery
    name: str
    mapping_dict: dict
    key: DBColType | str | Sequence[DBColType | str]

    class Config:
        arbitrary_types_allowed = True
        smart_union = True


AnyColType = DBColType | str | MappedColumn
