from pydantic import BaseModel, Field
import sqlalchemy as sa
from typing import Union, Optional, Sequence
import random


from pydantic import BaseModel

class MyModel(BaseModel):
    val: list[str] | list[int]
    val2: str | int

    class Config:
        smart_union = True

m = MyModel(val=[1, 2, 3], val2=1)
print(m)
