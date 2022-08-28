import array
from multiprocessing.dummy import Array
import numpy
from pydantic import BaseModel
class Image(BaseModel):
    url: str
    