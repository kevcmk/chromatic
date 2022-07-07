
import numpy as np
from typing import Generic, TypeVar

Shape = TypeVar("Shape")
DataType = TypeVar("DataType")

class NDArray(np.ndarray, Generic[Shape, DataType]):
    pass
