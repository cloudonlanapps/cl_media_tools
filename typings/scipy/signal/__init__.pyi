from typing import Literal

import numpy as np
from numpy.typing import NDArray

_Mode = Literal["full", "valid", "same"]
_Boundary = Literal["fill", "wrap", "symm"]

def convolve2d(
    in1: NDArray[np.floating] | NDArray[np.integer],
    in2: NDArray[np.floating] | NDArray[np.integer],
    *,
    mode: _Mode = "full",
    boundary: _Boundary = "fill",
    fillvalue: float = 0.0,
) -> NDArray[np.floating]: ...
