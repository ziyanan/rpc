from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass
class Atomic:
    w: np.ndarray
    c: float
    feature_idx: int = 0
    relop: str = ">="

@dataclass
class STLNot:
    child: "Formula"

@dataclass
class STLAnd:
    left: "Formula"
    right: "Formula"

@dataclass
class STLOr:
    left: "Formula"
    right: "Formula"

@dataclass
class Always:
    child: "Formula"
    t_start: int
    t_end: int

@dataclass
class Eventually:
    child: "Formula"
    t_start: int
    t_end: int

@dataclass
class STLUntil:
    left: "Formula"
    right: "Formula"
    t_start: int
    t_end: int

Formula = Union[Atomic, STLNot, STLAnd, STLOr, Always, Eventually, STLUntil]

