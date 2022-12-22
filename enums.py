from enum import Enum

class NormMode(Enum):
    VECTOR = 'vector'
    AREA = 'area'
    MINMAX = 'minmax'

class BaseLineMode(Enum):
    RB = 'rb'
    ALSS = 'alls'
    ZHANG = 'zhang'