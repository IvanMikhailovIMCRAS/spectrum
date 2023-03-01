from enum import Enum
from numpy import sum, sqrt, log2, mean, abs, square
class NormMode(Enum):
    VECTOR = 'vector'
    AREA = 'area'
    MINMAX = 'minmax'

class BaseLineMode(Enum):
    RB = 'rb'
    ALSS = 'alls'
    ZHANG = 'zhang'
    
class Scale(Enum):
    WAVENUMBERS = 'Wavenumber cm-1'
    WAVELENGTH_nm = 'Wavelength nm'
    WAVELENGTH_um = 'Wavelength um'

class LossFunc(Enum):
    RMSE = lambda y, y0: sqrt(sum(square(y - y0)))
    BINARY = lambda y, y0: sum(y == y0)
    MAE = lambda y, y0: mean(abs(y - y0))
    LOG2 = lambda y, y0:  log2(sum(abs(y - y0)) + 1)

class Smooth(Enum):
    LINEAR = 'linear'
    CUBIC_SPLINE = 'cubic'
    HERMITE = 'hermite'
