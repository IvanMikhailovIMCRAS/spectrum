from enum import Enum
import numpy as np

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
    RMSE = lambda y, y0: np.sqrt(np.sum(np.square(y - y0)))
    BINARY = lambda y, y0: np.sum(y == y0)
    MAE = lambda y, y0: np.mean(np.abs(y - y0))
