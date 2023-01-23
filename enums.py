from enum import Enum

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
