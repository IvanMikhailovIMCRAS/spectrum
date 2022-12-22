import numpy as np
import spectrum as sp
from scipy.stats import mode

class Matrix():
    def __init__(self, values):
        self.vals = values

    @classmethod
    def create_matrix(cls, raw_spectra):
        # преобразование спектров других длин ?

        matrix = []
        if not raw_spectra:
            return np.zeros((1, 1))
        length, _ = mode([len(x)
                          for x in raw_spectra if len(x) != 0], keepdims=False)
        for raw_spectrum in raw_spectra:
            if raw_spectrum and len(raw_spectrum) == length:
                matrix.append(raw_spectrum.data)
            else:
                pass
        return Matrix(np.array(matrix))



