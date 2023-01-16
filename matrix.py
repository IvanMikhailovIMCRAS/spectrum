import numpy as np
import spectrum as sp
from scipy.stats import mode

class Matrix():
    def __init__(self, spectra):
        self.spectra = spectra
        self.sample_spectrum = None

    @classmethod
    def create_matrix(cls, raw_spectra):
        # преобразование спектров других длин ?

        matrix = []
        if not raw_spectra:
            return np.zeros((1, 1))
        length, _ = mode([len(x)
                          for x in raw_spectra if len(x) != 0], keepdims=False)
        sample = np.zeros((length,))
        for raw_spectrum in raw_spectra:
            if raw_spectrum and len(raw_spectrum) == length:
                matrix.append(raw_spectrum.data)
                sample += raw_spectrum.data
            else:
                pass
        return Matrix(np.array(matrix))

    @property
    def sample_spectrum(self):






