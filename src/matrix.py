import numpy as np
import spectrum as sp
from scipy.stats import mode
from enumerations import Scale
from scan import get_spectra_list, get_spectra_dict
from output import scale_change

class Matrix():
    def __init__(self, spectra):
        self.spectra = spectra

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
        pass

    @classmethod
    def read_csv(self, path, scale_type=Scale.WAVENUMBERS):
        with open(path, 'r') as csv:
            scale = csv.readline().split(',')
            scale_type, *scale = scale
            if scale_type == Scale.WAVENUMBERS.value:
                f = float
            elif scale_type == Scale.WAVELENGTH_um.value:
                f = lambda x: 10_000. / float(x)
            else:
                f = lambda x: 10_000_000. / float(x)
            scale = np.array(list(map(f, scale)))
            while True:
                spc = csv.readline().strip()
                if len(spc) == 0:
                    break
                clss, *data = spc.split(',')
                data = np.array(list(map(float, data)))
                yield scale, data, clss

    def save_matrix(self, path='matrix.csv', scale_type=Scale.WAVENUMBERS):
        if not self.spectra:
            return
        sc = self.spectra[0]
        f = scale_change(scale_type)
        scale = list(map(f, sc.wavenums))
        with open(path, 'w') as out:
            print(scale_type.value, *scale, sep=',', file=out)
            for spc in self.spectra:
                print(spc.clss, *spc.data, sep=',', file=out)


if __name__ == '__main__':
    print('HI')
    spa = get_spectra_list(classify=True, recursive=True)
    mtr = Matrix(spa)
    # mtr.save_matrix()
    spa = Matrix.read_csv('matrix.csv')
    for _, _, clss in spa:
        print(clss)





