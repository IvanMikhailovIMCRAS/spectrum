import numpy as np
# from spectrum import Spectrum
from enumerations import Scale, BaseLineMode, NormMode
from scan import get_spectra_list, get_spectra_dict
from miscellaneous import scale_change
from smoothing import Smoother

class Matrix():
    def __init__(self, spectra):
        self.spectra = spectra

    @classmethod
    def create_matrix(cls, raw_spectra,  config):
        # преобразование спектров других длин ?
        matrix = []
        if not raw_spectra:
            return
        for spc in raw_spectra:
            # spc.transform()
            if 'smooth' in config:
                spc.smooth(**config['smooth'])
            if 'baseline' in config:
                spc.correct_baseline(**config['baseline'])
            if 'normalize' in config:
                spc.normalize(**config['normalize'])
            if 'derivative' in config:
                spc.get_derivative(**config['derivative'])
            matrix.append(spc)
        return Matrix(matrix)

    @property
    def sample_spectrum(self):
        pass

    @classmethod
    def read_csv(cls, path, scale_type=Scale.WAVENUMBERS):
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
    spa = get_spectra_list(path='new_data', classify=True, recursive=False)
    spc = spa[2]
    # spc.correct_baseline(BaseLineMode.ZHANG)
    from output import show_spectra
    # print(spc)
    example = spc * 1
    example.clss = 'ex'
    mtr = Matrix.create_matrix(spa, {
        'baseline': {'method': BaseLineMode.ZHANG},
        'normalize': {'method': NormMode.VECTOR},
        'smooth': {'method': Smoother.moving_average, 'window_length': 13},
        #'derivative': {'n': 2}
    })
    # spc.smooth(Smoother.moving_average, window_length=45)
    # show_spectra(mtr.spectra)
    # print(mtr.spectra) + [example]
    mtr.save_matrix()





