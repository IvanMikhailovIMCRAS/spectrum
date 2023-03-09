import numpy as np
# from spectrum import Spectrum
from enumerations import Scale, BaseLineMode, NormMode
from scan import get_spectra_list, get_spectra_dict
from miscellaneous import scale_change
from smoothing import Smoother

class Matrix():
    def __init__(self, spectra):
        self.spectra = spectra
        
    @property
    def shape(self):
        if self.spectra:
            return len(self.spectra), len(self.spectra[0])
        return 0, 0

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
                
    def as_2d_array(self, predicate=lambda x: True):
        return np.vstack([spc.data for spc in self.spectra if predicate(spc)])


    def correlation(self):
        avgs = sum(self.spectra) * (1 / self.shape[0]) 
        res = np.zeros((self.shape[1], self.shape[1]))
        mtr = self.as_2d_array()
        for i in range(self.shape[1]):
            for j in range(self.shape[1]):
                res[i][j] = ((mtr[:, i] - avgs[i]) * (mtr[:, j] - avgs[j])).sum() / (
                np.sqrt(np.std(mtr[:, i]) * np.std(mtr[:, j])))

        return res
                




if __name__ == '__main__':
    print('HI')    
    spa = get_spectra_list(path='new_data', classify=True, recursive=False)
    spc = spa[2]
    # spc.correct_baseline(BaseLineMode.ZHANG)
    from output import show_spectra
    # print(spc)
    example = spc * 1
    example.clss = 'ex'
    pre_0der_conf = {
        'baseline': {'method': BaseLineMode.RB},
        'normalize': {'method': NormMode.MINMAX},
        'smooth': {'method': Smoother.moving_average, 'window_length': 13},
        #'derivative': {'n': 2}
    }
    mtr = Matrix.create_matrix(spa, {
        'baseline': {'method': BaseLineMode.RB},
        'normalize': {'method': NormMode.MINMAX},
        #'smooth': {'method': Smoother.moving_average, 'window_length': 13},
        'derivative': {'n': 2}
    })
    for i, spc in enumerate(mtr.spectra):
        mtr.spectra[i] = spc.range(1850., 600.)
    # spc.smooth(Smoother.moving_average, window_length=45)
    # show_spectra(mtr.spectra)
    # print(mtr.spectra) + [example]
    spc = mtr.spectra[0]
    import numpy as np
    import matplotlib.pyplot as plt
    from output import heatmap, auto_heatmap
    tp = 'healthy_131'
    
    def each_to_each(spc):
        mtr = []
        for i in range(len(spc)):
            mtr.append(np.roll(spc.data, i))
        return np.vstack(mtr)
    
    # corrcoefs = np.corrcoef(mtr.as_2d_array(lambda spc: spc.clss == tp))
    # corrcoefs = np.corrcoef(each_to_each(spc))
    # # corrcoefs = np.corrcoef(mtr.as_2d_array(lambda spc: spc.clss == 'epilepsy_71'))
    
    # print(corrcoefs.shape)
    # fig, clrbr, ax = heatmap(corrcoefs)
    # # plt.savefig(tp + '_heatmap.jpg')
    # plt.show()
    
    heatmap(mtr.correlation())


    
    
    





