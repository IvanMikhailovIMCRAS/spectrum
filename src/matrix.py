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

    def corr_half(self, predicate):
        C = 4
        r = self.corr(predicate) + C
        r = np.triu(r, 1) - C
        return np.array(list(filter(lambda x: x >= -1, r.flatten())))

    # def correlation(self):
    #     avgs = sum(self.spectra) * (1 / self.shape[0])
    #     res = np.zeros((self.shape[1], self.shape[1]))
    #     mtr = self.as_2d_array()
    #     print(mtr.shape, len(avgs))
    #     def corr(spc):
    #         for i in range(self.shape[1]):
    #             for j in range(self.shape[1]):
    #                 res[i][j] = ((mtr[:, i] - avgs[i])
    #                              *
    #                              (mtr[:, j] - avgs[j])).sum() / (
    #                 np.sqrt(np.std(mtr[:, i]) * np.std(mtr[:, j])))
    #         return res

    def average_by_point(self):
        res = np.zeros(self.shape[1])
        for spc in self.spectra:
            res += spc.data
        return res / self.shape[0]

    def corr(self, predicate):
        import pandas as pd
        mtr = self.as_2d_array(predicate)
        # mtr = np.vstack([mtr[0], mtr[0]])
        mtr = pd.DataFrame(mtr)
        # print('SAMPLE: \n', mtr.sample())
        # print('SHAPE: ', mtr.shape)
        return mtr.corr().to_numpy()
    
    def one_spectrum(self, spc):
        avgs = self.average_by_point()
        res = np.zeros(shape=(len(spc), len(spc)))
        y = spc.data
        for i in range(len(spc)):
            for j in range(len(spc)):
                res[i][j] = (y[i] -  avgs[i]) * (y[j] - avgs[j])
        return res
    
    def get_stat_matrix(self):
        mtr = self.as_2d_array()
        res = np.zeros(shape=(self.shape[1], self.shape[1]))
        avgs = self.average_by_point()
        res_std = res.copy()
        for i in range(res.shape[0]):
            for j in range(res.shape[0]):
                distr = -(mtr[:, i] - mtr[:, j])
                res[i][j] = distr.mean()
                res_std[i][j] = distr.std()
            # res[i, :] += avgs[i]
        return {
            'mean': res,
            'std': res_std
        }

    def get_random_values(self, correls, d):
        # d = self.get_stat_matrix()
        stds = d['std']
        means = d['mean']
        size = stds.shape[0]
        res = np.zeros(shape=(size,))
        # tmp = np.zeros(shape=(size, size))
        # for i in range(size):
        #     for j in range(size):
        #         tmp[i][j] = np.random.normal(means[i, j], stds[i, j]) 
        for i in range(size):
            res[i] = np.random.normal(
                (means[:, i] * np.abs(correls[:, i])).sum() / np.abs(correls[:, i]).sum(),
                stds[:, i].mean()
                #(stds[:, i] * np.abs(correls[:, i])).sum() / np.abs(correls[:, i]).sum()
                ) #(tmp[:, i] * np.abs(correls[:, i])).sum() / np.abs(correls[:, i]).sum()
        return res
                
                
        

if __name__ == '__main__':
    print('MATRIX')
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
    spa = [spc for spc in spa if 'heal' in spc.clss ]
    mtr = Matrix.create_matrix(spa, {
        'baseline': {'method': BaseLineMode.RB},
        'normalize': {'method': NormMode.MINMAX},
        #'smooth': {'method': Smoother.moving_average, 'window_length': 13},
        # 'derivative': {'n': 2}
    })
    for i, spc in enumerate(mtr.spectra):
        mtr.spectra[i] = spc.range(1850., 600.)
    

    from output import heatmap
    import matplotlib.pyplot as plt
    from spectrum import Spectrum
    
    correls = mtr.corr(lambda x: True)
    d = mtr.get_stat_matrix()
    art_spa_wiener = []
    art_spa_movage = []
    for i in range(10):
        spc_art1 = Spectrum(mtr.spectra[0].wavenums, mtr.get_random_values(correls, d)) 
        spc_art2 = spc_art1 * 1  
        spc_art1.smooth(Smoother.wiener)
        art_spa_wiener.append(spc_art1)
        spc_art2.smooth(Smoother.moving_average, window_length=17)
        art_spa_movage.append(spc_art2)
    show_spectra(art_spa_wiener)
    show_spectra(art_spa_movage)
    # spc_art2 = Spectrum(mtr.spectra[0].wavenums, mtr.get_random_values(correls))
    # avg_spec = Spectrum(mtr.spectra[0].wavenums, mtr.average_by_point())
    
    # spc_diff = spc_art2 - spc_art1
    # data = mtr.get_random_values()
    # show_spectra([spc, spc_diff, spc_art1, spc_art2, avg_spec]) 
    # plt.plot(data)
    # plt.show()
    
    
    # heatmap()
    # plt.imshow(mtr.corr(lambda x: 'heal' in x.clss))
    # plt.show()
    # res = np.linalg.eigvals(mtr.one_spectrum(mtr.spectra[0])).real[:10]
    # print(res)
    # show_spectra([Spectrum(np.arange(len(res)), res)])


    
    
    





