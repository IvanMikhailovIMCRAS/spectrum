from sklearn.svm import SVC
from sklearn.linear_model import SGDOneClassSVM
import numpy as np
from matrix import Matrix
from enumerations import Scale
import pandas as pd
from scipy.stats import expon, mode, geom, norm, uniform
from miscellaneous import load_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from deconvolution import Deconvolutor
from spectrum import Spectrum
from smoothing import Smoother


class Darwin:
    fitness_functions = []
    epsilon = 0.01
    expon_scale = 2
    additional_transform = lambda x: x  # lambda x: np.exp(-x) if x > 0 else -x**2
    mutation_area_limits = (10, 26)
    norm_params = {
        'mus': 0.0001,
        'widths': 0.0002,
        'amps': 0.0005
    }
    uniform_params = {
        'mus': 0.005,
        'widths': 0.02,
        'amps': 0.01
    }


    def __init__(self, estimator_path='../tmp/darwin_estimator.pkl', proba_distr=None):
        self._estimator = load_model(estimator_path)
        self._pca = PCA(30)
        self._separator = SVC()
        self.fitness = None
        self.current_population = None
        self.current_target = None
        self.offspring = None
        self.limit = 1000
        self.fitted = False
        self.scale = None
        self.veclen = 0
        self.target_mapping = lambda x: -1 if 'heal' in x else 1
        self.proba_distr = proba_distr if proba_distr else expon(loc=0, scale=3)
        # self.mutation_distr = geom(0.2)
        self.peak_lability = norm()

    @staticmethod
    def margin_accuracy(margins):
        return np.round(sum(1 for i in margins if i > 0) / len(margins), 4)

    def get_margins(self, need_fit=False):

        X, y = self.current_population, self.current_target
        X = self._pca.fit_transform(X)
        if not self.fitted or need_fit:
            self._separator.fit(X, y)
            self.fitted = True

        return self._separator.decision_function(X) * y

    def download_population(self, path):
        population = pd.read_csv(path)
        y = population.pop(population.columns[0])
        y = y.apply(self.target_mapping)
        self.current_population = population
        self.current_target = y
        self.scale = population.columns.astype(float)
        self.veclen = len(self.scale)
    chokes = []

    @classmethod
    def breed(cls, f_body, s_body):
        choke_points = f_body - s_body
        indices = [i for i in range(len(choke_points)) if abs(choke_points[i]) < cls.epsilon]
        times = min(len(indices), int(np.ceil(expon.rvs(loc=1, scale=cls.expon_scale))))
        for _ in range(times):
            ind = np.random.randint(0, len(indices))
            choke = indices[ind]
            cls.chokes.append(choke)
            f_body = np.concatenate([f_body[:choke], s_body[choke:]])
            s_body = np.concatenate([s_body[:choke], f_body[choke:]])
            del indices[ind]
        return s_body, f_body

    def __select_to_breed(self, distr_generator, max_ind=-1, **kwargs):
        if max_ind == -1:
            max_ind = len(self.current_population) - 1
        while True:
            f, s = np.round(distr_generator.rvs(size=2, **kwargs))
            if f <= max_ind and s <= max_ind:
                break
        return int(f), int(s)

    def __sort_by_fitness(self):
        self.current_population['fitness'] = pd.Series(self.get_margins())
        self.current_population.sort_values(['fitness'], ascending=False, inplace=True)
        self.current_target = pd.Series([self.current_target[i] for i in self.current_population.index])
        self.current_population.drop(['fitness'], axis=1, inplace=True)

    def check(self):
        # plt.plot(self.current_population.index)
        self.__sort_by_fitness()
        # plt.plot(self.current_population.index)
        plt.show()

    def form_generation(self, gen_size):
        assert gen_size % 2 == 0, 'Please, let the gen_size be even!'
        ng = []
        new_labels = []
        self.__sort_by_fitness()
        for _ in tqdm(range(gen_size)):
            f, s = self.__select_to_breed(self.proba_distr)
            newf, news = self.breed(self.current_population.iloc[f, :], self.current_population.iloc[s, :])
            ng.append(newf)
            ng.append(news)
            new_labels.extend(self._estimator.predict(np.vstack([newf, news])))
        self.offspring = pd.DataFrame(ng, columns=self.current_population.columns)
        print(pd.Series(new_labels).value_counts())
        # for ind in self.offspring.index:
        #     plt.plot(self.offspring.loc[ind, :])
        # plt.show()

    def smooth_mutate(self):
        pass

    def __peak_mutate(self, spc, hwl = 8):
        d = Deconvolutor(spc)
        orig = spc * 1

        # define the mutation region
        wavenums = d.peaks_by_2der()[0]
        i = np.random.randint(0, len(wavenums))
        while True:
            j = np.random.randint(i - self.mutation_area_limits[1], i + 1 + self.mutation_area_limits[1])
            if abs(j - i) > self.mutation_area_limits[0]:
                break
        tmp_spc = spc.range(wavenums[i], wavenums[j])
        start = tmp_spc[0][0]
        length = len(tmp_spc)
        # deconvolute the region
        d.spectrum = tmp_spc
        peaks, params = d.deconvolute([
            ('voi', 'amps', 'mus'),
            ('voi', 'mus'),
            ('amps', 'voi',),
            ('mus', 'voi',)
        ])
        peaks.sort(key=lambda x: x[1])
        peaks = peaks[1:-1]
        deconvoluted_band = Spectrum(orig.wavenums, peaks=peaks)
        # deconvoluted_band.smooth(Smoother.moving_average, window_length=2 * hwl + 1)

        from output import show_curve_approx
        show_curve_approx(Spectrum(tmp_spc.wavenums, peaks=peaks), peaks)
        _, wavenums = tmp_spc.get_extrema(locals=True, minima=True, include_edges=True)
        # form new region
        cnt = 0
        for i in range(int(geom.rvs(p=0.9))):
            cnt += 1
            ind = np.random.randint(0, len(peaks))
            peaks[ind] = self.__change_band(peaks[ind])
        print('Mutatuions: ', cnt)
        return (start, length), deconvoluted_band, Spectrum(wavenums=orig.wavenums, peaks=peaks)

    def mutate(self, data, hwl=8):
        from output import show_spectra, show_curve_approx
        spc = Spectrum(self.scale, data)
        orig = spc * 1
        (start, length), deconvoluted_band, reconstructed = self.__peak_mutate(spc, hwl)
        # reconstructed.smooth(Smoother.moving_average, window_length=2 * hwl + 1)
        # reconstructed = reconstructed.range(reconstructed.wavenums[hwl + 1], reconstructed.wavenums[-hwl - 1])
        #
        for i, (w, _) in enumerate(spc):
            if abs(w - start) < Darwin.epsilon:
                break
        # self.__correct_linearly(spc, reconstructed, i)
        
        spc -= deconvoluted_band
        spc += reconstructed
        
        show_spectra([deconvoluted_band, reconstructed])
        
        # spc.smooth(rangeind=(i, i + length ), window_length=2*hwl + 1)
        # spc.smooth(rangeind=(i, i + length ), window_length=2*hwl + 1)

        # for j, intensity in enumerate(reconstructed.data):
            # spc.data[i + j] = intensity

        show_spectra([spc, orig])

    @staticmethod
    def __change_band(band):
        # band = list(band)
        # ind = np.random.randint(0, 3)
        # sigma2 = band[ind] * Darwin.norm_params[Deconvolutor.vseq[ind]]
        # print('SIGMA2: ', sigma2)
        # noise = norm.rvs(0, sigma2)
        # print('NOISE: ', noise)
        # band[ind] += noise
        # print(Deconvolutor.vseq[ind])
        # return tuple(band)
        band = list(band)
        ind = np.random.randint(1, 3)
        rng = (band[ind] * (1 - Darwin.uniform_params[Deconvolutor.vseq[ind]]), 
               band[ind] * (1 + Darwin.uniform_params[Deconvolutor.vseq[ind]]))
        print('RANGE: ', rng)
        noise = uniform.rvs(rng[0], rng[1] - rng[0])
        print('NOISE: ', noise)
        band[ind] = noise
        print(Deconvolutor.vseq[ind])
        return tuple(band)

    @staticmethod
    def __correct_linearly(spc, recon, start):
        v = spc[start][1] - recon[0][1]
        recon += v
        v = recon[-1][1] - spc[start + len(recon) - 1][1]
        a = v / (len(recon) - 1)
        dely = np.arange(0, len(recon)) * a
        # dely = np.arange(len(recon), 0, -1) * a
        recon -= dely



if __name__ == '__main__':
    print('SYNTHESIS')
    d = Darwin('tmp/darwin_estimator.pkl')
    d.download_population('tmp/EH_preproc.csv')
    d.mutate(d.current_population.iloc[0, :].to_numpy())
    # d.check()
    # d.form_generation(1000)
    # plt.hist(Darwin.chokes, bins=100)
    # plt.show()










