import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from spectrum import Spectrum
from output import show_spectra
from enumerations import LossFunc
from random import random
import statsmodels.api as sm

width_sigma = 2 * np.sqrt(np.log(2))  # * np.sqrt(2)
width_lambda = 2.


# def gauss(x, amp, mu, sigma):
#     return amp / sigma / np.sqrt(2.*np.pi) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def gauss(x, amp, mu, w):
    sigma = w / width_sigma
    return amp * np.exp(-np.square((x - mu) / sigma))


def lorentz(x, amp, x0, w):
    return amp / (np.square(2 * (x - x0) / w) + 1.)


def voigt(x, amp, x0, w, gauss_prop):
    assert 0 <= gauss_prop <= 1
    return gauss_prop * gauss(x, amp, x0, w) + (1 - gauss_prop) * lorentz(x, amp, x0, w)


class Deconvolutor:
    def __init__(self, spectrum, loss_func=LossFunc.RMSE, threshold=0.01):
        self.spectrum = spectrum * 1
        self.threshold = threshold
        self.minima = None
        self.loss_func = loss_func

    @staticmethod
    def __split_params(v):
        assert len(v) % 3 == 0
        assert len(v) != 0
        third = len(v) // 3
        ams, mus, sigmas = v[:third], v[third: 2 * third], v[2 * third:]
        return ams, mus, sigmas

    def loss(self, v):
        # ams, mus, sigmas = Deconvolutor.__split_params(v)
        # approx_data = np.zeros(len(data))
        # for a, m, s in zip(ams, mus, sigmas):
        #     approx_data += a * gauss(wavenums, m, s)
        return self.loss_func(self.spectrum.data, self.__approximation(v))  # np.sum(np.square(data - approx_data))

    def deconvolute(self):
        spc = self.spectrum * 1
        spc.data = spc.get_derivative(n=2)
        indices = spc.get_extrema(minima=True)[0]
        thr = spc.std * self.threshold
        indices = filter(lambda x: abs(self.spectrum.data[x]) > thr, indices)
        self.minima = list(indices)
        initial_params = self.__init_params()
        res = minimize(self.loss,  # lambda x: self.loss(x, data=self.spectrum.data),
                       initial_params, method='Nelder-Mead', tol=1e-6).x
        return zip(*Deconvolutor.__split_params(res))

    def __approximation(self, params_vector):
        approx_data = np.zeros(len(self.spectrum))
        for a, m, s in zip(*Deconvolutor.__split_params(params_vector)):
            approx_data += Deconvolutor.distribution()(self.spectrum.wavenums, a, m, s)
        return approx_data

    @classmethod
    def distribution(cls, gauss_proba=1):
        if random() < gauss_proba:
            return gauss
        else:
            return lorentz

    def __init_params(self):
        amplitudes = np.ones(len(self.minima))
        mus = np.array(self.spectrum.wavenums[self.minima])
        sigmas = 1. / np.array(self.spectrum.data[self.minima]) / np.sqrt(2. * np.pi)
        # amplitudes = self.spectrum.data[self.minima]
        # mus = self.spectrum.wavenums[self.minima]
        # sigmas = 1. / self.spectrum.data[self.minima] / np.sqrt(2. * np.pi)
        return np.concatenate((amplitudes, mus, sigmas))


def create_spectrum(x, params):
    data = np.zeros(len(x))
    for amp, mu, w, g in params:
        data += voigt(x, amp, mu, w, g)
    return Spectrum(wavenums=x, data=data)


if __name__ == '__main__':
    print('Hi')
    # x = np.arange(400., 12500., 0.093)
    #
    # params = [
    #     (3.0, 2567., 560., 0.3),
    #     (3.5, 3245., 123., 0.8),
    #     (4.7, 10000., 256., 1.0),
    #     (3.8, 8464., 430., 0.0),
    #     (2.9, 3024., 467., 0.4),
    #     (0.5, 10532., 380., 0.1)
    # ]
    # sp = create_spectrum(x, params)
    # # sp.data += np.random.normal(loc=0, scale=sp.std * 0.01, size=(len(x),))
    # sp2 = sp * 1
    # # sp.data = sp.get_derivative(n=2)
    # # d = sm.nonparametric.lowess(sp2.data, sp2.wavenums, frac=0.10, delta=15, it=3)
    # # sp2.data, sp2.wavenums = d[::-1, 1], d[::-1, 0]# d = sm.nonparametric.lowess(sp2.data, sp2.wavenums, frac=0.10, delta=15, it=3)
    # # sp2.data, sp2.wavenums = d[::-1, 1], d[::-1, 0]
    # sp.clss = 'smoothed'
    # sp2.data = sp2.get_derivative(n=2)
    # plt.plot(x, sp2.data)
    # plt.show()
    # # show_spectra()

    # plt.figure()
    # x = np.arange(-4.5, 4.5, 0.01)
    # w, amp, mu = 1., 3., 0.
    # gau = gauss(x, amp, mu, w)
    # lor = lorentz(x, amp, mu, w)
    # voi = voigt(x, amp, mu, w, 0.5)
    # s =  np.max(np.fft.ifft(np.fft.fft(lor) * np.fft.fft(gau)).real)
    # slv =  np.max(np.fft.ifft(np.fft.fft(lor) * np.fft.fft(voi)).real)
    # sgv =  np.max(np.fft.ifft(np.fft.fft(voi) * np.fft.fft(gau)).real)
    # sgg =  np.max(np.fft.ifft(np.fft.fft(gau) * np.fft.fft(gau)).real)
    # sll =  np.max(np.fft.ifft(np.fft.fft(lor) * np.fft.fft(lor)).real)
    # print(sgg, sgv, s, slv, sll)
    # plt.plot(x, gau, label='gauss')
    # plt.plot(x, lor, label='lorentz')
    # plt.plot(x, voi, label='voigt')
    # plt.legend()
    # plt.show()

    # print(width_sigma)
    # amp, m, w = 3., 1., 0.5
    # x = np.arange(-3., 4., 0.01)
    # #print(voigt(x, amp, m, w, 0.5))
    # plt.figure(figsize=(7, 9))
    # plt.xlim((-3., 4.))
    # # plt.plot(x, voigt(x, amp, m, w, 0.5))
    # g = gauss(x, amp, m, w)
    # l = lorentz(x, amp, m, w)
    # plt.plot(x, g, 'g', label='gauss')
    # plt.plot(x, l, 'b', label='lorentz')
    # for i in np.arange(.1, .9, .1):
    #     plt.plot(x, voigt(x, amp, m, w, i), label=f'voigt {np.round(i, 1)}')
    # plt.legend()
    # plt.show()
    # # print(*[i for i in dcv.deconvolute()], sep='\n')

