import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from spectrum import Spectrum, show_spectra
from enums import LossFunc
from random import random


def gauss(x, mu, sigma):
    return 1./sigma/np.sqrt(2.*np.pi) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def lorentz(x, x0, gamma):
    return 1./np.pi/gamma * (gamma**2 / (gamma**2 + (x - x0)**2))



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
        return self.loss_func(self.spectrum.data, self.__approximation(v)) # np.sum(np.square(data - approx_data))

    def deconvolute(self):
        #spc = Spectrum()
        # plt.figure()
        spc = self.spectrum * 1
        # plt.plot(self.spectrum.wavenums, self.spectrum.data, 'b')
        spc.data = spc.get_derivative(n=2)
        # plt.plot(spc.wavenums, spc.data, 'p')
        # plt.show()
        indices = spc.get_extrema(minima=True)[0]
        thr = spc.std * self.threshold
        indices = filter(lambda x: abs(self.spectrum.data[x]) > thr, indices)
        self.minima = list(indices)
        # print('Indices: ', self.minima)
        initial_params = self.__init_params()
        res = minimize(self.loss, #lambda x: self.loss(x, data=self.spectrum.data),
                       initial_params, method='Nelder-Mead', tol=1e-6).x
        return zip(*Deconvolutor.__split_params(res))

    def __approximation(self, params_vector):
        approx_data = np.zeros(len(self.spectrum))
        for a, m, s in zip(*Deconvolutor.__split_params(params_vector)):
            approx_data += a * Deconvolutor.distribution()(self.spectrum.wavenums, m, s)
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

# def split_params(v):
#     assert len(v) % 3 == 0
#     assert len(v) != 0
#     third = len(v) // 3
#     ams, mus, sigmas = v[:third], v[third: 2 * third], v[2 * third:]
#     return ams, mus, sigmas

#
if __name__ == '__main__':
    print('Hi')
    wavenums = np.arange(4000., 600., -0.9367)
    data = 1.5 * gauss(wavenums, 2567, 100) + gauss(wavenums, 1230, 200) * 2 + gauss(wavenums, 840, 29)
    spc = Spectrum(wavenums, data)
    dcv = Deconvolutor(spc)
    approx = np.zeros(len(spc))
    for a, m, s in dcv.deconvolute():
        approx += a * gauss(spc.wavenums, m, s)
    approx = Spectrum(spc.wavenums, approx, clss='app')
    show_spectra([approx, spc])
    # print(*[i for i in dcv.deconvolute()], sep='\n')


# wavenums = np.arange(4000., 600., -0.9367)
# data = 1.5 * gauss(wavenums, 2567, 100) + gauss(wavenums, 1230, 200) * 2 + gauss(wavenums, 840, 29)
# spc= Spectrum(wavenums, data)
#
# def get_derivative(data, n=1, win_wight=13, order=5):
# 	return savgol_filter(
# 		data, win_wight, polyorder=order, deriv=n)
#
#
# der2 = spc.get_derivative(n=2)
#
# minima = []
# for i in range(1, len(der2) - 1):
#     if der2[i-1] > der2[i] < der2[i+1]:
#         minima.append(i)
# print('Der2 minima', minima)
# threshold = 0.01 * np.std(data)
# minima_wn = []
# minima_data = []
# for i in minima:
#     if data[i] >= threshold:
#         minima_wn.append(wavenums[i])
#         minima_data.append(data[i])
# print('Threshold', threshold)
# print('filtered wavenums: ', minima_wn)
#
# amplitudes = np.ones(len(minima_wn))
# mus = np.array(minima_wn)
# sigmas = 1./np.array(minima_data)/np.sqrt(2.*np.pi)
# # minima_wn = np.array(minima_wn, dtype=int)
# # amplitudes = data[minima_wn]
# # mus = wavenums[minima_wn]
# # sigmas = 1. / data[minima_wn] / np.sqrt(2. * np.pi)
#
# approx_data = np.zeros(len(data))
# for a, m, s in zip(amplitudes, mus, sigmas):
#     approx_data += a * gauss(wavenums, m, s)
#
# # plt.figure()
# # plt.plot(wavenums, data, 'r', wavenums, approx_data,  'b')
# # plt.show()
#
# def loss(v, data=data):
#     third = len(v) // 3
#     ams, mus, sigmas = v[:third], v[third: 2*third], v[2*third:]
#     approx = np.zeros(len(data))
#     for a, m, s in zip(ams, mus, sigmas):
#         approx += a * gauss(wavenums, m, s)
#     return np.sum(np.square(approx - data))
#
#
# initial_params = np.array(list(amplitudes) + list(mus) + list(sigmas))
# print("initial_params", initial_params)
# res = minimize(lambda x: loss(x, data=data), initial_params, method='Nelder-Mead', tol=1e-6).x
# third = len(res) // 3
# ams, mus, sigmas = res[:third], res[third: 2 * third], res[2*third:]
# approx_data = np.zeros(len(data))
# for a, m, s in zip(ams, mus, sigmas):
#     print(a, m, s)
#     approx_data += a * gauss(wavenums, m, s)
#
# plt.plot(wavenums, data, 'b', wavenums, approx_data, 'r')
# plt.show()
    
    
    
    
# x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

# res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)

# res.x
# array([ 1.,  1.,  1.,  1.,  1.])