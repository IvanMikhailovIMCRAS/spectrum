from scipy.signal import savgol_filter, wiener, sosfilt, spline_filter, deconvolve
from scipy.fft import fft, ifft, fftshift, fftfreq
from enumerations import LossFunc, NormMode, BaseLineMode
import numpy as np
import matplotlib.pyplot as plt
from scan import get_spectra_list
from scipy.stats import mode
from output import show_spectra
import os
import cv2
import statsmodels.api as sm
from spectrum import Spectrum

def savgol(data, window_length=7, order=5):
    return savgol_filter(data, window_length=window_length, polyorder=order)


def moving_average(data, window=5):
    w = data[:window - 1].sum()
    hw = window // 2
    res = list(data[:hw])
    for i in range(hw, len(data) - hw):
        w += data[i + hw]
        res.append(w / window)
        w -= data[i - hw]
    res.extend(data[len(data) - hw:])
    assert len(res) == len(data)
    #print(res)
    res = np.array(res, dtype=float)
    return res


def iterative_smoothing(data, smoothfunc, n_iter, lossfunc=LossFunc.RMSE, show_smoothed=False, **kwargs):
    iterations = list(range(1, n_iter + 1))
    res = []
    smoothed = np.array(data, copy=True)
    if show_smoothed:
        plt.figure()
    for iteration in iterations:
        smoothed = smoothfunc(smoothed, **kwargs)
        if show_smoothed and iteration % 100 == 0:
            plt.plot(smoothed)
        res.append(lossfunc(data, smoothed))
    if show_smoothed:
        plt.show()
    plt.figure()
    plt.plot(iterations, res, label='Loss(n)')
    plt.plot(iterations, savgol_filter(res, deriv=1, window_length=13, polyorder=5), label='dLoss/dn')
    plt.legend()
    plt.show()

def Fourier_filter(y, l, h):
    x = np.array(range(len(y)))
    xf = np.array(range(0, len(y), 2))
    plt.figure()
    plt.plot(y, label='D')
    y = fftshift(fft(y))
    m = mode(y)[0]
    y = y[len(y) // 2:]
    y[:int(len(y) * l)] = 0
    y[:int((1 - h) * len(y))] = 0
    y = ifft(y)
    y = np.interp(x, xf, y)
    plt.plot(y, label='F')
    plt.show()

def lowess_frac(spc):
    res = []
    frs = []
    for fr in np.arange(0.8, 1., 0.01):
        # fr = 1. / i
        frs.append(fr)
        data = sm.nonparametric.lowess(spc.wavenums, spc.data, frac=fr, return_sorted=False)
        res.append(LossFunc.RMSE(data, spc.data))
    # plt.plot(spc.wavenums, res[0], label='0')
    # plt.plot(spc.wavenums, res[1], label='1')
    plt.plot(frs, res)
    plt.legend()
    plt.show()

methods = [savgol, moving_average, ]


def iter_methods(spc, methods, lossfunc, n_iter=1):
    print(f'method', *[i.__name__ for i in methods], sep='\t'*3)

    data = [spc.data[:] for i in methods]
    for i in range(1, n_iter + 1):
        line = [i]
        for j, method in enumerate(methods):
            try:
                data[j] = method(data[j])
                line.append(str(lossfunc(spc.data, data[j])))
            except:
                line.append('###')
        print(*line, sep='\t'*3)


if __name__ == '__main__':
    print('Hi!')
    spc = get_spectra_list(path='../data', classify=True, recursive=True)[67]
    iter_methods(spc, methods, LossFunc.RMSE, 3)

    # iterative_smoothing(spc.data, sm.nonparametric.lowess, 1,
    #                     show_smoothed=True, lossfunc=LossFunc.RMSE, return_sorted=False, exog=spc.wavenums)
    # res = list(zip(*sm.nonparametric.lowess(spc.data, spc.wavenums)))
    # print(*res, sep='\n')
    # res = []
    # frs = []
    # for fr in np.arange(0.8, 1., 0.01):
    #     # fr = 1. / i
    #     frs.append(fr)
    #     data = sm.nonparametric.lowess(spc.wavenums, spc.data, frac=fr, return_sorted=False)
    #     res.append(LossFunc.RMSE(data, spc.data))
    # # plt.plot(spc.wavenums, res[0], label='0')
    # # plt.plot(spc.wavenums, res[1], label='1')
    # plt.plot(frs, res)
    # plt.legend()
    # plt.show()
    # Fourier_filter(spc.data, 0.95, 0.95)


    # sp2 = Spectrum(data= sm.nonparametric.lowess(spc.wavenums, spc.data, frac=0.93, return_sorted=False),
    #                wavenums=spc.wavenums, clss='approx')
    # spc.normalize(NormMode.MINMAX)
    # sp2.normalize(NormMode.MINMAX)
    # sp2.correct_baseline(BaseLineMode.ALSS)
    # spc.correct_baseline(BaseLineMode.ALSS)
    # show_spectra([sp2, spc])

    # plt.plot(spc.wavenums, spc.data, label='data')
    # plt.plot(spc.wavenums, data, label='approx')
    # plt.legend()
    # plt.show()

