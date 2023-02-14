from scipy.signal import savgol_filter, wiener, sosfilt, spline_filter, deconvolve
from enumerations import LossFunc, NormMode, BaseLineMode
import numpy as np
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scan import get_spectra_list, read_columns
from scipy.stats import mode
from output import show_spectra
import os
import statsmodels.api as sm
from spectrum import Spectrum
import time
from deconvolution import create_spectrum


class Smoother:
    #methods = [savgol, moving_average, ]
    @staticmethod
    def param_gen(param, iterby=None):     
        names = list(param)
        if iterby:
            del names[names.index(iterby)]
            names.append(iterby)
            limit = len(param[iterby])
        count = 0
        res = {}
        def choose_one(i):
            nonlocal count
            for val in param[names[i]]:
                res[names[i]] = val
                if i == len(names) - 1:
                    count += 1
                    yield res.copy()
                    if iterby and count == limit:
                        return
                else:
                    yield from choose_one(i + 1)
        yield from choose_one(0)

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


#
# def lowess_params_grid(spc, deltas, fracs, iters,
#                        path='lowess_report.csv',
#                        delimiter=','):
#     headers = ['delta', 'frac', 'iter', 'peaksnumber', 'comptime']
#     with open(path, 'w') as out:
#         out.write(delimiter.join(headers) + '\n')
#         for delta in tqdm(deltas):
#             for frac in tqdm(fracs):
#                 for iter in iters:
#                     cursp = spc * 1
#                     line = [str(int(delta)), str(np.round(frac, 3)), str(iter),]
#                     t1 = time.time()
#                     d = sm.nonparametric.lowess(spc.data, spc.wavenums, frac=frac, delta=delta, it=iter)
#                     t2 = time.time()
#                     cursp.data = d[::-1, 1]
#                     cursp.data = spc.get_derivative(n=2)
#                     line.append(str(len(cursp.get_extrema(minima=True, locals=True, include_edges=False)[1])))
#                     line.append(str(t2 - t1))
#                     out.write(delimiter.join(line) + '\n')

def lowess_params_grid_compare(spc, spc_ideal, prng,
                       path='lowess_report.csv',
                       delimiter=',',
                       loss=LossFunc.RMSE):
    headers = list(prng) + ['peaks', 'comptime', 'loss']
    with open(path, 'w') as out:
        out.write(delimiter.join(headers) + '\n')
        for params in Smoother.param_gen(prng):
            line = [str(np.round(params[param], 3) for param in params)]
            cursp = spc * 1
            t1 = time.time()
            d = sm.nonparametric.lowess(spc.data, spc.wavenums, **params)
            t2 = time.time()
            cursp.data = d[::-1, 1]
            cursp.data = spc.get_derivative(n=2)
            line.append(str(len(cursp.get_extrema(minima=False, locals=True, include_edges=False)[1])))
            line.append(str(t2 - t1))
            line.append(str(loss(cursp.data, spc_ideal.data)))
            out.write(delimiter.join(line) + '\n')
            
def lowess(spc, delta=17, frac=0.003, iter=3):
    d = sm.nonparametric.lowess(spc.data, spc.wavenums, delta=delta, frac=frac, it=iter)
    data = d[::-1, 1]
    return data

def fourier(spc, thr, size):
    y = spc.data
    y = fft(y)
    abs_vals = np.abs(y)
    level = thr * np.max(abs_vals)
    y[abs_vals < level] = 0
    y[int(len(y) * size):] = 0
    assert len(spc) == len(ifft(y))
    return ifft(y)

def peaks_losses(noised, ideal, smooth, config: dict, iterby: str, loss=LossFunc.RMSE):
    real_peaks_number = len(ideal.get_extrema(minima=False, locals=True)[0])
    losses = []
    peaks = [] #[len(noised.get_extrema(minima=False, locals=True)[0])]
    addspc = noised * 1
    prms = []
    # print(len(prms), prms)
    # print(len(peaks), peaks)
    for prm in Smoother.param_gen(config, iterby):
        prms.append(prm[iterby])
        y = smooth(noised, **prm)
        addspc.data = y
        peaks.append(len(addspc.get_extrema(minima=False, locals=True)[0]))
        losses.append(loss(y, ideal.data))
    

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(prms, losses)
    ax1.set_title('Losses')
    ax2.plot(prms, peaks)
    ax2.set_title('Peak number')
    ax2.axhline(real_peaks_number, color='r')
    ax1.set_xlabel(iterby)
    ax2.set_xlabel(iterby)
    plt.show()


if __name__ == '__main__':
    print('Hi!')
    spc = get_spectra_list(path='/home/imc/spectrum/data', classify=True, recursive=True)[127]
    show_spectra([spc])
    noised = spc + np.random.normal(loc=0, scale=spc.std * 0.01, size=(len(spc),))
    print(len(spc.get_extrema(minima=False, locals=True)[0]))
    # print(len(noised.get_extrema(minima=False, locals=True)[0]))
    # noised.clss = 'noised'
    # config = {'frac': np.arange(0.00001, 0.1, 0.001), 
    #           'delta':[5]
    #         }
    # show_spectra([spc, noised])
    # peaks_losses(noised, spc, lowess,
    #              config=config,
    #              iterby='frac')
    params = [
        (3.0, 2567., 560., 0.3),
        (3.5, 3245., 123., 0.8),
        (4.7, 10000., 256., 1.0),
        (3.8, 8464., 430., 0.0),
        (2.9, 3024., 467., 0.4),
        (0.5, 10532., 380., 0.1)
    ]
    spc = create_spectrum(spc.wavenums, params)
    # noised = spc + np.random.normal(loc=0, scale=spc.std * 0.01, size=(len(spc),))
    
    # der = noised * 1
    smoothed = spc * 1
    smoothed.data = moving_average(smoothed.data)
    smoothed.data = smoothed.get_derivative(2)
    noised = spc + np.random.normal(loc=0, scale=spc.std * 0.01, size=(len(spc),))
    noised.data = moving_average(smoothed.data)
    noised.data = noised.get_derivative(2)
    spc.data = spc.get_derivative(2)
    spc.clss, noised.clss, smoothed.clss = 'ori', 'noi', 'smo'
    show_spectra([ noised])
    
    



    # spc = get_spectra_list(path='../data', classify=True, recursive=True)[127]
    # print('Original peaks number ', len(spc.get_extrema(minima=False, locals=True)[0]))
    # spc += np.random.normal(loc=0, scale=spc.std * 0.01, size=(len(spc),))
    # losses = [0]
    # peaks = [len(spc.get_extrema(minima=False, locals=True)[0])]
    # wndws = list(range(1, 12, 2))
    # cur = spc * 1
    # for w in wndws:
    #     cur.data = moving_average(spc.data, w)
    #     peaks.append(len(cur.get_extrema(minima=True, locals=True)[0]))
    #     losses.append(LossFunc.RMSE(cur.data, spc.data))
    #
    # plt.plot([0] + wndws, peaks,)
    # plt.legend()
    # plt.show()
    # plt.plot([0] + wndws, losses)
    # plt.show()


    # spc = get_spectra_list(path='../data', classify=True, recursive=True)[127]
    # spc_ideal = spc * 1
    # spc += np.random.normal(loc=0, scale=spc.std * 0.01, size=(len(spc),))
    #
    # lowess_params_grid_compare(spc, spc_ideal,
    #                    deltas=list(range(1, 36, 2)),
    #                    fracs=np.arange(0.001, 0.04, 0.002),
    #                    iters=[1, 4, 16],
    #                    path=r"C:\Users\user\Desktop\lowess_report.csv",
    #                    loss=LossFunc.LOG2)


