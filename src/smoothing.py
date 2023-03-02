from scipy.signal import savgol_filter, wiener, sosfilt, spline_filter, deconvolve
from scipy.ndimage import gaussian_filter
from enumerations import LossFunc, NormMode, BaseLineMode, Smooth
import numpy as np
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
# from scan import get_spectra_list, read_columns
from output import show_spectra
import os
import statsmodels.api as sm
import time
from miscellaneous import gauss

class ParamGrid:
    configs = {}
    methods = []

    @staticmethod
    def param_gen(param: dict, iterby: str=None):
        if not param:
            return {}
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
    
    @staticmethod
    def params_grid_compare(spc, spc_ideal, prng,
                            process,
                                   path='lowess_report.csv',
                                   delimiter=',',
                                   loss=LossFunc.RMSE):
        headers = list(prng) + ['peaks', 'comptime', 'loss']
        with open(path, 'w') as out:
            out.write(delimiter.join(headers) + '\n')
            for params in ParamGrid.param_gen(prng):
                line = [str(np.round(params[param], 3)) for param in params]
                cursp = spc * 1
                try:
                    t1 = time.time()
                    d = process(spc, **params)
                    t2 = time.time()
                except:
                    continue
                cursp.data = d
                line.append(str(len(cursp.get_extrema(minima=False, locals=True, include_edges=False)[1])))
                line.append(str(t2 - t1))
                line.append(str(loss(cursp.data, spc_ideal.data)))
                out.write(delimiter.join(line) + '\n')

    @classmethod
    def _comb_spectrum(cls, spc, fold=2):
        from spectrum import Spectrum
        assert fold >= 2, 'Can\'t split in case fold < 2!'
        size = len(spc)
        for offset in range(fold):
            ideal = Spectrum(*spc[offset::fold])
            gen = [spc[i] for i in range(size) if (i - offset) % fold != 0]
            left = Spectrum(*map(np.array, zip(*gen)))
            yield left, ideal

    @classmethod
    def score(cls, spc, fold, process, loss=LossFunc.RMSE, **process_params):
        results = []
        for basis, reference in ParamGrid._comb_spectrum(spc, fold):
            try:
                basis.interpolate(spc.wavenums)
                reference.interpolate(spc.wavenums)
                reference.clss = 'ref'
                yproc = process(basis, **process_params)
                results.append(loss(yproc, reference.data))
            except Exception as ex:
                # print(ex)
                continue
        if not results:
            return np.inf
        results = np.array(results)
        return np.round(results.mean(), 4)


    @classmethod
    def best_method_perfomance(cls, spc, method, grid, loss=LossFunc.RMSE, fold=2):
        best_score = np.inf
        best_params = None
        for prmset in ParamGrid.param_gen(grid):
            score = ParamGrid.score(spc, fold, method, loss, **prmset)
            if score < best_score:
                best_params = prmset
                best_score = score
        return best_params, best_score

    @classmethod
    def best_method(cls, spc, general_grid=None, loss=LossFunc.RMSE, fold=2):
        best_params = None
        best_score = np.inf
        best_method = None
        if not general_grid:
            general_grid = cls.configs
        for methodname in cls.configs:
            grid = general_grid[methodname]
            print(methodname, grid)
            prmset, score = cls.best_method_perfomance(spc, getattr(cls, methodname), grid, loss, fold)
            print('Parameters: ', prmset, ', Score: ', score)
            if score < best_score:
                best_score = score
                best_params = prmset
                best_method = methodname
        return best_method, best_params, best_score


    @classmethod
    def peaks_losses(cls, noised, ideal, process, config: dict, iterby: str, loss=LossFunc.RMSE, ):
        real_peaks_number = len(ideal.get_extrema(minima=False, locals=True)[0])
        losses = []
        peaks = []  # [len(noised.get_extrema(minima=False, locals=True)[0])]
        addspc = noised * 1
        prms = []

        for prm in ParamGrid.param_gen(config, iterby):
            prms.append(prm[iterby])
            y = process(noised, **prm)
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



class Smoother(ParamGrid):
    configs = {
        'savgol': {
            'window_length': list(range(1, 36, 2)),
            'order': [3, 4, 5, 6, 7]
        },
        'moving_average': {
            'window_length': list(range(1, 36, 2))
        },
        'fourier': {
            'threshold': [1e-5],
            'size': np.arange(0.95, 1., 0.01)
        },
        'lowess': {
        },
        'gaussian': {
            'window_length': list(range(1, 18, 2)),
            'order': list(range(9))
        },
        'wiener': {
            'window_length': list(range(1, 18, 2)),
        }
    }


    @staticmethod
    def gaussian(spc, window_length=5, order=5):
        return gaussian_filter(spc.data, sigma=window_length, order=order)
        # hw = window_length // 2
        # filt = gauss(np.array(np.arange(-hw, hw + 1, 1)), 1., 0., hw)
        # filt /= filt.sum()
        # data = spc.data
        # res = list(data[:hw])
        # # print('filt ', filt)
        # for i in range(hw, len(data) - hw):
        #     # print(i -hw, i + hw)
        #     # print(data[i - hw : i + hw + 1])
        #     res[i] = sum(filt * res[i - hw : i + hw + 1])
        # res.extend(data[len(data) - hw:])
        # assert len(res) == len(data)
        # res = np.array(res, dtype=float)
        # return res

    @staticmethod
    def wiener(spc, window_length=5):
        data = spc.data
        return wiener(data, window_length)

    @staticmethod
    def savgol(spc, window_length=7, order=5):
        data = spc.data
        return savgol_filter(data, window_length=window_length, polyorder=order)

    @staticmethod
    def moving_average(spc, window_length=5):
        data = spc.data
        w = data[:window_length - 1].sum()
        hw = window_length // 2
        res = list(data[:hw])
        for i in range(hw, len(data) - hw):
            w += data[i + hw]
            res.append(w / window_length)
            w -= data[i - hw]
        res.extend(data[len(data) - hw:])
        assert len(res) == len(data)
        res = np.array(res, dtype=float)
        return res

    @classmethod
    def iterative_smoothing(cls, spc, smoothfunc, n_iter, lossfunc=LossFunc.RMSE, show_smoothed=False, **kwargs):
        data = spc.data
        iterations = list(range(1, n_iter + 1))
        res = []
        smoothed = np.array(data, copy=True)
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

    @staticmethod
    def lowess(spc, delta=17, frac=0.003, iter=3):
        d = sm.nonparametric.lowess(spc.data, spc.wavenums, delta=delta, frac=frac, it=iter)
        data = d[::-1, 1]
        return data

    @staticmethod
    def fourier(spc, thr=1e-4, size=0.97):
        y = spc.data
        y = fft(y)
        abs_vals = np.abs(y)
        level = thr * np.max(abs_vals)
        y[abs_vals < level] = 0
        y[int(len(y) * size):] = 0
        assert len(spc) == len(ifft(y))
        return ifft(y)


if __name__ == '__main__':
    print('Smooth!')
    from scan import get_spectra_list
    spa = get_spectra_list(path='../data', classify=True, recursive=True)
    spc = spa[127]
    # print(Smoother.best_method_perfomance(spc, Smoother.savgol, Smoother.configs['savgol']))
    print(Smoother.best_method(spc))

    # print(getattr(Smoother, 'savgol')(spc))


    # print(*list(ParamGrid.param_gen(Smoother.configs['savgol'])), sep='\n')
    # for prmset in ParamGrid.param_gen(Smoother.configs['savgol']):
    #     print(Smoother.score(spc, 2, Smoother.savgol, loss=LossFunc.RMSE,
    #                          **prmset))
    #     # methodfunc = Smoother.__dict__[method]
    #     print(methodfunc, method, Smoother.optima

