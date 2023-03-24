from spectrum import Spectrum
from output import show_spectra
from scan import read_columns
import os
from deconvolution import Deconvolutor
import matplotlib.pyplot as plt
from enumerations import BaseLineMode, Scale, NormMode
from miscellaneous import gauss
import numpy as np
import raman_fitting
from scipy.optimize import least_squares, minimize

path = r"C:\Users\user\Desktop\bsa"
spa = []
# print(raman_fitting.SpectrumTemplate()
# )
for file in os.listdir(path):
    fpath = os.path.join(path, file)
    w, d = read_columns(fpath, v_offset=0, columns_indices=(0, 1), delimiter=',', scale=Scale.WAVENUMBERS)
    spc = Spectrum(w, d, clss=file[:-18])
    spc.correct_baseline(BaseLineMode.ALSS)
    spc = spc.range(1590., 1710., x=True)
    points = np.linspace(1590., 1710., 1000)
    spc.interpolate(points)
    spc.normalize(NormMode.MINMAX)
    spa.append(spc)
show_spectra(spa)
for spc in spa:
    # dec = Deconvolutor(spc, threshold=0.04)
    # peaks = dec.deconvolute()
    orig = spc * 1
    spc.get_derivative(2)
    maxdata = orig[spc.get_extrema(locals=False, minima=False)[0]][1]
    eps = 0.04
    # print(spc.get_extrema(minima=True, locals=True))
    indices = list(filter(lambda x: orig[x][1] > eps * maxdata, spc.get_extrema(minima=True, locals=True)[0]))
    true_indices = indices[:]
    indices = list(map(lambda x: orig[x][0], indices))
    plt.subplot(2, 1, 1)
    plt.plot(orig.wavenums, orig.data, color='b')
    for line in indices:
        plt.axvline(line, color='r')
    plt.ylabel('orig')
    plt.subplot(2, 1, 2)
    plt.plot(spc.wavenums, spc.data, color='b')
    for line in indices:
        plt.axvline(line, color='r')
    plt.ylabel('der2')
    plt.show()
    init_amps = [0.8 * orig[amp][1] for amp in true_indices]
    init_widths = [len(orig) / len(true_indices) / 50] * len(true_indices)
    init_pos = [orig[ind][0] for ind in true_indices]
    init = np.array(init_pos + init_amps + init_widths)
    x = orig.wavenums
    def residues(v, *args, **kwargs):
        y = np.zeros(len(orig))
        pos = v[:len(v)//3]
        widths = v[len(v)//3:-len(v)//3]
        amps = v[-len(v) // 3:]
        assert len(pos) == len(widths) == len(amps)
        for i in range(len(init_pos)):
            y += gauss(x, amps[i], pos[i], widths[i])
        return np.square(orig.data - y)

    res = least_squares(residues, init, xtol=10e-10, ftol=10e-10, bounds=list(zip(*[(0, np.inf)] * len(init)))).x

    # print(res)
    pos = res[:len(res) // 3]
    widths = res[len(res) // 3:-len(res) // 3]
    amps = res[-len(res) // 3:]
    gaussians = []
    summ = Spectrum(x, np.zeros(len(x)), clss='summ')
    bands = []
    for i in range(len(amps)):
        gaussians.append(Spectrum(x, gauss(x, amps[i], pos[i], widths[i])))
        bands.append((np.round(gaussians[-1].auc(), 2), np.round(pos[i], 2)))
        summ += gaussians[-1]
    bands = list(map(lambda x: (np.round(x[0] / summ.auc(), 2), x[1]), bands))
    # bands = np.array(bands)
    # bands *= 100 / summ.auc()
    print(orig.clss, '\t', sorted(bands, key=lambda x: x[0],  reverse=True))
    show_spectra(gaussians + [orig, summ])





    # x = orig.wavenums
    # asd = dict(orig)
    # fitter = raman_fitting.Fitter(spectra_arg=asd)
    # fitter.fit_delegator()
    # deconvoluted = [Spectrum(wavenums=x, data=gauss(x, *params)) for params in peaks]
    # show_spectra(deconvoluted + [orig])
