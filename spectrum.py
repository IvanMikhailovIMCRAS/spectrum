import numpy as np
import os
import matplotlib.pyplot as plt
import brukeropusreader as opus
from random import sample
import logging
from scipy.stats import mode
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull
from scipy.linalg import cholesky
from scipy.signal import savgol_filter
from BaselineRemoval import BaselineRemoval


class Spectrum:
    ATR_to_AB = 1000
    spectrum_id = 0
    # error_log = os.path.join(os.getcwd(), 'errlog.txt')
    logging.basicConfig(level=logging.INFO,
                        filemode='w',
                        filename=os.path.join(os.getcwd(), 'log.txt'),
                        format='%(asctime)s %(levelname)s %(message)s')

    def __init__(self, path, clss: str):
        self.path = path
        self.wavenums, self.data = Spectrum.read_opus(path)
        self.id = Spectrum.spectrum_id
        self.clss = clss

    def __len__(self):
        return len(self.wavenums)

    def __str__(self):
        return '\t'.join((str(self.id), self.path, self.clss))

    def __bool__(self):
        return len(self.wavenums) != 0

    def normalize(self, method='vector'):
        if method == 'area':
            norm_coef = np.sum(self.data)
            self.data /= norm_coef
        elif method == 'minmax':
            min_val = np.min(self.data)
            max_val = np.max(self.data)
            self.data = (self.data - min_val) / (max_val - min_val)
        else:
            norm_coef = np.sqrt(np.sum(np.square(self.data)))
            self.data /= norm_coef

    def correct_baseline(self, method='rb', **kwargs):
        # https://stackoverflow.com/questions/66039235/how-to-subtract-baseline-from-spectrum-with-rising-tail-in-python
        if method == 'alss':
            self.data = baseline_alss(self.data, **kwargs)
        elif method == 'zhang':
            self.data = baseline_zhang(self.data, **kwargs)
        else:
            self.data = baseline_rubberband(self.data, **kwargs)

    def atr_to_absorbace(self):
        self.data = Spectrum.ATR_to_AB * self.data / self.wavenums

    def smooth(self, method='savgol', **kwargs):
        # пока один метод сглаживания, но можно дописать другие
        self.data = savgol_filter(self.data, **kwargs)

    def get_derivative(self, n=1, win_wight=13, order=5):
        self.data = savgol_filter(
            self.data, win_wight, polyorder=order, deriv=n)

    def __sub__(self, other):
        pass

    @staticmethod
    def read_opus(path):
        x, y = [], []
        try:
            logging.info(f"Trying to open file: {path}")
            file = opus.read_file(path)
            x = file.get_range()
            y = file['AB']
            logging.info(f'{path} is successfully read.')
            Spectrum.spectrum_id += 1
        except Exception as err:
            logging.error(f'Error occured!', exc_info=True)
        finally:
            if len(x) > 1:
                return x[:-1], y[:-1]
            return x, y


def baseline_alss(y, lam=1e6, p=1e-3, niter=10):
    """
    an algorithm called "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return y - z


# Проверить, почему не работает
def rubberband(x, y):
    # Find the convex hull
    v = ConvexHull(np.vstack((x, y)).T).vertices
    # Rotate convex hull vertices until they start from the lowest one
    v = np.roll(v, v.argmax())
    # Leave only the ascending part
    v = v[:v.argmax()]
    # Create baseline using linear interpolation between vertices
    return y - np.interp(x, x[v], y[v])


def baseline_rubberband(y):
    size = len(y)
    im = [0]
    for i in range(1, size-1):
        if y[i-1] > y[i] < y[i+1]:
            im.append(i)
    im.append(size-1)
    i_gm = np.argmin(y)  # index of global minimum
    i_lm = i_gm         # index of local minimum
    max_delta = np.max(y) - np.min(y)
    indexies = []
    while i_lm < size - 1:
        min_slope = max_delta
        i0 = im.index(i_lm)
        i_lm = size - 1
        for i in im[i0:]:
            slope = (y[i] - y[im[i0]]) / float(i - im[i0])
            if slope < min_slope:
                min_slope = slope
                i_lm = i
        indexies.append(i_lm)
    i_lm = i_gm
    while i_lm > 1:
        min_slope = max_delta
        i0 = im.index(i_lm)
        i_lm = 0
        for i in im[i0:0:-1]:
            slope = (y[i] - y[im[i0]]) / float(i - im[i0])
            if slope < min_slope:
                min_slope = slope
                i_lm = i
        indexies.append(i_lm)
    indexies.append(i_gm)
    indexies = sorted(list(set(indexies)))
    z = np.interp(range(size), indexies, y[indexies])
    return y - z


def baseline_zhang(y, polynomial_degree=2):
    """
    adaptive iteratively reweighted Penalized Least Squares (airPLS) - Zhi-Min Zhang et.al
    https://pubs.rsc.org/is/content/articlelanding/2010/an/b922045c#!divAbstract
    """
    baseObj = BaselineRemoval(y)
    return baseObj.ZhangFit()


def create_matrix(raw_spectra):
    # преобразование спектров других длин ?

    matrix = []
    if not raw_spectra:
        return np.zeros((1, 1))
    length, _ = mode([len(x)
                      for x in raw_spectra if len(x) != 0], keepdims=False)
    for raw_spectrum in raw_spectra:
        if raw_spectrum and len(raw_spectrum) == length:
            matrix.append(raw_spectrum.data)
        else:
            pass
    return np.array(matrix)


def get_spectra_list(**kwargs):
    res = []
    for p, clss in zip(*read_data(**kwargs)):
        res.append(Spectrum(p, clss))
    return res


def read_data(path=None, classify=False, recursive=False):
    '''
    rtype: (List[paths: str], List[classes: str])
    '''
    paths = []
    if path is None:
        path = '.'
    base = os.path.basename(path)
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            paths.append(os.path.join(dirpath, filename))
    paths = list(filter(filter_opus, paths))

    if not paths:
        return paths, []

    classes = [None] * len(paths)
    parts0 = len(path.split(os.sep)) - 1
    for ind, path in enumerate(paths):
        path_parts = len(path.split(os.sep)) - parts0
        if not classify and not recursive:
            if path_parts == 2:
                classes[ind] = base
        elif not classify and recursive:
            classes[ind] = base
        elif classify and not recursive:
            if path_parts == 3:
                classes[ind] = path_parts[1]
        else:
            if path_parts >= 3:
                classes[ind] = path_parts[1]
    paths = [paths[i] for i, clss in enumerate(classes) if clss is not None]
    classes = [clss for clss in classes if clss is not None]
    return paths, classes


def filter_opus(path):
    ext = path[path.rfind('.') + 1:]
    if not ext.isdigit():
        return False
    with open(path, 'r') as f:
        try:
            f.read()
            return False
        except:
            return True


def show_spectra(spectra):
    # Зафиксировать цвета для классов
    # Если класс один, то легенда не нужна
    # добавить опцию печати в файл
    classes = set(map(lambda x: x.clss, spectra))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    colors = dict(zip(classes, colors))
    plt.figure()
    lines = []
    clrs = []
    for spc in spectra:
        if spc:
            lines.append(plt.plot(spc.wavenums, spc.data, c=colors[spc.clss]))
            clrs.append(spc.clss)
    plt.legend(clrs)
    plt.xlim(spc.wavenums[0], spc.wavenums[-1])
    plt.xlabel('wavenumber, cm-1')
    plt.ylabel('intensity')
    plt.show()


if __name__ == '__main__':
    sp = get_spectra_list(
        path='Спектры сывороток животных/Черепаха raw', recursive=False)[0]

    sp1 = Spectrum(sp.path, clss='alss')
    sp1.correct_baseline(method='alss')
    sp2 = Spectrum(sp.path, clss='rb')
    sp2.correct_baseline(method='rb')
    sp3 = Spectrum(sp.path, clss='zhang')
    sp3.correct_baseline(method='zhang')
    sp4 = Spectrum(sp.path, clss='deriv')
    sp4.correct_baseline(method='zhang')
    sp4.smooth(window_length=31, polyorder=5)
    sp4.get_derivative(n=2)
    sp4.data *= sp3.data
    sp4.normalize(method='minmax')
    show_spectra([sp, sp1, sp2, sp3, sp4])
