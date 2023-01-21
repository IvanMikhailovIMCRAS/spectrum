from copy import deepcopy
import numpy as np
import os
import matplotlib.pyplot as plt
import brukeropusreader as opus
from random import sample
import logging
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull
from scipy.linalg import cholesky
from scipy.signal import savgol_filter
from BaselineRemoval import BaselineRemoval
from enums import NormMode, BaseLineMode
import exceptions

#
# add range
# show specta - give intervals
# Проверить, почему не работает rubberband

# interpolate
# change size!!!

class Spectrum:
    __ATR_to_AB = 1000
    spectrum_id = 0
    __ops = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y
    }

    def __init__(self, wavenums=[], data=[], path='', clss: str = 'undefined'):
        self.path = path
        self.clss = clss
        if path and path.endswith('.csv'):
            self.wavenums, self.data, self.clss = Spectrum.read_csv(path)
        elif path:
            self.wavenums, self.data = Spectrum.__read_opus(path)
        elif wavenums and data:
            self.wavenums, self.data = wavenums, data
        else:
            raise exceptions.SpcCreationEx
        Spectrum.spectrum_id += 1
        self.id = Spectrum.spectrum_id
        self.step = abs(self.wavenums[1] - self.wavenums[0]) if len(self.wavenums) > 1 else 0

    def __len__(self):
        return len(self.wavenums)

    def __str__(self):
        return '\t'.join((str(self.id), self.path, self.clss))

    def __bool__(self):
        return len(self.wavenums) != 0

    def __getitem__(self, ind):
        return self.wavenums[ind], self.data[ind]

    def __iter__(self):
        for i in range(len(self)):
            yield self.wavenums[i], self.data[i]

    def range(self, bigger, lesser):
        start_ind = int((self.wavenums[0] - bigger) / self.step) if bigger < self.wavenums[0] else 0
        stop_ind = len(self) - int((lesser - self.wavenums[-1]) / self.step) if lesser > self.wavenums[-1] else len(
            self)
        s = self * 1
        s.wavenums = s.wavenums[start_ind:stop_ind]
        s.data = s.data[start_ind:stop_ind]
        return s

    def select(self, *intervals):
        """
		intervals --  края отрезков (wavenums), которые НУЖНО включить
		rtype: None
		"""
        mask = [False] * len(self)
        intevals = [interval.sort() for interval in intervals]
        for i in range(len(self)):
            for interval in intervals:
                if interval[0] <= self.wavenums[i] <= interval[1]:
                    mask[i] = True
                    break
        self.data = self.data[mask]
        self.wavenums = self.wavenums[mask]

    def normalize(self, method=NormMode.VECTOR):
        if method == NormMode.AREA:
            norm_coef = np.sum(self.data)
            self.data /= norm_coef
        elif method == NormMode.MINMAX:
            min_val = np.min(self.data)
            max_val = np.max(self.data)
            self.data = (self.data - min_val) / (max_val - min_val)
        else:
            norm_coef = np.sqrt(np.sum(np.square(self.data)))
            self.data /= norm_coef

    def correct_baseline(self, method=BaseLineMode.RB, **kwargs):
        # https://stackoverflow.com/questions/66039235/how-to-subtract-baseline-from-spectrum-with-rising-tail-in-python
        if method == BaseLineMode.ALSS:
            self.data = baseline_alss(self.data, **kwargs)
        elif method == BaseLineMode.ZHANG:
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

    def get_extrema(self, locals=True, minima=False, include_edges=False):
        indices = []
        wavenums = []
        iextr = 1

        if minima:
            f = lambda i: self.data[i - 1] > self.data[i] and self.data[i] < self.data[i + 1]
            comp = lambda i, iextr: self.data[i] < self.data[iextr]
        else:
            f = lambda i: self.data[i - 1] < self.data[i] and self.data[i] > self.data[i + 1]
            comp = lambda i, iextr: self.data[i] > self.data[iextr]

        for i in range(1, len(self) - 1):
            if f(i):
                indices.append(i)
                wavenums.append(self.wavenums[i])
                iextr = i if comp(i, iextr) else iextr

        if include_edges:
            if minima and self.data[0] < self.data[1] \
                    or (not minima) and self.data[0] > self.data[1]:
                indices.insert(0, 0)
                wavenums.insert(0, self.wavenums[0])
            if minima and self.data[-1] < self.data[-2] \
                    or (not minima) and self.data[-1] > self.data[-2]:
                indices.append(len(self) - 1)
                wavenums.append(self.wavenums[-1])
            iextr = 0 if comp(0, iextr) else iextr
            iextr = len(self) - 1 if comp(len(self) - 1, iextr) else iextr
        if not locals:
            return [iextr], [self.wavenums[iextr]]
        return indices, wavenums

    def standartize(self):
        self.data = (self.data - self.mean) / self.std

    @property
    def mean(self):
        return np.mean(self.data)

    @property
    def std(self):
        return np.std(self.data)

    def reset(self):
        '''
        Restore the values of wavenumbers and intensities according to the file in self.path.
        '''
        self.wavenums, self.data = Spectrum.__read_opus(self.path)

    @classmethod
    def __two_op_spectra_operation(cls, self, other, op):
        s = deepcopy(self)
        Spectrum.spectrum_id += 1
        s.id = Spectrum.spectrum_id
        if isinstance(other, (float, int)):
            s.data = Spectrum.__ops[op](self.data, other)
        elif isinstance(other, Spectrum):
            if self.is_comparable(other):
                s.data = Spectrum.__ops[op](self.data, other.data)
            else:
                raise exceptions.SpcChangeEx
        else:
            raise TypeError
        return s

    @classmethod
    def __in_place_spectra_operation(cls, self, other, op):
        if isinstance(other, (float, int)):
            self.data = Spectrum.__ops[op](self.data, other)
        elif isinstance(other, Spectrum):
            if self.is_comparable(other):
                self.data = Spectrum.__ops[op](self.data, other.data)
            else:
                raise exceptions.SpcChangeEx
        else:
            raise TypeError
        return self

    def __iadd__(self, other):
        return Spectrum.__in_place_spectra_operation(self, other, '+')

    def __isub__(self, other):
        return Spectrum.__in_place_spectra_operation(self, other, '-')

    def __imul__(self, other):
        return Spectrum.__in_place_spectra_operation(self, other, '*')

    def __add__(self, other):
        return Spectrum.__two_op_spectra_operation(self, other, '+')

    def __radd__(self, other):
        return Spectrum.__two_op_spectra_operation(self, other, '+')

    def __mul__(self, other):
        return Spectrum.__two_op_spectra_operation(self, other, '*')

    def __rmul__(self, other):
        return Spectrum.__two_op_spectra_operation(self, other, '*')

    def __sub__(self, other):
        '''
		resulting spectrum inherits all the attributes of the first argument
		'''
        return Spectrum.__two_op_spectra_operation(self, other, '-')

    def __rsub__(self, other):
        return Spectrum.__two_op_spectra_operation(-1 * self, other, '+')

    def is_comparable(self, other):
        '''
        params:
        other: Spectrum
        rtype: bool
        Returns whether it is possible to compare two spectra value by value and to operate with them further.
        '''
        return len(self) == len(other) \
               and self.wavenums[0] == other.wavenums[0] \
               and self.wavenums[-1] == other.wavenums[-1]

    def change_size(self, sample):
        size = len(sample)
        sample_minval = sample.wavenums[0]
        sample_maxval = sample.wavenums[-1]
        spc_minval = self.wavenums[0]
        spc_maxval = self.wavenums[-1]

        if sample_maxval >= spc_maxval and spc_minval >= sample_minval:  # s lies inside the spectrum
            for i in range(len(self)):
                pass

    @staticmethod
    def __read_opus(path):
        x, y = [], []
        try:
            file = opus.read_file(path)
            x = file.get_range()
            y = file['AB']
        except exceptions.SpcReadingEx as err:
            pass
        finally:
            if len(x) > 1:
                return x[:-1], y[:-1]
            return x, y

    @classmethod
    def read_csv(cls, path):
        '''
        Read the only spectrum from the .csv file
        '''
        with open(path, 'r') as csv:
            scale = csv.readline().split(',')
            scale_type = scale[0]
            if scale_type in ('Type', 'Wavelength'): # 'Type' refers to the case when the
                f = lambda x: 10_000_000 / float(x)
            else: # scale_type -- wavenums in cm-1
                f = float
            scale = np.array([f(scale[i]) for i in range(1, len(scale))])
            spc = csv.readline().strip().split(',')
            clss = spc[0]
            data = np.array([float(spc[i]) for i in range(1, len(spc))])
            return scale, data, clss

    def save_as_csv(self, path, scale_type='Wavenumbers'):
        if scale_type == 'Wavelengths':
            scale = 10_000_000. / self.wavenums
        else:
            scale = self.wavenums
        with open(path, 'w') as out:
            print(scale_type, *scale, sep=',', file=out)
            print(self.clss, *self.data, sep=',', file=out)

    def intrerpolate(self):
        pass


def spectra_log(spectra_dict, path='log.txt'):
    with open(path, 'w') as f:
        for spc in spectra_dict:
            print(spectra_dict[spc], file=f)


def baseline_alss(y, lam=1e6, p=1e-3, niter=10):
    """
	an algorithm called "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens
	https://stackoverflow.com/questions/29156532/python-baseline-correction-library
	"""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z


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
    for i in range(1, size - 1):
        if y[i - 1] > y[i] < y[i + 1]:
            im.append(i)
    im.append(size - 1)
    i_gm = np.argmin(y)  # index of global minimum
    i_lm = i_gm  # index of local minimum
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


def get_spectra_list(**kwargs):
    res = []
    for p, clss in read_data(**kwargs):
        res.append(Spectrum(path=p, clss=clss))
    return res


def get_spectra_dict(**kwargs):
    res = {}
    for p, clss in read_data(**kwargs):
        s = Spectrum(path=p, clss=clss)
        res[s.id] = s
    return res


def read_data(path='data', classify=False, recursive=False):
    '''
    params: path: str = 'data'
    path to the root directory of spectra. If not defined, the path is the current directory
    params: classify: bool = False
    if True, the classes are defined as the first-layer directory name; or paths aren't included in case they're in the root directory.
    else all paths are marked as '.'
    params: recursive: bool = False
    if True, explores all the nested directories, else is limited with only 0,1-level directories
	rtype: List[Tuple(path: str, class: str)]
	'''
    paths = []
    base = os.path.basename(path)
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            paths.append(os.path.join(dirpath, filename))
    paths = list(filter(filter_opus, paths))

    if not paths:
        return paths, []

    classes = [None] * len(paths)
    #parts0 = len(path.split(os.sep)) - 1
    for ind, path in enumerate(paths):
        path_parts = path.split(os.sep) # len(path.split(os.sep)) - parts0
        if not classify and not recursive:
            if len(path_parts) == 2:
                classes[ind] = base
        elif not classify and recursive:
            classes[ind] = base
        elif classify and not recursive:
            if len(path_parts) == 3:
                classes[ind] = path_parts[1]
        else:
            if len(path_parts) >= 3:
                classes[ind] = path_parts[1]
    paths = [paths[i] for i, clss in enumerate(classes) if clss is not None]
    classes = [clss for clss in classes if clss is not None]
    return list(zip(paths, classes))


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


def show_spectra(spectra, save_path='', wavenumbers=None):
    if not spectra:
        return
    classes = list(sorted(set(map(lambda x: x.clss, spectra))))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    colors = dict(zip(classes, colors))
    plt.figure()

    lines = []
    clrs = []
    for spc in spectra:
        if spc:
            if wavenumbers:
                spc = spc.range(*wavenumbers)
            lines.append(plt.plot(spc.wavenums, spc.data, c=colors[spc.clss], linewidth=0.5))
            clrs.append(spc.clss)
    if len(clrs) > 1:
        plt.legend(clrs)
    plt.xlim(spc.wavenums[0], spc.wavenums[-1])
    plt.xlabel('wavenumber, cm-1')
    plt.ylabel('intensity')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    print('HI')
    with open('ex.csv', 'r') as f:
        plt.figure()
        f.readline()
        for line in f.readlines():
            values = [float(i) for i in line.strip().split(',')[1:]]
            plt.plot(values)
        plt.show()

    # spa = get_spectra_list(recursive=False, classify=True)
    # sp = spa[0]
    # print(sp.wavenums)
    # sp1 = sp.wavenums[:]
    # print(sp1 is sp.wavenums)
    # print(sp1 == sp.wavenums)


    import test
    # s1 = Spectrum(wavenums=list(range(6)), data=list(range(6)))
    # s2 = Spectrum(wavenums=list(range(6)), data=list(range(-5, 1)))
    # print(s1.data)
    # print(s2.data)
    # s = s1 + s2
    # print(s)
    # f = opus.read_file(r'C:\Users\user\PycharmProjects\spectrum\SD10.30')
    # print(*[it for it in f.items()], sep='\n\n')
    # sp = get_spectra_list(
    #     recursive=False)[:2]
    # sp[0] *= 1
    # show_spectra(sp)

# sp = sp.range(4000, 2578)
# sp = get_spectra_list(
# 	path='Спектры сывороток животных/Черепаха raw', recursive=False)[0]
# spca = get_spectra_dict(path=r'Спектры сывороток животных/Черепаха raw', recursive=False)
# spectra_log(spca)
# sp_rb = sp * 1
# sp_rb.correct_baseline()
# sp_rb.clss = 'RB correction'
# show_spectra([sp, sp_rb])

# sp.reset()
# show_spectra([sp])
