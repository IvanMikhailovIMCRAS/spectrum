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
from enums import NormMode, BaseLineMode, Scale
import exceptions

# add range
# show specta - give intervals
# reader class common for spectrum and matrix?
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

    def __init__(self, wavenums=None, data=None, path='', clss: str = 'undefined'):
        if wavenums is None:
            wavenums, data = np.array([], dtype=float), np.array([], dtype=float)
        self._path = path
        if path and path.endswith('.csv'):
            self.wavenums, self.data, clss = Spectrum.read_csv(path)
        elif path:
            self.wavenums, self.data = Spectrum.__read_opus(path)
        elif len(wavenums) == len(data) != 0:
            self.wavenums, self.data = wavenums, data
        else:
            raise exceptions.SpcCreationEx

        self.clss = clss
        Spectrum.spectrum_id += 1
        self.__id = Spectrum.spectrum_id
        self.step = abs(self.wavenums[1] - self.wavenums[0]) if len(self.wavenums) > 1 else 0

    @property
    def id(self):
        return self.__id

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
        """
        Create a new Spectrum limited by wavenumbers with the passed values.

        params
        bigger: float - The greater wavenumber value, cm-1
        lesser: float - The lesser wavenumber value, cm-1

        rtype: Spectrum
        """
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
        """
        Normalize intensities values in-place.
        """
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
        """
        Recalculate the values of the spectrum from attenuated total reflectance to absorbance.
        """
        self.data = Spectrum.__ATR_to_AB * self.data / self.wavenums

    def smooth(self, method='savgol', **kwargs):
        # пока один метод сглаживания, но можно дописать другие
        self.data = savgol_filter(self.data, **kwargs)

    def get_derivative(self, n=1, win_width=13, order=5):
        """
        Return the n-th derivative of intensity values array.

        params
        n: int - derivative order
        win_width: int - the window size (only odd numbers are allowed).
        order: the order of the polynomial used to approximate the derivative

        rtype: numpy.ndarray(dtype=float)
        """
        if len(self) < 39:
            win_width = len(self) // 2 + 1
        if win_width % 2 != 1:
            win_width += 1

        return savgol_filter(
            self.data, win_width, polyorder=order, deriv=n)

    def get_extrema(self, locals=True, minima=False, include_edges=False):
        """
        params
        locals: bool - if True, return every local extrema specified, else only the global one.
        minima: bool - if True, function searches for minima, else for maxima.
        include_edges: bool - The edge points are suspicious, thus, their presence is at the user's discretion.
        rtype: Tuple(List(indices), List(wavenumbers))
        """
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
        if self._path.endswith('.csv'):
            self.wavenums, self.data, _ = Spectrum.read_csv(self._path)
        else:
            self.wavenums, self.data = Spectrum.__read_opus(self._path)

    @classmethod
    def __two_op_spectra_operation(cls, self, other, op):
        s = deepcopy(self)
        Spectrum.spectrum_id += 1
        s.__id = Spectrum.spectrum_id
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
        """
		resulting spectrum inherits all the attributes of the first argument
		"""
        return Spectrum.__two_op_spectra_operation(self, other, '-')

    def __rsub__(self, other):
        return Spectrum.__two_op_spectra_operation(-1 * self, other, '+')

    def is_comparable(self, other):
        '''
        Returns whether it is possible to compare two spectra value by value and to operate with them further.

        params:
        other: Spectrum
        rtype: bool
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
        """
        Read the only spectrum from the .csv file
        """
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
            clss, *data = csv.readline().strip().split(',')
            data = np.array(list(map(float, data)))
            return scale, data, clss

    def save_as_csv(self, path, scale_type=Scale.WAVENUMBERS):
        """
        Saves the file in csv format with the specified scale.
        params
        path: str - path to the destination file
        scale_type: Scale
        """
        if scale_type == Scale.WAVELENGTH_nm:
            scale = 10_000_000. / self.wavenums
        elif scale_type == Scale.WAVELENGTH_um:
            scale = 10_000. / self.wavenums
        else:
            scale = self.wavenums
        with open(path, 'w') as out:
            print(scale_type.value, *scale, sep=',', file=out)
            print(self.clss, *self.data, sep=',', file=out)

    def interpolate(self):
        pass


def scale_change(scale_type):
    """
    Define the function of wavenumbers recalculation.
    params
    scale_type: Scale - determines the scale units

    rtype: (str|float,) -> float
    """
    if scale_type == Scale.WAVELENGTH_nm:
         return lambda x: 10_000_000. / float(x)
    elif scale_type == Scale.WAVELENGTH_um:
        return lambda x: 10_000. / float(x)
    else:
        return lambda x: float(x) / 1.


def spectra_log(spectra_dict, path='log.txt'):
    """
    Types the spectra collection into the file by path.

    """
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
    plt.figure()
    base = ConvexHull(list(zip(x, y))).vertices
    # print(base)
    # base = np.roll(base, -base.argmin() - 1)
    # base = base[base.argmax():]
    base = np.roll(base, -base.argmax() - 1)
    base1 = base[base.argmin():]
    base2 = base[:base.argmin() + 1]
    # print(base1, base2)
    # print('base1' if y[base1[1]] < y[base2[1]] else 'base2')
    base1 = list(base1 if y[base1[1]] < y[base2[1]] else base2)
    base1 =  [len(x) - 1] + base1 + [0]
    # plt.plot(x, np.interp(x, x[base1], y[base1]), color='k')
    # print(x[base])
    # plt.plot(x[base1], y[base1], 'b--')
    # print(x[base1])
    new_y = y - np.interp(x, x[base1], y[base1])
    # plt.plot(x, new_y, color='g')
    # plt.show()
    return x, new_y

    # # Find the convex hull
    # v = ConvexHull(np.vstack((x, y)).T).vertices
    # # Rotate convex hull vertices until they start from the lowest one
    # v = np.roll(v, v.argmax())
    # # Leave only the ascending part
    # v = v[:v.argmax()]
    # # Create baseline using linear interpolation between vertices
    # return y - np.interp(x, x[v], y[v])


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
    """
    params
    path: str = 'data' - path to the root directory of spectra. If not defined, the path is the current directory
    classify: bool = False - if True, the classes are defined as the first-layer directory name;
    or paths aren't included in case they're in the root directory, else all paths are marked as '.'
    recursive: bool = False - if True, explores all the nested directories,
    else is limited with only 0,1-level directories
    rtype: List(Spectrum)
    """
    res = []
    for p, clss in read_data(**kwargs):
        res.append(Spectrum(path=p, clss=clss))
    return res


def get_spectra_dict(**kwargs):
    """
    Returns the dictionary: keys are spectra IDs and Spectrum objects as values.
    params
    path: str = 'data' - path to the root directory of spectra. If not defined, the path is the current directory
    classify: bool = False - if True, the classes are defined as the first-layer directory name;
    or paths aren't included in case they're in the root directory, else all paths are marked as '.'
    recursive: bool = False - if True, explores all the nested directories,
    else is limited with only 0,1-level directories
    rtype: Dict(int: Spectrum)
    """
    res = {}
    for p, clss in read_data(**kwargs):
        s = Spectrum(path=p, clss=clss)
        res[s.id] = s
    return res


def read_data(path='data', classify=False, recursive=False):
    '''
    path: str = 'data' - path to the root directory of spectra. If not defined, the path is the current directory
    classify: bool = False - if True, the classes are defined as the first-layer directory name;
    or paths aren't included in case they're in the root directory, else all paths are marked as '.'
    recursive: bool = False - if True, explores all the nested directories,
    else is limited with only 0,1-level directories
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
    """
    Check whether the file is a valid opus one.
    params:
    path: str - the path to the destination file
    rtype: bool
    """
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
    if isinstance(spectra, Spectrum):
        spectra = [spectra]
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
    # spc = Spectrum(wavenums=np.array(list(range(1, 11)), dtype=float),
    #                          data=np.array([
    #                              -10., -7., 4., -8., -6., 2., 3., 2., 7., 6.
    #                          ]))

    #plt.figure()
    spa = get_spectra_list(path='data', classify=True)
    show_spectra(spa[:10])
    # spa = [spc]
    # spc1 = spc * 1
    # spc1.data = np.exp( -1. / spc1.data)
    # spa.append(spc1)
    # spc2 = spc * 1
    # spc2.atr_to_absorbace()
    # spa.append(spc2)
    # show_spectra(spa)

    # for i, spc in enumerate(spa):
    #     spc.wavenums, spc.data = rubberband(spc.wavenums, spc.data)
    #     print(i)
    #     plt.figure()
    #     plt.plot(spc.wavenums, spc.data)
    #     plt.show()

    # show_spectra(spa)
    #plt.show()
    # plt.figure()
    # plt.plot(spc.wavenums, spc.data)
    # x = ConvexHull(list(iter(spc))).vertices
    # x = np.roll(x, -x.argmin()) # min at 0
    # # for i in x:
    # #     plt.axvline(spc.wavenums[i])
    # # x = x[:x.argmax() + 1]
    # print(x)
    # x = x[x.argmax() + 1:]
    #
    # print(*[spc.wavenums[i] for i in x])
    # y = spc.data[x]
    # w = spc.wavenums[x]
    #
    # plt.plot(w, y)
    # plt.plot(spc.wavenums, np.interp(spc.wavenums, spc.wavenums[x], y))
    # new_y = spc.data - np.interp(spc.wavenums, spc.wavenums[x], y)
    # plt.plot(spc.wavenums, new_y)
    #
    # # plt.plot(*rubberband(spc.wavenums, spc.data))
    # plt.show()






