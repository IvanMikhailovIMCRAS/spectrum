from copy import deepcopy
import numpy as np
# import os
# import matplotlib.pyplot as plt
import brukeropusreader as opus
# from random import sample
# import logging
# from scipy import sparse
# from scipy.sparse.linalg import spsolve
# from scipy.spatial import ConvexHull
# from scipy.linalg import cholesky
from scipy.signal import savgol_filter
#from BaselineRemoval import BaselineRemoval
from enumerations import NormMode, BaseLineMode, Scale
from exceptions import SpcCreationEx, SpcReadingEx, SpcChangeEx
from baseline import baseline_alss, baseline_zhang, baseline_rubberband


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
            raise SpcCreationEx

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
                raise SpcChangeEx
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
                raise SpcChangeEx
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
        except SpcReadingEx as err:
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
    
if __name__ == '__main__':
    print('Hi')