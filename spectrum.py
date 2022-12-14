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

#add range
#range as a class?
#show specta - give intervals
# Проверить, почему не работает rubberband
# add associativity to the operations
# customize exceptions
# ids when operations on spectra -- do we need them?
# every list urn into dict

class Spectrum:
	ATR_to_AB = 1000
	spectrum_id = 0
	logging.basicConfig(level=logging.INFO,
						filemode='w',
						filename=os.path.join(os.getcwd(), 'errlog.txt'),
						format='%(asctime)s %(levelname)s %(message)s')

	def __init__(self, wavenums=[], data=[], path='', clss: str=''):
		self.path = path #наследуем от себя 
		if path:
			self.wavenums, self.data = Spectrum.read_opus(path)
		else:
			self.wavenums, self.data = wavenums, data
		Spectrum.spectrum_id += 1
		self.id = Spectrum.spectrum_id
		self.clss = clss
		self.step = abs(self.wavenums[1] - self.wavenums[0]) if len(self.wavenums) > 1 else 0


	def __len__(self):
		return len(self.wavenums)

	def __str__(self):
		return '\t'.join((str(self.id), self.path, self.clss))

	def __bool__(self):
		return len(self.wavenums) != 0

	def __getitem__(self, ind):
		return self.wavenums[ind], self.data[ind]

	def range(self, bigger, lesser):
		start_ind = int((self.wavenums[0] - bigger) / self.step)  if bigger < self.wavenums[0] else 0
		stop_ind = len(self) - int((lesser - self.wavenums[-1]) / self.step)  if lesser > self.wavenums[-1] else len(self)
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

	def get_extrema(self, locals=True, minima=False):
		indices = []
		wavenums = []
		iextr = 1

		if minima:
			f = lambda i: self.data[i - 1] > self.data[i] and self.data[i] < self.data[i + 1]
			comp = lambda i, iextr: self.data[i] < self.data[iextr]
		else:
			f = lambda i: self.data[i - 1] < self.data[i] and self.data[i] > self.data[i + 1]
			comp = lambda i, iextr: self.data[i] > self.data[iextr]

		for i in range(1, len(self) - 2):
			if f(i):
				indices.append(i)
				wavenums.append(self.wavenums[i])
				iextr = i if comp(i, iextr) else iextr
		if not locals:
			return iextr, self.wavenums[iextr]
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
		self.wavenums, self.data = Spectrum.read_opus(self.path)

	def __add__(self, other):
		s = deepcopy(self)
		Spectrum.spectrum_id += 1
		s.id = Spectrum.spectrum_id
		if isinstance(other, (float, int)):
			s.data += other
		elif hasattr(other, '__iter__') and self.is_comparable(other):
			s.data += other.data
		else:
			raise Exception('Spectra should have the same wavenumber ranges!')
		return s

	def __mul__(self, other):
		s = deepcopy(self)
		Spectrum.spectrum_id += 1
		s.id = Spectrum.spectrum_id
		if isinstance(other, (float, int)):
			s.data *= other
		elif hasattr(other, '__iter__') and self.is_comparable(other):
			s.data *= other.data
		else:
			raise Exception('Spectra should have the same wavenumber ranges!')
		return s

	def __sub__(self, other):
		'''
		resulting spectrum inherits all the attributes of the first argument
		'''
		s = deepcopy(self)
		Spectrum.spectrum_id += 1
		s.id = Spectrum.spectrum_id
		if isinstance(other, (float, int)):
			s.data -= other
		elif hasattr(other, '__iter__') and self.is_comparable(other):
			s.data -= other.data
		else:
			raise Exception('Spectra should have the same wavenumber ranges!')
		return s

	def is_comparable(self, other):
		return len(self) == len(other) \
			and self.wavenums[0] == other.wavenums[0] \
				and self.wavenums[-1] == other.wavenums[-1]
		

	def change_size(self, sample):
		size = len(sample)
		sample_minval = sample.wavenums[0]
		sample_maxval = sample.wavenums[-1]
		spc_minval = self.wavenums[0]
		spc_maxval = self.wavenums[-1]

		if sample_maxval >= spc_maxval and spc_minval >= sample_minval: # s lies inside the spectrum
			for i in range(len(self)):
				pass

	@staticmethod
	def read_opus(path):
		x, y = [], []
		try:
			file = opus.read_file(path)
			x = file.get_range()
			y = file['AB']
		except Exception as err:
			logging.error(f'Reading {path}', exc_info=True)
		finally:
			if len(x) > 1:
				return x[:-1], y[:-1]
			return x, y

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


def get_spectra_list(**kwargs):
	res = []
	for p, clss in zip(*read_data(**kwargs)):
		res.append(Spectrum(path=p, clss=clss))
	return res

def get_spectra_dict(**kwargs):
	res = {}
	for p, clss in zip(*read_data(**kwargs)):
		s = Spectrum(path=p, clss=clss)
		res[s.id] = s
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


def show_spectra(spectra, path='', wavenumbers=None):
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
	if path:
		plt.savefig(path)
	else:
		plt.show()


if __name__ == '__main__':
	print('HI')
	# f = opus.read_file(r'C:\Users\user\PycharmProjects\spectrum\SD10.30')
	# print(*[it for it in f.items()], sep='\n\n')
	# sp = get_spectra_list(
	# 	recursive=False)[0]
	# for spec in sp:
	# 	spec.select([3000, 2000], [1110, -1])
	# show_spectra([sp])
	#sp = sp.range(4000, 2578)
	# sp = get_spectra_list(
	# 	path='Спектры сывороток животных/Черепаха raw', recursive=False)[0]
	spca = get_spectra_dict(path=r'Спектры сывороток животных/Черепаха raw', recursive=False)
	spectra_log(spca)
	# sp_rb = sp * 1
	# sp_rb.correct_baseline()
	# sp_rb.clss = 'RB correction'
	# show_spectra([sp, sp_rb])
	
	# sp.reset()
	# show_spectra([sp])


	
