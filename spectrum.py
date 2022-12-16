import numpy as np
import os
import matplotlib.pyplot as plt
import brukeropusreader as opus
from random import sample
import logging
from scipy.stats import mode



class Spectrum:
	spectrum_id = 0
	#error_log = os.path.join(os.getcwd(), 'errlog.txt')
	logging.basicConfig(level=logging.INFO,
						filemode='w',
						filename=os.path.join(os.getcwd(), 'log.txt'),
						format='%(asctime)s %(levelname)s %(message)s')

	def __init__(self, path, cls: str):
		self.id = Spectrum.spectrum_id
		self.path = path
		self.wavenums, self.data = Spectrum.read_opus(path)
		self.cls = cls
	
	def __len__(self):
		return len(self.wavenums)

	def __str__(self):
		return '\t'.join((str(self.id), self.path, self.cls))
  
	def __bool__(self):
		return len(self.wavenums) != 0
	
	def normalize(self):
		norm_coef = np.sqrt(np.sum(np.square(self.data)))
		self.data /= norm_coef

	def correct_baseline(self):
		pass

	def to_absorbace(self):
		pass

	def smooth(self):
		pass

	def get_derivative(self):
		pass

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
			return x, y
		
def create_matrix(raw_spectra):
	# преобразование спектров других длин ? 

	matrix = []
	if not raw_spectra:
		return np.zeros((1,1))
	length, _ = mode([len(x) for x in raw_spectra if len(x) != 0], keepdims=False )
	for raw_spectrum in raw_spectra:
		if raw_spectrum and len(raw_spectrum) == length:
			matrix.append(raw_spectrum.data)
		else:
			print(raw_spectrum.path)
	
	return np.array(matrix)
			
			
def get_spectra_list(**kwargs):
	res = []
	for path, cls in zip(*read_data(**kwargs)):
		res.append(Spectrum(path, cls))
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
	for ind, path in enumerate(paths):
		path_parts = path.split(os.sep)
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

	paths = [paths[i] for i, cls in enumerate(classes) if cls is not None]
	classes = [cls for cls in classes if cls is not None]

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
	classes = set(map(lambda x: x.cls, spectra))
	colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
	colors = dict(zip(classes, colors))
	plt.figure()
	lines = []
	clrs = []
	for spc in spectra:
		if spc:
			lines.append(plt.plot(spc.wavenums[::-1], spc.data[::-1], c=colors[spc.cls]))
			clrs.append(spc.cls)
		#print(cls, ': ', max(y))
	plt.legend(clrs)
	plt.xlabel('wavenumber, cm-1')
	plt.ylabel('intensity')
	plt.show()


if __name__ == '__main__':
	# dogs = get_spectra_list(path='Спектры сывороток животных/Собаки raw',classify=False, recursive=True)[:1]
	# turtle = get_spectra_list(path='Спектры сывороток животных/Черепаха raw',classify=False, recursive=True)[:1]
	# chickens = get_spectra_list(path='Спектры сывороток животных/Куры raw',classify=False, recursive=True)[:1]
	# animals = dogs + turtle + chickens
	spca = get_spectra_list(classify=False, recursive=True)
	print('Total number: ', len(spca))
	print("Matrix size: ", create_matrix(spca).shape)
	# show_spectra(spca)
	# for animal in animals:
	# 	animal.normalize()
	# show_spectra(animals)
 
	
	
	# for path, cls in paths:
	# 	spectra.append(Spectrum(path, cls))
	# print(create_matrix(spectra))
		
	# valids = open('delta/valids.txt', 'a')
	# invalids = open('delta/invalids.txt', 'a')

	# for path, cls in paths:
	# 	s = Spectrum(path, cls)
	# 	print(s, file=(valids if s else invalids))
	
	# valids.close()
	# invalids.close()

		 	
			



