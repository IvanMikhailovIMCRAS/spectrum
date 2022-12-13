import numpy as np
import os
import matplotlib.pyplot as plt
import brukeropusreader as opus
from random import sample
import logging



class Spectrum:
    #error_log = os.path.join(os.getcwd(), 'errlog.txt')
    logging.basicConfig(level=logging.INFO,
                        filemode='w',
                        filename=os.path.join(os.getcwd(), 'log.txt'),
                        format='%(asctime)s %(levelname)s %(message)s')

    def __init__(self, directory: str):
        self.directory = directory  # from which we load the spectra
        self.names = list()  # names of files
        self.paths = list()
        self.data = [[0, ], [0, ]]  # data storage: data[0] - wave-numbers; data[1:] - intensities
        self.X = np.zeros(shape=(1, 1))  # usable data-matrix for cleaning and processing
        self.freq_mask = np.array([True])  # mask for data clipping by frequency
        self.obs_mask = np.array([True])  # mask for data clipping by observations
        self.empty = True  # label that downloading from {directory} is failed
        
        # for current_dir, _, files in os.walk(self.directory):
        #     for file in files:
        #         self.download_one_data_file(current_dir, file)
        # if not self.empty:
        #     self.freq_mask = np.ones(len(self.data[0]), dtype=bool)
        #     self.obs_mask = np.ones(len(self.data), dtype=bool)
        #     self.recalc_X_under_masks()
        #     self.X = np.array(self.data)   
        
        
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
            if len(path_parts) == 3:
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

def read_opus(path):
    x, y = [], []
    try:
        logging.info(f"Trying to open file: {path}")
        file = opus.read_file(path)
        x = file.get_range()
        y = file['AB']
        logging.info(f'{path} is successfully read.')
    except Exception as err:
        logging.error(f'Error occured!', exc_info=True)
    finally:
        return x, y



def show_spectra(classified_spectra):
    classes = set(map(lambda x: x[1], classified_spectra))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    colors = dict(zip(classes, colors))
    plt.figure()
    lines = []
    clrs = []
    for spc, cls in classified_spectra:
        x, y = read_opus(spc)
        if len(x) != 0:
            lines.append(plt.plot(x, y, c=colors[cls]))
            clrs.append(cls)
        #print(cls, ': ', max(y))
    plt.legend(clrs)
    plt.xlabel('wavenumber, cm-1')
    plt.ylabel('intensity')
    plt.show()


if __name__ == '__main__':
    paths= list(zip(*read_data(classify=True, recursive=False)))
    paths = sample(paths, 3)
    show_spectra(paths)
    # for i, j in zip(*read_data(r"C:\Users\user\PycharmProjects\spectrum\data", classify=False, recursive=True)):
    #     print(j, i, sep='\t\t')

    
