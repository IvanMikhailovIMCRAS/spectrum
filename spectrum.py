import numpy as np
import os
import matplotlib.pyplot as plt
import brukeropusreader as opus


class Spectrum:
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
        
        
def read_data(path, classify=False, recursive=False):
    '''
    rtype: (List[paths: str], List[classes: str])
    ''' 
    paths, classes = [], []
    
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            paths.append(os.path.join(dirpath, filename))
    paths = list(filter(filter_opus, paths))
    mask = [True] * len(paths)
    if not paths:
        return paths, classes
    
    delimiter = '\\' if '\\' in paths[0] else '/'
    if not classify:
        classes = [path] * (len(paths))
        if not recursive:
            for i, path in enumerate(paths):
                path = path.split(delimiter)
                if len(path) == 2:
                    mask[i] = False
    else:
        for i, path in enumerate(paths):
            path = path.split(delimiter)
            if len(path) == 2:
                classes.append(None)
                mask[i] = False
            elif len(path) == 3:
                classes.append(path[1])
            elif recursive:
                classes.append(path[1])
            else:
                classes.append(None)
                mask[i] = False
    assert len(paths) == len(classes) == len(mask), "read_data, lengths of lists must be equal!"
    paths = [paths[i] for i in range(len(paths)) if mask[i]]
    classes = [classes[i] for i in range(len(classes)) if mask[i]] 
    return paths, classes
           
def filter_opus(path):
    ext = path[path.rfind('.') + 1:]
    if not ext.isdigit():
        return False
    try:
        f = open(path, 'r')
        f.read()
        return False
    except:
        return True
    finally:
        f.close()
        
    
    
        
         
if __name__ == '__main__':
     
    for i, j in zip(*read_data(r'delta', classify=False, recursive=False)):
        print(j, i, sep='\t\t')
    
    
	
        
        
        