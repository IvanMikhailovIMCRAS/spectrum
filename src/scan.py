from spectrum import Spectrum
import os
from filter import filter_opus
from enumerations import Scale
from miscellaneous import scale_change
import numpy as np

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
    read = read_data(**kwargs)
    if read:
        for p, clss in read:
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
    read = read_data(**kwargs)
    if read:
        for p, clss in read:
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

def read_columns(path, v_offset=0, delimiter=',', columns_indices=(0, 1), scale=Scale.WAVELENGTH_nm):
    with open(path, 'r') as inp:
        for _ in range(v_offset):
            next(inp)
        ws, ds = [], []
        for line in inp.readlines():
            try:
                cols = [n for i, n in enumerate(line.strip().split(delimiter)) if i in columns_indices]
                if len(cols) != 2:
                    continue
                w, d = [float(col.replace(',', '.')) for col in cols]
                w = scale_change(scale)(w)
                ws.append(w)
                ds.append(d)
            except:
                continue
        return np.array(ws), np.array(ds)






