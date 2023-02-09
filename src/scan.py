from spectrum import Spectrum
import os
from filter import filter_opus

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