# from spectrum import Spectrum
import numpy as np
import matplotlib.pyplot as plt
from enumerations import Scale

def show_spectra(spectra, save_path='', wavenumbers=None):
    if not spectra:
        return
    classes = list(sorted(set(map(lambda x: x.clss, spectra))))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    colors = dict(zip(classes, colors))
    plt.figure(figsize=(20,8))
    lines = []
    clrs = []
    for spc in spectra:
        if spc:
            if wavenumbers:
                spc = spc.range(*wavenumbers)
            lines.append(plt.plot(spc.wavenums, spc.data, c=colors[spc.clss], linewidth=0.5))
            clrs.append(spc.clss)
    clrs = list(set(clrs))
    #font = {'family':'serif','color':'black','size':18} 
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    if len(clrs) > 1:
        plt.legend(clrs, loc=0)
    plt.xlim(spc.wavenums[0], spc.wavenums[-1])
    plt.xlabel('Wavenumber, cm-1')
    plt.ylabel('ATR units')
    if save_path:
        plt.savefig(fname=save_path, dpi=600)
    else:
        plt.show()
        
def spectra_log(spectra_dict, path='log.txt'):
    """
    Types the spectra collection into the file by path.

    """
    with open(path, 'w') as f:
        for spc in spectra_dict:
            print(spectra_dict[spc], file=f)
            
