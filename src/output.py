from spectrum import Spectrum
import numpy as np
import matplotlib.pyplot as plt
from enums import Scale

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
        
def spectra_log(spectra_dict, path='log.txt'):
    """
    Types the spectra collection into the file by path.

    """
    with open(path, 'w') as f:
        for spc in spectra_dict:
            print(spectra_dict[spc], file=f)
            
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