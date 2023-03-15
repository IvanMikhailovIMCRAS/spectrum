# from spectrum import Spectrum
import numpy as np
import matplotlib.pyplot as plt
from enumerations import Scale

def show_spectra(spectra, save_path='', wavenumbers=None, legend=True):
    spectra = list(filter(lambda x: bool(x), spectra))
    if not spectra:
        return
    classes = list(sorted(set(map(lambda x: x.clss, spectra))))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    colors = dict(zip(classes, colors))
    plt.figure()
    lines = []
    clrs = []
    minx, maxx = 0, 10
    for spc in spectra:
        if spc:
            if wavenumbers:
                spc = spc.range(*wavenumbers)
                mi, ma = sorted((spc.wavenums[0], spc.wavenums[-1]))
                if minx == 0 and maxx == 10:
                    minx, maxx = mi, ma
                minx = min(mi, minx)
                maxx = max(ma, maxx)
            lines.append(plt.plot(spc.wavenums, spc.data, c=colors[spc.clss], linewidth=0.5))
            clrs.append(spc.clss)
    if len(clrs) > 1 and legend:
        plt.legend(clrs)
    plt.xlim(minx, maxx)
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
            

def heatmap(data, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.show()

    return im, cbar, ax

def auto_heatmap(spc, step=100):
    def each_to_each(spc):
        mtr = []
        for i in range(len(spc)):
            mtr.append(np.roll(spc.data, i))
        return np.vstack(mtr)
    corrcoefs = np.corrcoef(each_to_each(spc))
    *_, ax = heatmap(corrcoefs)
    ax.set_xticks(np.arange(0, len(spc), step), labels=spc.wavenums[::step])
    ax.set_yticks(np.arange(0, len(spc), step), labels=spc.wavenums[::step])
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    
    plt.show()
    
    
    
    
            
