# from spectrum import Spectrum
import numpy as np
import matplotlib.pyplot as plt
from enumerations import Scale
from miscellaneous import voigt


def show_spectra(spectra, save_path='', wavenumbers=None):
    if not spectra:
        return
    classes = list(sorted(set(map(lambda x: x.clss, spectra))))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    colors = dict(zip(classes, colors))
    plt.figure(figsize=(20, 8))
    class_color = dict.fromkeys(classes, None)
    for spc in spectra:

        if wavenumbers:
            spc = spc.range(*wavenumbers)
        line = plt.plot(spc.wavenums, spc.data, c=colors[spc.clss], linewidth=0.5)
        class_color[spc.clss] = line[0]

    # clrs = list(set(clrs))
    # font = {'family':'serif','color':'black','size':18}
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    labels, handlers = zip(*class_color.items())
    if len(labels) > 1:
        plt.legend(handlers, labels, loc=0)
    spectrum = spectra[-1]
    # if len(spectrum) > 1:
    #     plt.xlim(spectrum.wavenums[0], spectrum.wavenums[-1])
    # print(spectrum.wavenums[0], spectrum.wavenums[-1])

    plt.xlabel('Wavenumber, cm-1')
    plt.ylabel('ATR units')
    if save_path:
        plt.savefig(fname=save_path, dpi=600)
    else:
        plt.show()


def show_curve_approx(spc, params, *, path=None):
    x = spc.wavenums
    plt.plot(x, spc.data)

    for amp, mu, w, g in params:
        plt.plot(x, voigt(x, amp, mu, w, g))
    plt.xlabel('???')
    plt.ylabel('Intensity')
    if path:
        plt.savefig(path)
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
            cbar_kw=None, cbarlabel="", path='', **kwargs):
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
    if path:
        plt.savefig(path, dpi=600)
    else:
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

def plot_margins(X, y, margins, path='', cm=None):
    if not cm:
        cm = plt.cm.get_cmap('tab20')
    plt.figure(figsize=(20, 16))
    plt.axhline(0)
    lab = ''
    counter = 0
    for i, label in enumerate(y):
        if lab != label:
            lab = label
            counter += 1
            plt.plot(X[i, 0], margins[i], 'o', label=lab, color=cm.colors[counter])
        else:
            plt.plot(X[i, 0], margins[i], 'o', color=cm.colors[counter])
    if path:
        plt.savefig(path)
    else:
        plt.show()
