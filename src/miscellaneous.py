from enumerations import Scale
from numpy import square, sqrt, log, exp, zeros, std, mean

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


width_sigma = 2 * sqrt(log(2))  # * np.sqrt(2)
width_lambda = 2.


# def gauss(x, amp, mu, sigma):
#     return amp / sigma / np.sqrt(2.*np.pi) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def gauss(x, amp, mu, w):
    sigma = w / width_sigma
    return amp * exp(-square((x - mu) / sigma))


def lorentz(x, amp, x0, w):
    return amp / (square(2 * (x - x0) / w) + 1.)


def voigt(x, amp, x0, w, gauss_prop):
    # assert 0 <= gauss_prop <= 1
    return gauss_prop * gauss(x, amp, x0, w) + (1 - gauss_prop) * lorentz(x, amp, x0, w)

def summ_voigts(x, params):
    data = zeros(len(x))
    for amp, mu, w, g in params:
        data += voigt(x, amp, mu, w, g)
    return data

def n_sigma_filter(sequence, n=1):
    sigma = std(sequence)
    mu = mean(sequence)
    lo = mu - sigma * n
    hi = mu + sigma * n
    return [lo <= sequencei <= hi for sequencei in sequence]


# def pack_dictionary(d, order):
#     res = []
#     for


