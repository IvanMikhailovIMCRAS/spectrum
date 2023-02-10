from BaselineRemoval import BaselineRemoval 
import numpy as np
from scipy.spatial import ConvexHull
from scipy import sparse
from scipy.sparse.linalg import spsolve
# rubberband

def baseline_zhang(y, polynomial_degree=2):
    """
	adaptive iteratively reweighted Penalized Least Squares (airPLS) - Zhi-Min Zhang et.al
	https://pubs.rsc.org/is/content/articlelanding/2010/an/b922045c#!divAbstract
	"""
    baseObj = BaselineRemoval(y)
    return baseObj.ZhangFit()


def baseline_rubberband(x, y):
    base = ConvexHull(list(zip(x, y))).vertices
    base = np.roll(base, -base.argmax() - 1)
    base1 = base[base.argmin():]
    base2 = base[:base.argmin() + 1]
    base1 = list(base1 if y[base1[1]] < y[base2[1]] else base2)
    base1 =  [len(x) - 1] + base1 + [0]
    new_y = y - np.interp(x, x[base1], y[base1])
    return x, new_y


def baseline_alss(y, lam=1e6, p=1e-3, niter=10):
    """
	an algorithm called "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens
	https://stackoverflow.com/questions/29156532/python-baseline-correction-library
	"""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z
