import functools

import numpy as np

""" module to compute weighted quantiles of masked arrays """

def _wquantile(a, w, q):
    i = np.argsort(a)
    c = np.cumsum(w[i])
    return a[i[np.searchsorted(c, q * c[-1])]]

@functools.partial(np.vectorize, signature='(a),(a),(),()->(),()')
def _wquantile_masked(a, w, q, minsize):
    mask = np.ma.getmaskarray(a) | np.ma.getmaskarray(w)
    a = a.data[~mask]
    w = w.data[~mask]
    if len(a) < minsize:
        return np.nan, True
    return _wquantile(a, w, q), False

def wquantile(a, w, q, minsize=1):
    a = np.asanyarray(a)
    w = np.asanyarray(w)
    am = hasattr(a, 'mask')
    wm = hasattr(w, 'mask')
    a = np.ma.asarray(a)
    w = np.ma.asarray(w)
    res, mask = _wquantile_masked(a, w, q, minsize)
    if am or wm:
        return np.ma.array(res, mask=mask)
    else:
        assert np.all(mask)
        return res
