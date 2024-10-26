import pathlib
import warnings

import lsqfitgp as lgp
import joblib
from scipy.stats import qmc
import numpy as np
import pandas as pd

import wquantile

""" computes the BART kernel at various accuracies, takes O(days) """

alpha = [0.01, 0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
beta = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 3.5, 4]
p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n = 10
N = 250
reps = [2, 5]
normalize = True
maxdepth = 5
depth_target = 2

root = pathlib.Path(__file__).parent

########################

warnings.filterwarnings('ignore', r'The balance properties of Sobol')
warnings.filterwarnings('ignore', r'invalid value encountered in divide')
warnings.filterwarnings('ignore', r'divide by zero encountered in divide')

# cached function to compute BART correlation
memory = joblib.Memory(root / 'testnd2-cache')
@memory.cache
def bart_correlation(*args, **kw):
    return np.array(lgp.BART.correlation(*args, **kw))

# sample x values
rng = qmc.Sobol(2 * max(p), seed=202212041109)
xy = rng.random(len(p) * len(alpha) * len(beta) * N)
xy = xy.reshape(len(p), len(alpha), len(beta), N, max(p), 2)

# discretize x
xy = xy * n // 1
assert np.max(xy) == n - 1
xy = xy.astype(int)

# transform x to split counts
x = xy[..., 0]
y = xy[..., 1]
before = np.minimum(x, y)
between = np.abs(x - y)
after = n - 1 - np.maximum(x, y)
assert np.all(before + between + after == n - 1)
unit = np.arange(before[..., 0].size).reshape(before[..., 0].shape)
idx = np.arange((maxdepth + 1) * (1 + len(reps)) * unit.size)

# make arrays broadcastable
#                                  d     r     p     a     b     u  covar
depth = np.arange(maxdepth + 1)[   :, None, None, None, None, None]
r =        np.array([0, *reps])[None,    :, None, None, None, None]
p =                 np.array(p)[None, None,    :, None, None, None]
alpha =         np.array(alpha)[None, None, None,    :, None, None]
beta =           np.array(beta)[None, None, None, None,    :, None]
before =                 before[None, None,    :,    :,    :,    :, :]
between =               between[None, None,    :,    :,    :,    :, :]
after =                   after[None, None,    :,    :,    :,    :, :]
unit =                     unit[None, None,    :,    :,    :,    :]

# set splits to zero to enforce p
for ip, pp in enumerate(p.squeeze()):
    idx = np.s_[:, :, ip, :, :, :, pp:]
    before[idx] = 0
    between[idx] = 0
    after[idx] = 0

# convert to float32 for faster calculation
alpha = alpha.astype('f4')
beta = beta.astype('f4')

def make_upper_depth_kw(d, r):
    reset = [i * d for i in range(1, r)]
    assert len(reset) == r - 1
    return dict(maxd=r * d, reset=reset)

# compute correlation
correlations = []
for maxd in depth.squeeze():
    corr_down = bart_correlation(before, between, after, alpha=alpha, beta=beta, gamma=0, maxd=maxd, intercept=not normalize)
    corr_up = []
    for rep in reps:
        corr = bart_correlation(before, between, after, alpha=alpha, beta=beta, gamma=1, **make_upper_depth_kw(maxd, rep), intercept=not normalize)
        corr_up.append(corr)
    corr = np.concatenate([corr_down, *corr_up], axis=1)
    assert corr.shape[1] == r.size
    assert corr.dtype == 'f4'
    correlations.append(corr)
correlations = np.concatenate(correlations, axis=0)

# compute derived quantities
imaxr = np.argmax(r)
lower_precise = correlations[-1:, :1]
upper_precise = correlations[-1:, imaxr:imaxr + 1]
width_precise = upper_precise - lower_precise

lower3 = correlations[-3:-2, :1]
upper3 = correlations[-3:-2, imaxr:imaxr + 1]
width3 = upper3 - lower3

# precise = (lower_precise + upper_precise) / 2
# precise = lower_precise / (1 - (upper_precise - lower_precise))

precise = (width3 * lower_precise - width_precise * lower3) / (width3 - width_precise)
precise = np.where(width_precise < 1e-5, (lower_precise + upper_precise) / 2, precise)

lower = correlations[:, :1]
upper = correlations[:, 1:]
r_upper = r[:, 1:]

# interpolation in proper bounding interval
width = upper - lower
gamma = (precise - lower) / width
gamma = np.where(np.isinf(gamma), np.nan, gamma)

# how many covariates have x == y
fzero = (np.sum(between == 0, axis=-1) - (between.shape[-1] - p)) / p

# compute "lower-upper" bound for target parameters
corr_up = []
for rep in reps:
    corr = bart_correlation(before, between, after, alpha=alpha, beta=beta, gamma=0, **make_upper_depth_kw(depth_target, rep), intercept=not normalize)
    corr_up.append(corr)
(depth_target_idx,), = np.nonzero(np.squeeze(depth) == depth_target)
lower_upper = np.full_like(upper, np.nan)
lower_upper[depth_target_idx] = np.concatenate(corr_up, axis=1)

# "interpolation" within upper bound formula only, can be negative
width_upper = upper - lower_upper
gamma_upper = (precise - lower_upper) / width_upper
gamma_upper = np.where(np.isinf(gamma_upper), np.nan, gamma_upper)

def average_gamma(gamma, width):
    """ median of gamma, weighted by width, handles nans """
    gamma = np.ma.fix_invalid(gamma)
    threshold = precise * np.finfo(gamma.dtype).eps
    width = np.ma.masked_less(width, threshold)
    gammaavg = wquantile.wquantile(gamma, width, 0.5, N / 2)[..., None]
    ok = np.ma.median(width / threshold, axis=-1, keepdims=True) >= 8
    return np.ma.masked_where(~ok, gammaavg)

# remove masks because broadcast_arrays does not support masks even if subok=True
gammaavg = average_gamma(gamma, width).filled(np.nan)
gammaavg_upper = average_gamma(gamma_upper, width_upper).filled(np.nan)

# some cross-checks
np.testing.assert_array_max_ulp(precise, np.clip(precise, lower_precise, upper_precise)) # lower_precise <= precise <= upper_precise
np.testing.assert_array_max_ulp(upper, np.maximum(upper, np.where(np.isnan(lower_upper), upper, lower_upper))) # lower_upper <= upper
np.testing.assert_array_max_ulp(np.broadcast_to(lower, upper.shape), np.minimum(lower, np.where(np.isnan(lower_upper), lower, lower_upper))) # lower <= lower_upper
np.testing.assert_array_max_ulp(gammaavg, np.clip(gammaavg, 0, 1)) # 0 <= gammaavg <= 1
np.testing.assert_array_max_ulp(gammaavg_upper, np.minimum(1, gammaavg_upper)) # gammaavg_upper <= 1
# I don't check gamma directly because it's often out of bounds due to numerical error

# put everything in a table
names = ['depth', 'r_upper', 'p', 'alpha', 'beta', 'unit', 'lower', 'upper', 'width', 'gammaavg', 'lower_upper', 'width_upper', 'gammaavg_upper', 'lower_precise', 'upper_precise', 'width_precise', 'precise', 'fzero']
arrays = np.broadcast_arrays(*map(eval, names))
data = pd.DataFrame({n: a.reshape(-1) for n, a in zip(names, arrays)})

# save to file
file = root / 'testnd2.feather'
print(f'write {file}...')
data.to_feather(file)
