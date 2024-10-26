import warnings

import lsqfitgp as lgp
import numpy as np
import joblib
from scipy.stats import qmc
from scipy import stats
from numpy.lib import recfunctions
from matplotlib import pyplot as plt

import bart
import textbox

""" generates figure 7, takes about 1 hour """

p = 10
n = 50
alpha = 0.7
beta = 1.3
maxd = 4
N = 100 # number of points whose covariance matrix is evaluated
nmc = 100000 # samples drawn from the prior
m = 10000
use_weights = True

################

# check parameters
assert nmc >= N
assert n >= 1
assert p >= 1
assert 0 <= maxd <= 6
assert beta >= 0
assert 0 <= alpha <= 1

cache = joblib.Memory('checkprior-cache')

@cache.cache
def sample_bart_prior(*args, **kw):
    return bart.sample_bart_prior(*args, **kw)

@cache.cache
def bart_correlation(X, **kw):
    gp = lgp.GP(lgp.BART(**kw), checksym=False, checkpos=False)
    x = recfunctions.unstructured_to_structured(X)
    gp = gp.addx(lgp.StructuredArray(x), 0)
    return np.array(gp.prior(0, raw=True))

# define data points (just for prior hyperparameters tuning)
X_data = np.repeat(np.linspace(0, 1, n + 1)[:, None], p, 1)
y_data = np.zeros(n + 1)
y_data[:2] = [-1, 1]

# sample quasi-random test points
rng = qmc.Sobol(p, seed=202211141311 + p + N)
warnings.filterwarnings('ignore', r'The balance properties of Sobol')
X_test = rng.random(N)

# generate weights, use integers for legibility
if use_weights:
    gen = np.random.default_rng(202211281634)
    weights = gen.integers(1, 10, size=p)
else:
    weights = np.ones(p, int)

# sample from prior and compute sample covariance matrix
# with max(y) = 1, min(y) = -1, and k = 1, the prior is standardized
samples = sample_bart_prior(X_data, X_test, y_data, nmc, seed=202211121317 + n + p + nmc + m + hash(alpha) + hash(beta), error=False, alpha=alpha, beta=beta, m=m, k=1, weights=weights)
# samples -= np.mean(samples, axis=0)
X = samples.T @ samples
samplecov = X / (nmc - 1)
# samplesdev = np.sqrt(np.diag(samplecov))
# samplecov /= np.outer(samplesdev, samplesdev)

# compute covariance matrix with approximated kernel
splits = lgp.BART.splits_from_coord(X_data)
kernelcov_lower = bart_correlation(X_test, splits=splits, alpha=alpha, beta=beta, maxd=maxd, gamma=0, weights=weights)
kernelcov_upper = bart_correlation(X_test, splits=splits, alpha=alpha, beta=beta, maxd=maxd, gamma=1, weights=weights)
kernelcov = (kernelcov_lower + kernelcov_upper) / 2
maxwidth = np.max(kernelcov_upper - kernelcov_lower)

# estimate error on sample cov with wishart distribution centered on kernel cov
# samples.T @ samples ~ Wishart(V, nmc)
# Wishart mean = nmc V, where V = true covariance matrix
# Wishart variance = nmc (V*V + outer(diag(V)))
V = kernelcov
diagV = np.diag(V)
Wvar = nmc * (np.square(V) + np.outer(diagV, diagV))
samplecov_err = np.sqrt(Wvar) / (nmc - 1)

# Do a likelihood ratio test for the sample covariance matrix with the
# Wishart distribution. Note that the distribution is Wishart under the
# assumption of multivariate Normality, so with an infinite number of trees.
class wishart:
    chi2 = 2 * (stats.wishart.logpdf(X, nmc, X / nmc) - stats.wishart.logpdf(X, nmc, V))
    dof = len(X) * (1 + len(X)) / 2
    pvalue = stats.chi2.sf(chi2, dof)
    chi2dof = chi2 / dof

# Do a likelihood ratio test directly for the multivariate Normal on the
# samples. (The results should be exactly equal to the Wishart test.)
class normal:
    L = stats.multivariate_normal.logpdf
    chi2 = 2 * (L(samples, cov=X / nmc) - L(samples, cov=kernelcov)).sum(0)
    dof = len(X) * (1 + len(X)) / 2
    pvalue = stats.chi2.sf(chi2, dof)
    chi2dof = chi2 / dof

# flatten and sort covariances for plotting
def covflatas(cov0, *covs):
    t = np.triu_indices(len(cov0))
    isort = np.argsort(cov0[t])
    out = []
    for cov in covs:
        out.append(cov[t][isort])
    return tuple(out)
kernelcov_flat, kernelcov_lower_flat, kernelcov_upper_flat, samplecov_flat, samplecov_err_flat = covflatas(
    kernelcov,
    kernelcov, kernelcov_lower, kernelcov_upper, samplecov, samplecov_err,
)

# plot covariances
warnings.filterwarnings('ignore', r'The resize_event function was deprecated')
plt.close('all')
fig, ax = plt.subplots(num='checkprior', layout='constrained')

abscissa = np.arange(len(kernelcov_flat))
center = kernelcov_flat
band0 = -center + kernelcov_lower_flat - samplecov_err_flat
band1 = -center + kernelcov_lower_flat
band2 = -center + kernelcov_upper_flat
band3 = -center + kernelcov_upper_flat + samplecov_err_flat
awish = ax.fill_between(abscissa, band0, band3, label='$\\pm$ Wishart sdev', facecolor='#f88')
akern = ax.fill_between(abscissa, band1, band2, label='Kernel bound', facecolor='black')
acov, = ax.plot(abscissa, -center + samplecov_flat, '.k', markersize=1, label='Sample covariance')

textbox.textbox(ax, f"""\
p = {p}, n = {n}
$\\alpha$ = {alpha}, $\\beta$ = {beta}
w = {weights}
{N} test points, {wishart.dof:.0f} covariances
kernel: D = {maxd}
trees: {nmc} samples, m = {m}
Wishart LR test: $\\chi^2/\\nu$ = {wishart.chi2dof:#.2g}, $p$ = {wishart.pvalue:.0%}""", 'lower left', bbox=dict(alpha=0.9))
ax.legend(handles=[akern, awish, acov], loc='upper left')
ax.set_ylabel('Covariance (centered on kernel)')
ax.set_xlabel('Covariance matrix entry sorted by kernel value')

fig.savefig(fig.get_label() + '.pdf')
fig.show()
