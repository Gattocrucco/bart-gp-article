import warnings
import pathlib
import time
import collections

import polars as pl
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm

import bartreg
from rbartpackages import BayesTree, BART, bartMachine, dbarts
import nips

""" makes figure 12, takes about 10 minutes """

####### config #######

burnin = 1000
samples = 1000
trees = 1000
nthreads = 4
dbarts_single_chain = False # dbarts is able to parallelize within-chain, a bit slower though
skip_bayestree = False # because it's slow
dataset = 'Abalone' # a chipman paper dataset
refkey = 'GP' # method which is compared to the others
keys = ['BART', 'bartMachine', 'dbarts', 'BayesTree']

######################

root = pathlib.Path(__file__).parent
datadir = pathlib.Path('datasets') / 'nipsdata' / 'data'
seedseq = np.random.SeedSequence(202301172344)

warnings.filterwarnings('ignore', r'The resize_event function was deprecated')

#### load data ####

data = nips.Nips(datadir / dataset)
print(f'n_0 = {data.n}')
print(f'p_0 = {data.p}')

x_train = data.xtrain()
y_train = pl.Series(data.ytrain()) # BayesTree and bartMachine want a Series
x_test = data.xtest()
y_test = data.ytest()

#### list functions to call ####

bartfunc = dict(
    GP=bartreg.bart,
    dbarts=dbarts.bart,
    BART=BART.mc_gbart,
    BayesTree=BayesTree.bart,
    bartMachine=bartMachine.bartMachine,
)

#### determine calling arguments ####

def OLS_error_sdev(X, y):
    x = X.to_pandas()
    X = sm.add_constant(X)
    y = y.to_pandas()
    model = sm.OLS(y, X)
    result = model.fit()
    return np.sqrt(result.mse_resid)

bartkw_shared = dict(x_test=x_test, base=0.95, power=2, sigdf=3, sigquant=0.9, k=2)
bartkw_mcmc = dict(
    usequants=True,
    numcut=len(y_train),
        # equiv. to <large number>, but BART actually allocates this size, so
        # keep it as small as possible
    nskip=burnin,
    ndpost=samples,
    ntree=trees,
)

bartkw = {}

bartkw['GP'] = dict(ndpost=samples, **bartkw_shared)

bartkw_db = dict(nchain=nthreads, printcutoffs=5, **bartkw_shared, **bartkw_mcmc)
bartkw_db.update(
    base=float(bartkw_db['base']), # convert to float because dbarts complains about integers
    power=float(bartkw_db['power']),
    sigdf=float(bartkw_db['sigdf']),
    sigquant=float(bartkw_db['sigquant']),
    k=float(bartkw_db['k']),
    ndpost=bartkw_db['ndpost'] // bartkw_db['nchain'], # dbarts wants samples per chain
    nthread=bartkw_db['nchain'], # one thread per chain
)
if dbarts_single_chain:
    bartkw_db.update(
        ndpost=bartkw_mcmc['ndpost'],
        nchain=1,
        nthread=nthreads,
    )

bartkw['dbarts'] = bartkw_db

bartkw['BART'] = dict(
    mc_cores=nthreads,
    offset=(y_train.max() + y_train.min()) / 2, 
        # same as BayesTree and dbarts (presumed, I haven't checked their code)
    **bartkw_shared,
    **bartkw_mcmc,
)

bartkw['bartMachine'] = dict(
    num_trees=bartkw_mcmc['ntree'],
    num_burn_in=bartkw_mcmc['nskip'],
    num_iterations_after_burn_in=bartkw_mcmc['ndpost'],
    alpha=float(bartkw_shared['base']),
    beta=float(bartkw_shared['power']),
    k=float(bartkw_shared['k']),
    q=float(bartkw_shared['sigquant']),
    nu=float(bartkw_shared['sigdf']),
    num_cores=1, # do not parallelize because bartMachine eats a ton shit of RAM
    run_in_sample=False, # to speed up
    flush_indices_to_save_RAM=True, # no speed compromise according to docstring
    megabytes=5000, # increase java memory
)

bartkw['BayesTree'] = dict(printcutoffs=5, **bartkw_shared, **bartkw_mcmc)

if skip_bayestree:
    bartkw.pop('BayesTree')

#### functions to extract information in a homogeneous way ####

def get_yhat_test_mean(bart):
    if hasattr(bart, 'yhat_test_marg_mean'): # bartreg.bart
        return bart.yhat_test_marg_mean
    if hasattr(bart, 'yhat_test_mean'): # BayesTree, BART, dbarts
        return bart.yhat_test_mean

def get_yhat_test_sdev(bart):
    if hasattr(bart, 'yhat_test_marg_var'): # bartreg.bart
        return np.sqrt(bart.yhat_test_marg_var)
    return bart.yhat_test.std(axis=0) # BayesTree, BART, dbarts, bartMachine

def get_rmse(bart):
    yhat_test_mean = get_yhat_test_mean(bart)
    return np.sqrt(np.mean((y_test - yhat_test_mean) ** 2))

def get_sigest(bart):
    if hasattr(bart, 'sigest'):
        sigest = bart.sigest
        if hasattr(sigest, 'item'): # BayesTree, dbarts
            return sigest.item()
        return sigest # bartreg.bart
    if hasattr(bart, 'sig_sq_est'): # bartMachine
        return np.sqrt(bart.sig_sq_est).item()

def get_sigma(bart):
    if hasattr(bart, 'sigma'):
        return bart.sigma
    if hasattr(bart, 'get_sigsqs'): # bartMachine
        return np.sqrt(bart.get_sigsqs(plot_CI=False))

#### run bart ####

class Timer:

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.duration = time.perf_counter() - self.start

def make_seed(seedseq):
    return seedseq.generate_state(1).item() >> 1 # compatible with R

bart = {}
for label, kw in bartkw.items():

    print(f'\n######### {label} ##########')
    seedseq, seed = seedseq.spawn(2)
    seed = make_seed(seed)

    sigest = OLS_error_sdev(x_train, y_train)
    if label == 'bartMachine':
        kw.update(sig_sq_est=sigest ** 2)
    else:
        kw.update(sigest=sigest)
    
    with Timer() as timer:
        obj = bartfunc[label](x_train, y_train, seed=seed, **kw)
        if hasattr(obj, 'bart_machine_get_posterior'):
            post = obj.bart_machine_get_posterior(x_test)
            obj.yhat_test = post['y_hat_posterior_samples'].T
            obj.yhat_test_mean = post['y_hat']

    obj.yhat_test_mean = get_yhat_test_mean(obj)
    obj.yhat_test_sdev = get_yhat_test_sdev(obj)
    obj.rmse = get_rmse(obj)
    obj.sigest = get_sigest(obj)
    obj.sigma = get_sigma(obj)

    obj.time = timer.duration
    bart[label] = obj

    del obj # to free memory

#### print comparison table ####

table = collections.defaultdict(list)
for label, obj in bart.items():
    table['method'].append(label)
    table['rmse'].append(obj.rmse)
    table['time'].append(obj.time)
    table['sigest'].append(obj.sigest)
table = pl.DataFrame(table)
print()
print(table)

#### plot ####

plt.close('all')
figlabel = pathlib.Path(__file__).stem
fig, axs = plt.subplots(len(bart) - 1, 3,
    num=figlabel,
    figsize=[7.34, 8.8 ],
    layout='constrained',
    sharex='col',
    sharey='col',
    clear=True,
)

for (ax0, ax1, ax2), key in zip(axs, keys):
    plotkw = dict(color='black', alpha=0.1, marker='.', linestyle='')

    # posterior mean
    ax0.plot(bart[refkey].yhat_test_mean, bart[key].yhat_test_mean, **plotkw)
    ax0.set_ylabel(key)

    # posterior sdev of latent function
    ax1.plot(bart[refkey].yhat_test_sdev, bart[key].yhat_test_sdev, **plotkw)

    # posterior distr of error sdev
    ax2.hist(
        [bart[refkey].sigma, bart[key].sigma],
        label=[refkey, key],
        color=['black', '#f55'],
        bins='auto',
        density=True,
    )
    ax2.legend(loc='upper right', fontsize='small')

    ss = ax0.get_subplotspec()
    if ss.is_first_row():
        ax0.set_title(f'$E[y_\\mathrm{{test}} \\mid y_\\mathrm{{train}}]$')
        ax1.set_title(f'$\\mathrm{{Std}}[f(x_\\mathrm{{test}}) \\mid y_\\mathrm{{train}}]$')
        ax2.set_title('$p(\\sigma\\mid y_\\mathrm{train})$')
    if ss.is_last_row():
        ax0.set_xlabel(refkey)
        ax1.set_xlabel(refkey)
        ax2.set_xlabel('$\\sigma$')

    # plot bisector line
    for ax in ax0, ax1:
        x = ax.get_xlim()
        y = ax.get_ylim()
        lim = (min(x[0], y[0]), max(x[1], y[1]))
        ax.plot(lim, lim, '-', color='lightgray', zorder=-2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)

root.mkdir(parents=True, exist_ok=True)
fig.savefig(root / (fig.get_label() + f'-{dataset}-burnin{burnin}-samp{samples}-tree{trees}.pdf'))
fig.show()
