import pathlib
import time
import datetime
import sys
import gc
import re
import contextlib
import warnings

import tqdm
import numpy as np
import polars as pl
import gvar
import lsqfitgp as lgp
from scipy import stats, special
import statsmodels.api as sm
import jax

import bartreg
from rbartpackages import BART, dbarts
import nips

""" run benchmark on Chipman (2010) datasets, takes about 1 day """

redo_methods = [
]
redo_datasets = [
]
methods = [
    'gp-ru-2-5',
    'mcmc',
    'gp-eb-2-5-ru',
    'gp-eb-2-5-mcmc',
    'mcmc-cv',
    'mcmc-xcv',
]
nsplits = 20 # 1 to 20
bart_hypers = dict(base=0.95, power=2., sigdf=3., sigquant=0.9, k=2.)
mcmc_kw_shared = dict(
    usequants=True, # numcut specified later because data dependent
    nskip=1000,
    ntree=200,
    # timeout=60,
    # retries=5,
)
ndpost = 1000
nchain = 4
kw_BART = dict(
    ndpost=ndpost,
    mc_cores=nchain,
    nice=0, # high process priority
    **mcmc_kw_shared,
)
kw_dbarts = dict(
    ndpost=ndpost // nchain,
    nchain=nchain,
    nthread=nchain,
    keeptrainfits=False,
    **mcmc_kw_shared,
)
gphyp_hyperprior = lgp.copula.makedict({
    'base': lgp.copula.beta(2, 1),
    'power': lgp.copula.invgamma(1, 1),
    'log(k)': gvar.gvar(np.log(2), 2),
})
baseseed = 202410171543

############################

# fixed config
datadir = 'datasets/nipsdata/data'
tablefile = 'comp42/comp42.npy'
logfilename = 'comp42/comp42-log.txt'
warnings.filterwarnings('ignore', r'os\.fork\(\) was called\. os\.fork\(\) is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock\.')

# make settings table
datadirs = sorted(p for p in pathlib.Path(datadir).iterdir() if p.is_dir())
ddf = lambda **kw: pl.DataFrame(kw)
settings = (
    ddf(split=np.arange(nsplits))
    .join(ddf(dataset=[p.name for p in datadirs], source=list(map(str, datadirs))), how='cross')
    .join(ddf(method=methods), how='cross')
)

# produce deterministic random seeds
def asbytes(value):
    return np.array([value]).view(np.uint8)
seeds = []
for entry in settings.iter_rows(named=True):
    entropy = np.concatenate(list(map(asbytes, [
        baseseed,
        entry['dataset'],
        entry['split'],
        entry['method'],
    ])))
    seed = np.random.SeedSequence(entropy)
    seeds.append(seed.generate_state(1, np.uint64).item())
settings = settings.with_columns(pl.Series('seed', seeds, pl.UInt64))

# load eventual preexisting table of settings and results
oldtable = pl.DataFrame([
    s[:0] for s in settings
] + [
    pl.Series('done', [], pl.Boolean),
    pl.Series('n_train', [], pl.Int64),
    pl.Series('n_test', [], pl.Int64),
    pl.Series('p', [], pl.Int64),
    pl.Series('alpha', [], pl.Float64),
    pl.Series('beta', [], pl.Float64),
    pl.Series('nu', [], pl.Float64),
    pl.Series('q', [], pl.Float64),
    pl.Series('k', [], pl.Float64),
    pl.Series('sigma', [], pl.Float64),
    pl.Series('m', [], pl.Int64),
    pl.Series('coverage_90_normal', [], pl.Float64),
    pl.Series('coverage_50_normal', [], pl.Float64),
    pl.Series('coverage_90_quantiles', [], pl.Float64),
    pl.Series('coverage_50_quantiles', [], pl.Float64),
    pl.Series('rmse', [], pl.Float64),
    pl.Series('logloss', [], pl.Float64),
    pl.Series('exp_logloss', [], pl.Float64),
    pl.Series('logloss_gp', [], pl.Float64),
    pl.Series('exp_logloss_gp', [], pl.Float64),
    pl.Series('time', [], pl.Float64),
])
tablepath = pathlib.Path(tablefile)
if tablepath.exists():
    table = np.load(tablepath)
    table = pl.DataFrame(table)
    shared_columns = list(set(table.columns) & set(oldtable.columns))
    oldtable = oldtable.join(table, on=shared_columns, how='full', coalesce=True)
    
# merge settings with already computed results
table = (
    settings
    .with_columns(todo=True)
    .join(oldtable, on=settings.columns, how='full', coalesce=True)
)
todo = table['todo'].fill_null(False)
table = table.drop('todo')
table = table.with_columns(pl.col('done').fill_null(False))
assert set(table.columns) == set(oldtable.columns)

# mark something to be done again even if it is already in the table
if redo_methods or redo_datasets:
    domethods = f'methods [{", ".join(redo_methods)}]' if redo_methods else ''
    dodatasets = f'datasets [{", ".join(redo_datasets)}]' if redo_datasets else ''
    join = ' for ' if domethods and dodatasets else ''
    confirm = input(f'WARNING: redoing {domethods}{join}{dodatasets}, confirm? [y]/n: ')
    if confirm not in ('', 'y', 'Y'):
        sys.exit()
    cond = True
    if redo_methods:
        cond = cond & pl.col('method').is_in(redo_methods)
    if redo_datasets:
        cond = cond & pl.col('dataset').is_in(redo_datasets)
    table = table.with_columns((pl.col('done') & ~cond).alias('done'))

# convert to structured numpy array
def getstrtype(col):
    maxlength = table[col].str.len_chars().max()
    return f'U{maxlength}'
assert table['seed'].null_count() == 0
table = table.with_columns([
    pl.col(col).fill_null(-1)
    for col, dtype in zip(table.columns, table.dtypes)
    if dtype.is_integer() and col != 'seed'
]) # fill integer nulls because to_pandas() converts them to floats to have nan
assert table['seed'].dtype == pl.UInt64
table = table.to_pandas().to_records(index=False, column_dtypes={
    col: getstrtype(col)
    for col, dtype in zip(table.columns, table.dtypes)
    if dtype == pl.Utf8
})
assert table['seed'].dtype == np.uint64

# backup old data and save
if tablepath.exists():
    now = datetime.datetime.now()
    date = now.isoformat(sep='-', timespec='minutes').replace(':', '')
    newfilename = f'{tablepath.stem}-backup-{date}{tablepath.suffix}'
    tablepath.rename(tablepath.parent / 'comp42-backups' / newfilename)
else:
    tablepath.resolve().parent.mkdir(parents=True, exist_ok=True)
np.save(tablepath, table)

def coverage_normal(yt, mean, var, *, CL):
    dist = stats.norm.ppf((1 + CL) / 2)
    interval = mean + np.array([-1, 1])[:, None] * dist * np.sqrt(var)
    return np.mean((interval[0] <= yt) & (yt <= interval[1]))

def coverage_quantiles(yt, y, *, CL):
    quantiles = [(1 - CL) / 2, (1 + CL) / 2]
    interval = np.quantile(y, quantiles, axis=0)
    return np.mean((interval[0] <= yt) & (yt <= interval[1]))

def add_error_to_samples(yhat, sigma, seed):
    rng = np.random.default_rng(seed)
    sigma = np.broadcast_to(sigma, yhat.shape[0])
    noise = rng.standard_normal(yhat.shape)
    return yhat + sigma[:, None] * noise

def OLS_error_sdev(X, y):
    x = X.to_pandas()
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    result = model.fit()
    return np.sqrt(result.mse_resid)

def log_loss_from_samples(y, yhat, sigma):
    """
    Estimate the log loss using samples.

    Parameters
    ----------
    y : (n,) array
        Values the distribution is evaluated at.
    yhat : (nsamples, n) array
        Regression function samples.
    sigma : (nsamples,) array
        Error standard deviation samples.

    Returns
    -------
    log_loss : float
        The log loss = -1/len(y) log p(yhat=y), estimate with an average over
        the samples
    exp_log_loss : float
        An estimate of the expected log loss. This is a biased estimate
        involving some approximations.
    """
    assert y.ndim == sigma.ndim == 1
    assert yhat.shape == (sigma.size, y.size)
    error = y - yhat
    sigma2 = np.square(sigma)
    norm = -1/2 * (y.size * np.log(2 * np.pi * sigma2))
    log_p_error_given_sigma = norm - 1/2 * np.sum(np.square(error), axis=1) / sigma2
    exp_log_p_error_given_sigma = norm - 1/2 * y.size
    log_p_error = special.logsumexp(log_p_error_given_sigma) - np.log(sigma.size)
    exp_log_p_error = np.mean(exp_log_p_error_given_sigma)
        # see bartreg.bart._log_posterior_ru for the proof
    return -log_p_error / y.size, -exp_log_p_error / y.size

class Timer:
    """ Context manager that cumulatively measures time each time it is used.
    Does not count time when there is an exception. """

    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, tb):
        if not exc_type:
            self.total += time.perf_counter() - self.start

# mark new run in log file
with open(logfilename, 'a') as logfile:
    logfile.write(f'\n\n\n\n################### NEW RUN ####################\n')
    logfile.write(f'{datetime.datetime.now()}\n')

# load table as memmap to write results progressively
table = np.load(tablepath, 'r+')
try: # try block to close memmap in any case

    indices, = np.nonzero(~table['done'] & todo.to_numpy())
    
    for idx in tqdm.tqdm(indices):

        with (open(logfilename, 'a', buffering=1) as logfile,
                # buffering=1 is line buffering
              contextlib.redirect_stdout(logfile)):

            entry = table[idx]
            seed = np.random.SeedSequence(entry['seed'])
            
            # read dataset
            dataset = nips.Nips(entry['source'])
            x = dataset.xtrain(split=entry['split'])
            y = dataset.ytrain(split=entry['split'])
            xt = dataset.xtest(split=entry['split'])
            yt = dataset.ytest(split=entry['split'])

            sigest = OLS_error_sdev(x, y)

            timer = Timer()
            results = dict(
                n_train=y.size,
                n_test=yt.size,
                p=len(x.columns),
            )

            def postprocess_bartreg_bart(bart, seed):
                logscore, negentropy, yhat_test = bart.log_posterior(yt, seed)
                if bart.sigma.shape:
                    sigma = bart.sigma
                else:
                    sigma = np.broadcast_to(bart.sigma, (ndpost,))
                logloss, exp_logloss = log_loss_from_samples(yt, yhat_test, sigma)
                results.update(
                    alpha=bart.base,
                    beta=bart.power,
                    q=bart.sigquant,
                    nu=bart.sigdf,
                    k=bart.k,
                    sigma=np.mean(bart.sigma),
                    coverage_50_normal=coverage_normal(yt, bart.yhat_test_marg_mean, bart.y_test_marg_var, CL=0.5),
                    coverage_90_normal=coverage_normal(yt, bart.yhat_test_marg_mean, bart.y_test_marg_var, CL=0.9),
                    coverage_50_quantiles=bart.coverage(yt, CL=0.5),
                    coverage_90_quantiles=bart.coverage(yt, CL=0.9),
                    rmse=np.sqrt(np.mean(np.square(yt - bart.yhat_test_marg_mean))),
                    logloss=logloss,
                    exp_logloss=exp_logloss,
                    logloss_gp=-logscore / yt.size,
                    exp_logloss_gp=-negentropy / yt.size,
                )

            def postprocess_bartreg_barteb2(bart, seed):
                logscore, negentropy = bart.log_posterior(yt)
                yhat_test = bart.posterior_samples(seed, ndpost)
                sigma = np.broadcast_to(bart.sigma, (ndpost,))
                logloss, exp_logloss = log_loss_from_samples(yt, yhat_test, sigma)
                results.update(
                    alpha=bart.base,
                    beta=bart.power,
                    nu=bart_hypers['sigdf'],
                    q=bart_hypers['sigquant'],
                    k=bart.k,
                    sigma=bart.sigma,
                    coverage_50_normal=coverage_normal(yt, bart.yhat_test_mean, bart.yhat_test_var + bart.sigma ** 2, CL=0.5),
                    coverage_90_normal=coverage_normal(yt, bart.yhat_test_mean, bart.yhat_test_var + bart.sigma ** 2, CL=0.9),
                    rmse=np.sqrt(np.mean(np.square(yt - bart.yhat_test_mean))),
                    logloss=logloss,
                    exp_logloss=exp_logloss,
                    logloss_gp=-logscore / yt.size,
                    exp_logloss_gp=-negentropy / yt.size,
                )
                results.update(
                    coverage_50_quantiles=results['coverage_50_normal'],
                    coverage_90_quantiles=results['coverage_90_normal'],
                )

            def postprocess_r_bart(kw, bart, seed):
                y_test_marg_var = np.var(bart.yhat_test, axis=0) + np.mean(bart.sigma ** 2)
                y_test = add_error_to_samples(bart.yhat_test, bart.sigma, seed)
                logloss, exp_logloss = log_loss_from_samples(yt, bart.yhat_test, bart.sigma)
                results.update(
                    alpha=kw['base'],
                    beta=kw['power'],
                    q=kw['sigquant'],
                    nu=kw['sigdf'],
                    k=kw['k'],
                    sigma=np.mean(bart.sigma),
                    m=kw['ntree'],
                    coverage_50_normal=coverage_normal(yt, bart.yhat_test_mean, y_test_marg_var, CL=0.5),
                    coverage_90_normal=coverage_normal(yt, bart.yhat_test_mean, y_test_marg_var, CL=0.9),
                    coverage_50_quantiles=coverage_quantiles(yt, y_test, CL=0.5),
                    coverage_90_quantiles=coverage_quantiles(yt, y_test, CL=0.9),
                    rmse=np.sqrt(np.mean(np.square(yt - bart.yhat_test_mean))),
                    logloss=logloss,
                    exp_logloss=exp_logloss,
                )

            method = entry['method']

            print(f'\n@@@@@@@@@@ DATASET {entry["dataset"]} METHOD {method} @@@@@@@@@@')

            if match := re.fullmatch(r'(gp|st)-(ru|la)-(\d)-(\d+)(-(.+))?', method):
                shape, variant, depth, repeat, _, nu = match.groups()
                depth, repeat = int(depth), int(repeat)
                seed, seed1 = seed.spawn(2)
                kw = dict(
                    seed=seed if variant == 'ru' else None,
                    kernelkw=dict(
                        maxd=depth * repeat,
                        reset=None if repeat == 1 else list(range(depth, depth * repeat, depth)),
                    ),
                    nu=float(nu) if shape == 'st' else None,
                    ndpost=ndpost,
                    sigest=sigest,
                    **bart_hypers,
                )
                with timer:
                    bart = bartreg.bart(x, y, x_test=xt, **kw)
                postprocess_bartreg_bart(bart, seed1)

            elif match := re.fullmatch(r'gp-eb-(\d)-(\d+)(-(.+))?', method):
                depth, repeat, _, post = match.groups()
                depth, repeat = int(depth), int(repeat)
                assert post in (None, 'ru', 'mcmc')
                kw = dict(
                    x_test=xt,
                    kernelkw=dict(
                        maxd=depth * repeat,
                        reset=None if repeat == 1 else list(range(depth, depth * repeat, depth)),
                    ),
                    fitkw=dict(
                        mlkw=dict(epsrel='auto' if repeat == 1 else 0),
                    ),
                    sigest=sigest,
                    hyperprior=gphyp_hyperprior,
                    **bart_hypers,
                )
                with timer:
                    bart = bartreg.barteb2(x, y, **kw)
                    if post == 'ru':
                        seed, seed1 = seed.spawn(2)
                        del kw['fitkw'], kw['hyperprior']
                        kw.update(
                            seed=seed,
                            ndpost=ndpost,
                            base=bart.base,
                            power=bart.power,
                            k=bart.k,
                        )
                        bart = bartreg.bart(x, y, **kw)
                    elif post == 'mcmc':
                        seed, seed1 = seed.spawn(2)
                        del kw['fitkw'], kw['hyperprior'], kw['kernelkw']
                        seed = seed.generate_state(1).item() >> 1
                        kw.update(
                            seed=seed,
                            base=bart.base,
                            power=bart.power,
                            k=bart.k,
                            numcut=y.size,
                            **kw_dbarts,
                        )
                        bart = dbarts.bart(x, y, **kw)

                if post == 'ru':
                    postprocess_bartreg_bart(bart, seed1)
                elif post == 'mcmc':
                    postprocess_r_bart(kw, bart, seed1)
                else:
                    postprocess_bartreg_barteb2(bart, seed)

            elif method == 'mcmc':
                seed, seed1 = seed.spawn(2)
                seed = seed.generate_state(1).item() >> 1
                kw = dict(
                    x_test=xt,
                    numcut=y.size,
                        # equiv. to <large number>, but BART allocates memory
                        # for numcut unconditionally, so keep it as small as
                        # possible
                    seed=seed,
                    offset=(y.max() + y.min()) / 2,
                    sigest=sigest,
                    **bart_hypers,
                    **kw_BART,
                )
                with timer:
                    bart = BART.mc_gbart(x, y, **kw)
                postprocess_r_bart(kw, bart, seed1)
            
            elif method == 'mcmc-cv':
                
                # cross validation exactly as in Chipman et al. 2010
                params = (
                    pl.DataFrame(dict(nu=[3., 3., 10.], q=[0.90, 0.99, 0.75]))
                    .join(pl.DataFrame(dict(m=[50, 200])), how='cross')
                    .join(pl.DataFrame(dict(k=[1., 2., 3., 5.])), how='cross')
                )
                def makekw(row):
                    kw = dict(**bart_hypers, **kw_dbarts)
                    kw.update(k=row['k'], sigquant=row['q'], sigdf=row['nu'], ntree=row['m'])
                    return kw
                rmse = np.zeros(len(params))
                    
                for i, row in enumerate(params.iter_rows(named=True)):
                    kw = makekw(row)
                    
                    sse = 0
                    n = 0
                    
                    for fold in range(1, dataset.nfolds + 1):
                        
                        xcv = dataset.xtrain(fold=-fold, split=entry['split'])
                        ycv = dataset.ytrain(fold=-fold, split=entry['split'])
                        xtcv = dataset.xtrain(fold=fold, split=entry['split'])
                        ytcv = dataset.ytrain(fold=fold, split=entry['split'])
                        
                        seed, seed1 = seed.spawn(2)
                        seed1 = seed1.generate_state(1).item() >> 1
                        with timer:
                            bart = dbarts.bart(xcv, ycv,
                                x_test=xtcv,
                                seed=seed1,
                                sigest=OLS_error_sdev(xcv, ycv),
                                numcut=ycv.size,
                                    # equiv. to <large number>, but BART allocates
                                    # memory for numcut unconditionally, so keep it
                                    # as small as possible
                                **kw,
                            )
                        sse += np.sum(np.square(ytcv - bart.yhat_test_mean))
                        n += len(ytcv)

                        # free up memory before running another mcmc
                        del bart
                        gc.collect()

                        # sleep, it might help with avoiding the thing getting stuck
                        time.sleep(0.01)
                    
                    rmse[i] = np.sqrt(sse / n)
                
                # rerun fit on whole training set with best hyperparameters
                ibest = np.argmin(rmse)
                pbest = params.row(ibest, named=True)
                seed, seed1 = seed.spawn(2)
                seed = seed.generate_state(1).item() >> 1
                kw = makekw(pbest)
                with timer:
                    bart = dbarts.bart(x, y,
                        x_test=xt,
                        seed=seed,
                        sigest=sigest,
                        numcut=y.size,
                        **kw,
                    )
                postprocess_r_bart(kw, bart, seed1)
                
            elif method == 'mcmc-xcv':

                seed, seed1, seed2 = seed.spawn(3)
                seed = seed.generate_state(1).item() >> 1
                seed1 = seed1.generate_state(1).item() >> 1
                
                params = (
                    pl.DataFrame(dict(ntree=[50, 200]))
                    .join(pl.DataFrame(dict(power=[1., 2., 4.])), how='cross')
                    .join(pl.DataFrame(dict(base=[0.6, 0.8, 0.95])), how='cross')
                )

                def plist(name):
                    return list(params.get_column(name).unique(maintain_order=True))

                with timer:
                    rmse_cv = dbarts.xbart(
                        x, y,
                        n_reps=1,
                        n_threads=kw_dbarts['nthread'],
                        n_trees=plist('ntree'),
                        power=plist('power'),
                        base=plist('base'),
                        drop=False,
                        seed=seed,
                    )
                    # shape of rmse-cv: n_reps, n_trees, k, power, base
                    rmse_cv = rmse_cv.squeeze((0, 2))
                    ibest = np.argmin(rmse_cv)
                    best_params = params.row(ibest, named=True)
                
                    # rerun fit on whole training set with best hyperparameters
                    kw = dict(**kw_dbarts, **bart_hypers)
                    kw.update(
                        x_test=xt,
                        seed=seed1,
                        sigest=sigest,
                        usequants=False,
                        sigquant=0.9,
                        sigdf=3.,
                        k=2.,
                        **best_params,
                    )
                    bart = dbarts.bart(x, y, **kw)
                
                postprocess_r_bart(kw, bart, seed2)

            else:
                print('######## UNIMPLEMENTED METHOD', method, '########')
                continue

        del bart
        jax.clear_caches()
        gc.collect()

        results.update(
            time=timer.total,
        )
        
        for n, v in results.items():
            entry[n] = v  
        entry['done'] = True
        table.flush()

finally:
    del table # to close writable memmap
