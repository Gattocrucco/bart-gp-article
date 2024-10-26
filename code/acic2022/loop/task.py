import sys
import time
import argparse
import pathlib

import numpy as np
import polars as pl
import lsqfitgp as lgp

sys.path.insert(0, '.')
from rbartpackages import dbarts

# fixed config
root = pathlib.Path('acic2022')
nsamples = 50 # samples from laplace of hyper posterior for gp bcf
nsamples_per_hp = 20 # samples from gp posterior for each hyper
artificial_effect_shift = 0 # shift the treated outcome by this amount

# parse command line
parser = argparse.ArgumentParser(prog='task', description='single inference')
parser.add_argument('-t', required=True, help='path of results table')
parser.add_argument('-i', type=int, required=True, help='index of entry in table to use')
parser.add_argument('-s', type=int, required=True, help='integer used as random seed')
args = parser.parse_args()

class Timer:

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.duration = time.perf_counter() - self.start

# load table as memmap to write results directly
entry = np.load(args.t, mmap_mode='r+')[args.i]

# set up random number generator
dataset = entry['dataset']
entry['seed'] = [dataset, args.s]
rng = np.random.default_rng(entry['seed'])

# load data
print('load data...')
prefix = root / 'track2_20220404'
df_p = pl.read_csv(prefix / 'practice' / f'acic_practice_{dataset:04d}.csv')
df_py = pl.read_csv(prefix / 'practice_year' / f'acic_practice_year_{dataset:04d}.csv')
df_py_aux = pl.read_ipc(root / 'practice_year_aux' / f'acic_practice_year_{dataset:04d}.arrow', memory_map=False)

# select the auxiliary summaries to use and merge everything
df = (df_py_aux
    .select(
        'id.practice',
        'year',
        pl.col(r'^V\d_(min|max|std)$'),
    )
    .join(df_py, on=('id.practice', 'year'))
    .join(df_p, on='id.practice')
)
assert len(df) == 500 * 4

# shift treated units by a fixed amount for testing purposes
df = df.with_columns(
    Y=pl.col('Y') + pl
    .when((pl.col('Z') == 1) & (pl.col('post') == 1))
    .then(artificial_effect_shift)
    .otherwise(0)
)

# move pretreatment outcomes and per-year covariates to new columns: we are
# going to use standard unconfoundedness given pretreatment outcomes instead
# of the DID-like assumption given by the competition rules
V_columns = [c for c in df.columns if c.startswith('V')]
posttreatment = df.filter(pl.col('post') == 1).drop(*V_columns, 'post')
for (year,), stratum in df.filter(pl.col('post') == 0).group_by('year'):
    posttreatment = posttreatment.join(
        stratum.select(
            'id.practice',
            pl.col(['Y', 'n.patients'] + V_columns).name.suffix(f'_year{year}')
        ), on='id.practice'
    )

# and pre-treatment trend as covariate
posttreatment = posttreatment.with_columns(
    pre_trend=pl.col('Y_year2') - pl.col('Y_year1'))

# extract outcome, treatment, weights and covariates
y = posttreatment['Y'].to_numpy()
z = posttreatment['Z'].to_numpy().astype(bool)
npatients_obs = posttreatment['n.patients'].to_numpy() # for SATT average
Xobs = (posttreatment
    .drop('Y', 'id.practice', 'n.patients')
    .to_dummies(columns=['X2', 'X4'])
)
Xmis = (Xobs
    .filter(pl.col('Z') == 1)
    .with_columns(Z=1 - pl.col('Z'))
)

# fit treatment
print('\nfit treatment (BART)...')
X_fit = Xobs.filter(pl.col('year') == pl.min('year')).drop('year')
X_pred = Xobs.drop('year', 'Z')
with Timer() as timer_ps:
    fit_treatment = dbarts.bart(
        x_train=X_fit.drop('Z'),
        y_train=X_fit.get_column('Z'),
        usequants=True,
        numcut=len(X_fit),
        nchain=4,
        nthread=4,
        nskip=1000,
        ndpost=250,
        keeptrees=True,
        verbose=True,
        keeptrainfits=False,
    )
    ps = fit_treatment.predict(X_pred, type='ev').mean(axis=0)

print('\nfit outcome (GP BCF)...')

auxhp = lgp.copula.makedict({
    'lambda_pi': lgp.copula.halfcauchy(2),
})

def gpaux(hp, gp):
    scale = hp['lambda_pi']
    kernel = scale ** 2 * lgp.Expon(dim='pihat')
    return gp.defproc('aux', kernel)

Xobs_ps = Xobs.with_columns(ps=pl.lit(ps))
Xmis_ps = (Xobs_ps
    .filter(pl.col('Z') == 1)
    .with_columns(Z=1 - pl.col('Z'))
)
with Timer() as timer_hypers:
    fit_outcome = lgp.bayestree.bcf(
        y=y,
        z=Xobs['Z'],
        x_mu=Xobs.drop('Z'),
        pihat=Xobs_ps['ps'],
        gpaux=gpaux,
        otherhp=auxhp,
        transf=['standardize', 'yeojohnson'],
        fitkw=dict(verbosity=4, raises=False),
        kernelkw_mu=dict(gamma=1),
        kernelkw_tau=dict(gamma=1),
    )
print()
print(fit_outcome)

# define groups of units for conditional SATT
strata = (df
    .filter((pl.col('Z') == 1) & (pl.col('post') == 1))
    .select([f'X{i}' for i in range(1, 6)] + ['year'])
    .rename({'year': 'Yearly'})
    .with_row_index('index')
)

def sortdict(d):
    return {k: d[k] for k in sorted(d)}

def compute_satt(ymis):
    """ compute effects on the treated """
    yobs = y[z]
    n = npatients_obs[z]
    effect = yobs - ymis

    satt = {}

    satt['Overall'] = np.average(effect, weights=n, axis=-1)
    for variable in strata.columns:
        if variable == 'index':
            continue
        for (level,), stratum in strata.group_by(variable):
            indices = stratum['index'].to_numpy()
            key = f'{variable}={level}'
            satt[key] = np.average(effect[..., indices], weights=n[indices], axis=-1)

    return sortdict(satt)

print('\ncompute satt...')

# sample the posterior and compute the SATT
satt_samples = {}
with Timer() as timer_satt:
    for _ in range(nsamples):

        # create GP at sampled hypers and get imputed outcomes
        ymis_sample = fit_outcome.pred(
            z=Xmis_ps.get_column('Z'),
            x_mu=Xmis_ps.drop('ps', 'Z'),
            pihat=Xmis_ps.get_column('ps'),
            error=True,
            hp='sample',
            samples=nsamples_per_hp,
            transformed=False,
            rng=rng,
        )

        # compute satt
        satt_sample = compute_satt(ymis_sample)
        for k, v in satt_sample.items():
            satt_samples.setdefault(k, []).append(v)

# concatenate samples taken at different hyperparameters
for k, v in satt_samples.items():
    satt_samples[k] = np.concatenate(v)

# compute posterior summaries
satt_summaries = {
    k: (np.mean(sample, axis=0), np.quantile(sample, [0.05, 0.95], axis=0))
    for k, sample in satt_samples.items()
}

# load actual true effect
df_results = (pl
    .read_csv(root / 'results' / 'ACIC_estimand_truths.csv', null_values='NA')
    .filter(pl.col('dataset.num') == dataset)
    .filter(pl.col('variable').is_not_null())
    .with_columns(
        level=pl
            .when(pl.col('variable') == 'Yearly')
            .then(pl.col('year'))
            .otherwise(pl.col('level')),
        SATT=pl.col('SATT') + artificial_effect_shift,
    )
)

# collect true values into same format of estimates
satt_true = {}
for (variable,), group in df_results.group_by('variable'):
    if len(group) > 1:
        for row in group.iter_rows(named=True):
            level = row['level']
            satt_true[f'{variable}={level}'] = row['SATT']
    else:
        satt_true[variable] = group['SATT'][0]

# save results
entry['time_ps'] = timer_ps.duration
entry['time_hypers'] = timer_hypers.duration
entry['time_satt'] = timer_satt.duration
for k, v in satt_summaries.items():
    entry[f'{k}_satt'] = v[0]
    entry[f'{k}_lower90'] = v[1][0]
    entry[f'{k}_upper90'] = v[1][1]
    entry[f'{k}_true'] = satt_true[k]
