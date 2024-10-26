import argparse
import pathlib
import re

from matplotlib import pyplot as plt
import numpy as np
import polars as pl
import gvar
import lsqfitgp as lgp

""" makes figures 5, 14, 15, using the output of driver.py """

# config
root = pathlib.Path('acic2022') # absolute path because this script could be a copy in a workdir
name = 'GPBCF*'
show_xerr = False
bart_subs = [
    'BCFPS',
    'BT',
    'BTPS',
    'BTPSY',
    'BTYPS',
    'BTYPSY',
    'FisherBART*',
    'flexBART_1',
    'flexBART_2',
    'flexBART_3',
    'flexBART_4*',
    'flexBART_5*',
    'flexBART_6*',
    's4b*',
    'xvalBART*',
    'xvalBARTps*',
]

# parse command line
parser = argparse.ArgumentParser(prog='analysis', description='analysis of inferences on simulated datasets')
parser.add_argument('workdir', type=pathlib.Path, help='path of job directory', nargs='?')
args = parser.parse_args()

# get working directory
if args.workdir:
    workdir = args.workdir
else:
    workdirs = sorted([
        p for p in pathlib.Path(root / 'loop' / 'workdirs').iterdir()
        if p.is_dir() and re.fullmatch(r'workdir_\d{14}', p.name)
    ])
    workdir = workdirs[-1]

# load results table
table = np.load(workdir / 'results.npy')
table = pl.DataFrame(table).filter(pl.col('done'))

# print running time
print(f'Time per fit: {table['time_total'].median():.2g}s')

# filter results, rearrange to match official format, compute basic metrics
table = (table
    .unpivot(
        index='dataset',
        on=pl.selectors.matches(r'(Overall|.+?=.+?)_.+'),
        variable_name='stratum',
        value_name='value',
    )
    .with_columns(pl
        .col('stratum')
        .str.split_exact('_', 1)
        .struct.rename_fields(['stratum', 'statistic'])
    )
    .unnest('stratum')
    .pivot(values='value', index=['dataset', 'stratum'], on='statistic')
    .sort('dataset', 'stratum')
    .with_columns(pl
        .col('satt', 'true')
        # .filter(pl.col('stratum') == 'Overall')
        .first()
        .over('dataset')
        .name.suffix('_Overall')
    )
    .with_columns(
        satt_rel=pl.col('satt') - pl.col('satt_Overall'),
        true_rel=pl.col('true') - pl.col('true_Overall'),
        covers=(pl.col('lower90') <= pl.col('true')) & (pl.col('true') <= pl.col('upper90')),
    )
    .with_columns(pl
        .col('stratum')
        .str.replace('Yearly', 'Year'),
    )
)

def error(suffix=''):
    return pl.col('satt' + suffix) - pl.col('true' + suffix)

def rmse(error):
    return (error ** 2).mean().sqrt()

def rmse_se(error):
    return ((error ** 2).std() / pl.len().sqrt()) / rmse(error)

# compute grouped performance metrics
perf = (table
    .group_by('stratum')
    .agg([
        error().mean().abs().alias('bias'),
        (error().std() / pl.len().sqrt()).alias('bias_se'),
        rmse(error()).alias('rmse'),
        rmse_se(error()).alias('rmse_se'),
        rmse(error('_rel')).alias('drmse'),
        rmse_se(error('_rel')).alias('drmse_se'),
        pl.col('covers').mean().alias('coverage'),
        (pl.col('covers').std() / pl.len().sqrt()).alias('coverage_se'),
        pl.len().alias('n'),
    ])
    .sort('stratum', descending=True)
)

# load competition results
results = (pl
    .read_csv(root / 'results' / 'ACIC_results.csv', null_values='NA')
    .filter(
        ~pl.col('variable').is_in(['Practices', 'Subgroups']),
        pl.col('Confounding Strength') == 'All',
    )
    .select(
        'Submission',
        'rmse',
        stratum=
            pl.when(pl.col('variable') == 'Yearly')
                .then('Year=' + pl.col('year').cast(pl.Utf8))
            .when(pl.col('variable') == 'Overall')
                .then(pl.col('variable'))
            .otherwise(pl.col('variable') + '=' + pl.col('level')),
        bias=pl.col('bias').abs(),
        coverage='cover',
    )
    .sort('stratum', descending=True)
)

# summarize competition results
coverage_badness = (pl.col('coverage') - 0.90).abs()
summary = (results
    .group_by('stratum', maintain_order=True)
    .agg([
        pl.col('rmse').min().alias('best_rmse'),
        pl.col('bias').min().alias('best_bias'),
        pl.col('coverage').sort_by(coverage_badness).drop_nulls().first().alias('best_coverage'),
        pl.col('rmse').median().alias('median_rmse'),
        pl.col('bias').median().alias('median_bias'),
        pl.col('coverage').sort_by(coverage_badness).drop_nulls()
            .implode().list.get(pl.len() // 2).first()
                # workaround in place of .get(...), see polars #19363
            .alias('median_coverage'),
    ])
)

# make list with 'Overall' results only
classif = (pl
    .concat([ # add my method to the list
        results,
        (perf
            .with_columns(Submission=pl.lit(name))
            .select(results.columns)
        ),
    ])
    .filter(pl.col('stratum') == 'Overall')
    .drop('stratum')
    .sort('Submission')
    .with_columns(
        mine=pl.col('Submission') == name,
        bart=pl.col('Submission').is_in(bart_subs),
    )
)

# extract results
overall = perf.row(by_predicate=pl.col('stratum') == 'Overall', named=True)
rmse = gvar.gvar(overall['rmse'], overall['rmse_se'])
coverage = gvar.gvar(overall['coverage'], overall['coverage_se'])
def getquant(by):
    pos = (classif
        .sort(by)
        .with_row_index('index')
        .row(by_predicate=pl.col('Submission') == name, named=True)['index']
    )
    return pos / (len(classif) - 1)
rmse_q = getquant('rmse')
coverage_q = getquant(coverage_badness)
best_drmse_row = perf.filter(pl.col('stratum') != 'Overall').sort('drmse').row(0, named=True)
worst_drmse_row = perf.sort('drmse', descending=True).row(0, named=True)
best_drmse = gvar.gvar(best_drmse_row['drmse'], best_drmse_row['drmse_se'])
worst_drmse = gvar.gvar(worst_drmse_row['drmse'], worst_drmse_row['drmse_se'])

# print results
with lgp.gvar_format():
    print(f'rmse = {rmse} ({rmse_q:.0%})')
    print(f'coverage = {coverage} ({coverage_q:.0%})')
    print(f'best drmse = {best_drmse} (2.95-6.16)')
    print(f'worst drmse = {worst_drmse} (15.8-26.0)')

# reset matplotlib
plt.close('all')
plt.rcdefaults()

# make directory for plots
plotdir = workdir / 'figures'
plotdir.mkdir(exist_ok=True)
def save_fig(fig):
    path = plotdir / f'{fig.get_label()}.pdf'
    print(f'write {path}...')
    fig.savefig(path)

# figure for detailed results per category
fig, axs = plt.subplots(1, 4, num='analysis', clear=True,
    layout='constrained', sharey=True, figsize=[8.3 , 2.94])

# assign plots and sync axes
ax_bias, ax_rmse, ax_drmse, ax_cov = axs
ax_drmse.sharex(ax_rmse)
ax_bias.sharex(ax_rmse)

# bias
def my_barh(ax, y, x, xerr, **kw):
    return ax.barh(y, x, xerr=xerr, facecolor='#f55', edgecolor='none', **kw)
my_barh(ax_bias, perf['stratum'], perf['bias'], perf['bias_se'], label='My method')
ax_bias.set_ylim(ax_bias.get_ylim()) # lock ylim
edges = np.pad(np.arange(0.5, len(summary) - 1), (1, 1), constant_values=(-100, 100))
ax_bias.stairs(summary['best_bias'], edges, orientation='horizontal', color='black', label='Best')
ax_bias.stairs(summary['median_bias'], edges, orientation='horizontal', color='black', linestyle='--', label='Median')
ax_bias.set(title='Bias (magnitude of)')
ax_bias.legend(loc='upper right', fontsize='small')

# rmse
my_barh(ax_rmse, perf['stratum'], perf['rmse'], perf['rmse_se'])
ax_rmse.stairs(summary['best_rmse'], edges, orientation='horizontal', color='black')
ax_rmse.stairs(summary['median_rmse'], edges, orientation='horizontal', color='black', linestyle='--')
ax_rmse.set(title='RMSE')

# coverage
my_barh(ax_cov, perf['stratum'], perf['coverage'], perf['coverage_se'])
ax_cov.stairs(summary['best_coverage'], edges, orientation='horizontal', color='black')
ax_cov.stairs(summary['median_coverage'], edges, orientation='horizontal', color='black', linestyle='--')
ax_cov.axvline(0.9, color='black', linestyle='-.')
ax_cov.set(xlim=(0.5, 1.0), title='Coverage')

# drmse
my_barh(ax_drmse, perf['stratum'], perf['drmse'], perf['drmse_se'])
ax_drmse.set(title='DRMSE')

# save figure
save_fig(fig)


# figure with list
fig, axs = plt.subplots(1, 2,
    num='analysis_list',
    clear=True,
    layout='constrained',
    figsize=[7.18, 8.75],
)

# plot rmse
ax = axs[0]
classif = classif.sort('rmse', descending=True)
mine = classif.row(by_predicate=pl.col('mine'), named=True)

kw = dict(facecolor='white', edgecolor='black')
bartkw = dict(facecolor='white', edgecolor='black', hatch='xxxx')
minekw = dict(facecolor='red', edgecolor='black')

ax.barh(classif['Submission'], classif['rmse'], **kw)
ax.barh(classif.filter('bart')['Submission'], classif.filter('bart')['rmse'], **bartkw, label='BART methods')
ax.barh(mine['Submission'], mine['rmse'], **minekw, label='My method')

ylim = (-0.6, len(classif) - 1 + 0.6)
ax.set(title='RMSE', xlim=(0, 100), ylim=ylim)

ax.legend()

# plot coverage
ax = axs[1]
classif = classif.sort(coverage_badness, descending=True)

ax.barh(classif['Submission'], classif['coverage'], **kw)
ax.barh(classif.filter('bart')['Submission'], classif.filter('bart')['coverage'], **bartkw)
ax.barh(mine['Submission'], mine['coverage'], **minekw)

ax.axvline(0.9, color='black', linestyle='-.', zorder=-1)
ax.annotate(
    'Target coverage',
    rotation='vertical',
    xy=(0.9, 0.2),
    xycoords='axes fraction',
    xytext=(-0.5, 0),
    textcoords='offset fontsize',
    ha='right',
    va='center',
)

ax.set(title='Coverage', ylim=ylim, xlim=(0, 1))

# save figure
save_fig(fig)


# figure with list, compact version
fig, axs = plt.subplots(1, 2,
    num='analysis_compact',
    clear=True,
    layout='constrained',
    sharey=True,
    figsize=[6.83, 2.65],
)

def plot_ecdf(ax, sample, **kw):
    sample = np.asarray(sample)
    values = np.linspace(0, 1, len(sample) + 1)
    edges = np.pad(np.sort(sample), 1, constant_values=(-1e10, 1e10))
    ax.stairs(values, edges, color='black', **kw)

axs[1].axvline(0.9, color='black', linestyle='-.')
axs[1].annotate(
    'Target coverage',
    rotation='vertical',
    xy=(0.9, 0.4),
    xytext=(-0.5, 0),
    textcoords='offset fontsize',
    ha='right',
    va='center',
)

for i, which in enumerate(['RMSE', 'Coverage']):
    ax = axs[i]
    col = which.lower()
    plot_ecdf(ax, classif[col])
    ax.set(xlabel=which)

    heights = np.linspace(0, 1, len(classif) + 1)
    heights = (heights[:-1] + heights[1:]) / 2
    classif = classif.sort(col).with_columns(height=pl.lit(heights))

    bart = classif.filter('bart')
    ax.plot(bart[col], bart['height'], 'o', markerfacecolor='yellow', markeredgecolor='black', label='BART-based methods')

    mine = classif.filter('mine')
    ax.plot(mine[col], mine['height'], 's', markerfacecolor='red', markeredgecolor='black', markersize=7, color='red', label='GPBCF (my method)')
    
axs[0].legend(loc='lower right')
axs[0].set(xlim=(0, 88), ylabel='ECDF over competitors')

axs[1].set(xlim=(0, 1), ylim=(0, 1))

save_fig(fig)

plt.show(block=False)
