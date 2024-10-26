import warnings
import pathlib

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects
import polars as pl

""" makes figure 4 from the output of comp42.py """

methods = {
    # method id: (label, style)
    'gp-ru-2-5': ('GP', dict(color='red', linestyle='-')),
    'gp-eb-2-5-ru': ('GP-hyp', dict(color='red', linestyle='--')),
    'gp-eb-2-5-mcmc': ('GP+MCMC', dict(color='red', linestyle=':')),
    'mcmc': ('MCMC', dict(color='black', linestyle='-')),
    'mcmc-xcv': ('MCMC-XCV', dict(color='black', linestyle='--')),
    'mcmc-cv': ('MCMC-CV', dict(color='black', linestyle=':')),
}

############################

tablefile = 'comp42/comp42.npy'

table = pl.DataFrame(np.load(tablefile))
table = (table
    .sort(['dataset', 'split', 'method'])
    .filter(pl.col('done'))
    .filter(pl.col('method').is_in(list(methods)))
    .with_columns(
        rrmse=(pl.col('rmse') / pl.col('rmse').min())
            .over(['dataset', 'split']),
        relloss=(pl.col('logloss') - pl.col('logloss').min())
            .over(['dataset', 'split']),
    )
)

print('min p =', table['p'].min(), 'max p =', table['p'].max())
print('min n =', table['n_train'].min(), 'max n =', table['n_train'].max())

fig, axs = plt.subplots(1, 3, num='articleplot', clear=True, layout='constrained', figsize=[8.61, 2.89])
axs = axs.reshape(-1)
ax_rmse, ax_score, ax_time = axs

############ RMSE #############
rrmse = (table
    .pivot(index=['dataset', 'split'], on='method', values='rrmse')
    .drop(['dataset', 'split'])
    .drop_nulls()
)
ax = ax_rmse
for i, (method, (label, style)) in enumerate(methods.items()):
    x = rrmse[method].sort().to_numpy()
    y = np.arange(len(x)) / len(x)
    ax.plot(x, y, drawstyle='steps-post', label=label, **style)
ax.set(
    title=f'RMSE w.r.t. best method',
    ylabel='ECDF over datasets',
    xlabel='RMSE/RMSE_min',
    xlim=(0.99, 1.51),
    ylim=(0, 1),
)
ax.legend(
    # handles=ax.get_lines()[::-1],
    loc='lower right',
)

############ LOG-LOSS #############
score = (table
    .pivot(index=['dataset', 'split'], on='method', values='relloss')
    .drop(['dataset', 'split'])
    .drop_nulls()
)
ax = ax_score
lines = {}
for i, (method, (label, style)) in enumerate(methods.items()):
    x = score[method].sort().to_numpy()
    y = np.arange(len(x)) / len(x)
    line, = ax.plot(x, y, drawstyle='steps-post', label=label, **style)
    lines[method] = line
ax.set(
    title='Log loss w.r.t. best method',
    ylabel='ECDF over datasets',
    xlabel='log_loss $-$ log_loss_min',
    xlim=(-0.02, 1.27),
    ylim=(0, 1),
)

center = np.array([0.6, 0.4])
nw = np.array([-1, 1])
se = np.array([1, -1])

for vec, label in (nw, 'Better'), (se, 'Worse'):
    pos = center + 0.2 * vec
    ax.annotate('', xy=pos, xytext=center + 0.02 * vec,
                arrowprops=dict(arrowstyle="->", color="black"),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.annotate(label, xy=pos, xytext=(-vec[0], 0),
                ha='left' if vec[0] < 0 else 'right', va='center',
                xycoords='axes fraction', textcoords='offset fontsize',
                path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])

# print maxima
maxima = (rrmse
    .max()
    .transpose(include_header=True, header_name='method', column_names=('rrmse',))
    .join(score
        .max()
        .transpose(include_header=True, header_name='method', column_names=('score',)),
        on='method',
    )
)
with pl.Config(float_precision=1):
    print(maxima)

############ TIME #############
time = (table
    .pivot(index=['dataset', 'n_train'], on='method', values='time', aggregate_function='min')
    .sort('n_train')
    .drop('dataset')
)
ax = ax_time
ax.set(
    title='Running time',
    ylabel='Time [s]',
    xlabel='Number of training points',
    xscale='log',
    yscale='log',
)
for i, (method, (label, style)) in enumerate(methods.items()):
    ax.plot(time['n_train'], time[method], label=label, **style)

for ax in axs:
    ax.grid(which='major', linestyle='--')
    ax.grid(which='minor', linestyle=':')

fig.savefig((pathlib.Path(tablefile).parent / fig.get_label()).with_suffix('.pdf'))
fig.show()
