import warnings
import pathlib

import numpy as np
from matplotlib import pyplot as plt
import polars as pl

""" makes figure 8 from the output of comp42.py """

methods = {
    # method id: (label, style)
    'mcmc': ('MCMC', dict(color='black', linestyle='-')),
    'gp-ru-2-5': ('GP', dict(color='red', linestyle='-')),
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

fig, axs = plt.subplots(1, 2, num='articleplot2', clear=True, layout='constrained', figsize=[5.74, 2.79])
axs = axs.reshape(-1)
ax_rmse, ax_score = axs

rrmse = (table
    .pivot(index=['dataset', 'split'], on='method', values='rrmse')
    .drop(['dataset', 'split'])
    .drop_nulls()
)
ax = ax_rmse
for i, (method, (label, style)) in enumerate(methods.items()):
    x = rrmse[method].sort().to_numpy()
    y = np.arange(len(x)) / len(x)
    ax.plot(x, y, drawstyle='steps-post', label=label, **style, zorder=10 - i)
ax.set(
    title=f'RMSE w.r.t. best method',
    ylabel='ECDF over datasets',
    xlabel='RMSE/RMSE_min',
    xlim=(0.99, 1.41),
    ylim=(0, 1),
)
with pl.Config(float_precision=1):
    print('Maximum RRMSE:')
    print(rrmse.max())

score = (table
    .pivot(index=['dataset', 'split'], on='method', values='relloss')
    .drop(['dataset', 'split'])
    .drop_nulls()
)
ax = ax_score
for i, (method, (label, style)) in enumerate(methods.items()):
    x = score[method].sort().to_numpy()
    y = np.arange(len(x)) / len(x)
    ax.plot(x, y, drawstyle='steps-post', label=label, **style, zorder=10 - i)
ax.set(
    title='Log loss w.r.t. best method',
    ylabel='ECDF over datasets',
    xlabel='log_loss $-$ log_loss_min',
    xlim=(-0.02, 1.02),
    ylim=(0, 1),
)
with pl.Config(float_precision=2):
    print('Maximum relative log loss:')
    print(score.max())

ax_rmse.legend(loc='lower right')

for ax in axs:
    ax.grid(which='major', linestyle='--')
    ax.grid(which='minor', linestyle=':')

fig.savefig((pathlib.Path(tablefile).parent / fig.get_label()).with_suffix('.pdf'))
fig.show()
