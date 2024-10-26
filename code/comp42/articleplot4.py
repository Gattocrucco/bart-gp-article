import pathlib

import numpy as np
from matplotlib import pyplot as plt
import polars as pl

""" makes figure 9 from the output of comp42.py """

tablefile = 'comp42/comp42.npy'

table = pl.DataFrame(np.load(tablefile))

gp_method = 'gp-eb-2-5-ru'
mcmc_method = 'mcmc-cv'
max_beta = 100

table = (table
    .filter(pl.col('done'))
    .group_by(['dataset', 'split'])
    .agg(
        pl.col('rmse').filter(pl.col('method') == gp_method).first().name.suffix('_gp'),
        pl.col('rmse').filter(pl.col('method') == mcmc_method).first().name.suffix('_mcmc'),
        pl.col('logloss').filter(pl.col('method') == gp_method).first().name.suffix('_gp'),
        pl.col('logloss').filter(pl.col('method') == mcmc_method).first().name.suffix('_mcmc'),
        pl.col('alpha').filter(pl.col('method') == gp_method).first(),
        pl.col('beta').filter(pl.col('method') == gp_method).first(),
        pl.col('n_train').first().alias('n'),
        pl.col('p').first(),
    )
    .with_columns(
        rmseratio=pl.col('rmse_gp') / pl.col('rmse_mcmc'),
        loglossdiff=pl.col('logloss_gp') - pl.col('logloss_mcmc'),
        betaclip=pl.min_horizontal('beta', max_beta),
    )
)

fig, axs = plt.subplots(1, 4,
    num='articleplot4',
    clear=True,
    layout='constrained',
    figsize=[8.81, 2.79],
    sharey=True,
)

linestyle = dict(color='red', linestyle='--')
axs[0].axvline(0.95, **linestyle)
axs[1].axvline(2, **linestyle)

things = [
    ('alpha', 'GP-hyp $\\alpha$'),
    ('betaclip', f'min(GP-hyp $\\beta$, {max_beta})'),
    ('n', '$n$'),
    ('p', '$p$'),
]

for (col, xlabel), ax in zip(things, axs):
    ax.plot(table[col], table['rmseratio'], '.k', alpha=0.1)
    ax.set(xlabel=xlabel)

axs[0].set(
    ylabel='RMSE GP-hyp / RMSE MCMC-CV',
    yscale='log',
    xlim=(0, 1),
)
axs[1].set(
    xscale='log',
)
axs[2].set(
    xscale='log',
)
axs[3].set(
    xscale='log',
)

linestyle = dict(color='lightgray', linestyle='--', zorder=-1)
def lock_lims(ax):
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
lock_lims(axs[0])
lock_lims(axs[1])
axs[0].plot(axs[0].get_xlim(), axs[0].get_ylim()[::-1], **linestyle)
axs[1].plot(axs[1].get_xlim(), axs[1].get_ylim(), **linestyle)

linestyle = dict(color='lightgray', linestyle=':', zorder=-1)
for ax in axs:
    ax.axhline(1, **linestyle)

fig.savefig((pathlib.Path(tablefile).parent / fig.get_label()).with_suffix('.pdf'))
fig.show()
