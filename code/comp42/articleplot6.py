import pathlib

import numpy as np
from matplotlib import pyplot as plt
import polars as pl

""" makes figure 11 from the output of comp42.py """

tablefile = 'comp42/comp42.npy'

table = pl.DataFrame(np.load(tablefile))

gp_method = 'gp-eb-2-5-ru'
mcmc_method = 'mcmc-xcv'
max_gp_beta = 30

table = (table
    .filter(pl.col('done'))
    .group_by(['dataset', 'split'])
    .agg(
        pl.col('alpha').filter(pl.col('method') == gp_method).first().name.suffix('_gp'),
        pl.min_horizontal(pl.col('beta'), max_gp_beta).filter(pl.col('method') == gp_method).first().name.suffix('_gp'),
        (pl.col('alpha') * 2 ** -pl.col('beta')).filter(pl.col('method') == gp_method).first().alias('P1_gp'),
        pl.col('alpha').filter(pl.col('method') == mcmc_method).first().name.suffix('_mcmc'),
        pl.col('beta').filter(pl.col('method') == mcmc_method).first().name.suffix('_mcmc'),
        (pl.col('alpha') * 2 ** -pl.col('beta')).filter(pl.col('method') == mcmc_method).first().alias('P1_mcmc'),
    )
)

fig, axs = plt.subplots(1, 3,
    num='articleplot6',
    clear=True,
    layout='constrained',
    figsize=[8, 2.79],
)

things = [
    (axs[0], 'alpha', '$\\alpha$'            ,           0.95, 'linear'),
    (axs[1], 'beta' , '$\\beta$'             ,              2, 'log'   ),
    (axs[2], 'P1'   , '$\\alpha 2^{-\\beta}$', 0.95 * 2 ** -2, 'linear'),
]

for ax, param, label, default, scale in things:
    ax.plot(table[param + '_gp'], table[param + '_mcmc'], '.k', alpha=0.05)
    ax.plot(default, default, 'xr', markersize=10)
    ax.set(
        xlabel=f'GP-hyp {label}',
        ylabel=f'MCMC-XCV {label}',
        xscale=scale,
        yscale=scale,
    )

axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1)

axs[1].set_yscale('linear')
axs[1].set_xlabel(f'min(GP-hyp $\\beta$, {max_gp_beta})')

axs[2].set(xlim=(0, 1), ylim=(0, 1))

fig.savefig((pathlib.Path(tablefile).parent / fig.get_label()).with_suffix('.pdf'))
fig.show()
