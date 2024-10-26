import pathlib

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import contour
import polars as pl

""" makes figure 10 from the output of comp42.py """

tablefile = 'comp42/comp42.npy'

gp_method = 'gp-eb-2-5-ru'
mcmc_method = 'mcmc-cv'
max_beta = 20
fmtkw = dict(linestyle='', marker='.', color='k', alpha=0.1)

table = pl.DataFrame(np.load(tablefile))

table = (table
    .filter(pl.col('done'))
    .group_by(['dataset', 'split'])
    .agg(
        pl.col('rmse').filter(pl.col('method') == gp_method).first().name.suffix('_gp'),
        pl.col('rmse').filter(pl.col('method') == mcmc_method).first().name.suffix('_mcmc'),
        pl.col('alpha').filter(pl.col('method') == gp_method).first(),
        pl.col('beta').filter(pl.col('method') == gp_method).first(),
    )
    .with_columns(
        rmseratio=pl.col('rmse_gp') / pl.col('rmse_mcmc'),
        rmsemin=pl.min_horizontal('rmse_gp', 'rmse_mcmc'),
        betaclip=pl.min_horizontal('beta', max_beta),
    )
)

fig, axs = plt.subplots(1, 3, num='articleplot5', clear=True, layout='constrained', figsize=[8.8, 2.79])

# plot RMSE ratio vs. min RMSE
for ax in axs[:2]:
    ax.set(
        ylabel='RMSE GP-hyp / RMSE MCMC-CV',
        xlabel='min(RMSE GP-hyp, RMSE MCMC-CV)',
        xscale='log',
        yscale='log',
    )

    ax.plot(table['rmsemin'], table['rmseratio'], **fmtkw)
    ax.axhline(1, color='lightgray', linestyle=':', zorder=-1)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.plot([0.001, 1, 0.001], [0.001, 1, 1000], '-', color='lightgray', zorder=-1)

axs[1].set(
    xlim=(0.08, 1.3),
    ylim=(0.74, 1 / 0.74),
    ylabel=None,
)

xl, xh = axs[1].get_xlim()
yl, yh = axs[1].get_ylim()
margin = 1.01
axs[0].plot([xl, xl, xh, xh, xl], [yl, yh, yh, yl, yl], '-', linewidth=1, color='red')

# plot beta vs. alpha
ax = axs[2]
ax.set(
    xlabel='GP-hyp $\\alpha$',
    ylabel=f'min(GP-hyp $\\beta$, {max_beta})',
    yscale='log',
    ylim=(0.038, max_beta * 1.1),
    xlim=(0.27, 1.00),
)

# alpha vs. beta at fixed P_d
d = 1
P_d = np.array([
    1e-5, 1e-4, 1e-3, 1e-2,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
])
alpha = np.linspace(0.2, 1, 100)
beta = np.log(alpha / P_d[:, None]) / np.log(1 + d)

# create a countour set object
alpha, beta = np.broadcast_arrays(alpha, beta)
allsegs = np.stack([alpha, beta], axis=-1)[:, None, :, :]
    # axes: levels, polygon list, polygon vertices, x/y
cs = contour.ContourSet(ax, P_d, allsegs, colors='lightgray', linewidths=1)
def fmt(level):
    if level < 0.1:
        return f'$10^{{{np.log10(level):.0f}}}$'
    return f'{level:.1f}'
# code to pick the coordinates with the mouse:
# labels = cs.clabel(manual=True)
# coordinates = [label.get_position() for label in labels]
coordinates = [
    (0.45939649638426694, 0.6147117709544686),
    (0.320388972875284, 0.6796883112900388),
    (0.3318339194719253, 1.7302938703424766),
    (0.3450210842870236, 5.108586434199103),
    (0.4563669509755983, 8.833997792956016),
    (0.3197290714099672, 11.642555711450921),
    (0.4427135691037427 + 0.12, 15.43407700601519),
    (0.42858463529744206, 0.09882402752194061),
    (0.525515696779824, 0.07113431936522066),
    (0.6308558089810812, 0.07184006724738512),
    (0.7358513324871863, 0.07168712613836534),
    (0.8384174256246406, 0.06766325720912529),
    (0.9398685170821861, 0.062211525512646436),
]
cs.clabel(fmt=fmt, manual=coordinates)

# mark bart defaults
ax.plot(0.95, 2, 'xr', markersize=10)

# plot alpha and beta picked by GP-hyp
ax.plot(table['alpha'], table['betaclip'], **fmtkw)

# save figure to file and show it
fig.savefig((pathlib.Path(tablefile).parent / fig.get_label()).with_suffix('.pdf'))
fig.show()
