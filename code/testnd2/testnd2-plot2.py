import warnings
import pathlib
import itertools

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import wquantile
import colormap

""" makes figure 6 from the output of testnd2.py """

# which interp interval to use
interval = 'upper' # 'upper' or 'proper'

# info over rows
zcols = ['gammaavg', 'width', 'error']
depth = 2
reps = [2, 5]

# info over columns
ps = [1, 2, 3, 10]

# colorbar properties
zcol_labels = ['$\\bar\\gamma_K$', 'Max width', 'Upper bound on max error']
zcol_scales = ['linear', 'log', 'log']
zcol_ranges = [(0, 1), (1e-7, 1), (1e-7, 1)]

# fixed params for printed table
gamma0 = 1
alpha0 = 0.95
beta0 = 2

# working directory
root = pathlib.Path(__file__).parent

########################

warnings.filterwarnings('ignore', r'The resize_event function was deprecated')

# load results
data = pd.read_feather(root / 'testnd2.feather')

# compute interpolation error
if interval == 'upper':
    lower = data.lower_upper
elif interval == 'proper':
    lower = data.lower
data['interp'] = (1 - gamma0) * lower + gamma0 * data.upper
data['error'] = np.maximum(
    np.abs(data.upper_precise - data.interp),
    np.abs(data.interp - data.lower_precise),
)

# group data
df = data.groupby(['depth', 'r_upper', 'p', 'alpha', 'beta'], as_index=False).max()

# print gamma at selected alpha, beta point
df_print = df[
    ((df.alpha - alpha0).abs() < 1e-5) &
    ((df.beta - beta0).abs() < 1e-5) &
    (df.depth == depth) &
    (df.r_upper.isin(reps)) &
    (df.p.isin(ps))
]
df_print = df_print[[
    'depth',
    'r_upper',
    'p',
    'gammaavg',
    'width',
    'error',
]]
print(f'values at alpha = {alpha0}, beta = {beta0}, gamma = {gamma0}:')
print(df_print)

# plot average gamma or width vs. alpha and beta for various depths and repetition counts
plt.close('all')
nrows = len(zcols) * len(reps)
fig, axs = plt.subplots(
    num=f'testnd2-plot2',
    nrows=nrows,
    ncols=len(ps),
    figsize=[7/4 * len(ps), 4.3/3 * nrows],
    sharex=True,
    sharey=True,
    layout='constrained',
)

alpha = df.alpha.unique()
beta = df.beta.unique()

groups = df.groupby(['depth', 'r_upper', 'p'])

meshes = {}

cmap = colormap.uniform(lrange=(5, 95))

for axrow, ((zcol, vminmax, norm), rep) in zip(axs, itertools.product(zip(zcols, zcol_ranges, zcol_scales), reps)):
    
    for ax, p in zip(axrow, ps):
    
        group = groups.get_group((depth, rep, p))

        z = group.pivot(index='alpha', columns='beta', values=zcol).values
        z += np.finfo('f').eps
        # vmin, vmax = vminmax[zcol]
        # norm = 'log' if zcol != 'gammaavg' else 'linear'
        vmin, vmax = vminmax
        mesh = ax.pcolormesh(alpha, beta, z.T, vmin=vmin, vmax=vmax, shading='nearest', norm=norm, cmap=cmap)
        
        ax.plot(alpha0, beta0, 'ko', markerfacecolor='none', markersize=7)

        ss = ax.get_subplotspec()
        if ss.is_first_row():
            ax.set_title(f'p = {p}')
        if ss.is_last_row():
            ax.set_xlabel('$\\alpha$')
        if ss.is_first_col():
            ax.set_ylabel(f'r = {rep}\n$\\beta$')
        if ss.is_last_col():
            mem = meshes.setdefault(zcol, {})
            mem.setdefault('mesh', mesh)
            mem.setdefault('axs', []).append(ax)

axs[0, 0].set(
    xlim=(0, 1),
    ylim=(beta.min(), beta.max()),
    xticks=[0, 1],
    yticks=[0, 2, 4],
)

for label, norm, (zcol, mem) in zip(zcol_labels, zcol_scales, meshes.items()):
    cbar = fig.colorbar(mem['mesh'], ax=mem['axs'], label=label, aspect=15)
    if norm == 'log':
        cbar.set_ticks(np.logspace(-7, 0, 8))

fig.savefig(root / (fig.get_label() + '.pdf'))

plt.show(block=False)
