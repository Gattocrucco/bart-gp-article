from matplotlib import pyplot as plt
import numpy as np

""" makes figure 2 """

n = [8, 7]
x = (0.3, 0.75)
xp = (0.6, 0.3)

####################

rng = np.random.default_rng(202211141205)
splits = [np.sort(rng.random(n[i])) for i in range(2)]
splits[1][2] -= 0.01

kx = [np.searchsorted(splits[i], x[i]) for i in range(2)]
kxp = [np.searchsorted(splits[i], xp[i]) for i in range(2)]

nminus = [min(kx[i], kxp[i]) for i in range(2)]
nzero = [abs(kx[i] - kxp[i]) for i in range(2)]
nplus = [n[i] - max(kx[i], kxp[i]) for i in range(2)]

plt.close('all')
fig, ax = plt.subplots(num='figcounts', figsize=[5.17, 3.53])

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

dotstyle = dict(marker='.', color='black', markersize=5)
dotann = dict(xytext=(5, 5), xycoords='data', textcoords='offset points')
ax.plot(x[0], x[1], **dotstyle)
ax.annotate('$\\mathbf{x}$', x, **dotann)
ax.plot(xp[0], xp[1], **dotstyle)
ax.annotate("$\\mathbf{x}'$", xp, **dotann)

linekw = dict(linestyle=':', color='lightgray')
for s in splits[0]:
    ax.axvline(s, **linekw)
for s in splits[1]:
    ax.axhline(s, **linekw)

barkw = dict(capsize=4, color='black')
barmargin = 0.02
def hbar(y, start, end, label):
    x = (start + end) / 2
    ax.errorbar(x, y, xerr=(end - start) / 2 + barmargin, **barkw)
    ax.annotate(label, (x, y), (0, -5), 'data', 'offset points', va='top', ha='center')
def vbar(x, start, end, label):
    y = (start + end) / 2
    ax.errorbar(x, y, yerr=(end - start) / 2 + barmargin, **barkw)
    ax.annotate(label, (x, y), (5, 0), 'data', 'offset points', va='center', ha='left')

hy = min([splits[1][0], x[0], xp[0]]) - 0.05
hbar(hy, splits[0][0], splits[0][min(kx[0], kxp[0]) - 1], f'$n^-_1 = {nminus[0]}$')
hbar(hy, splits[0][min(kx[0], kxp[0])], splits[0][max(kx[0], kxp[0]) - 1], f'$n^0_1 = {nzero[0]}$')
hbar(hy, splits[0][max(kx[0], kxp[0])], splits[0][-1], f'$n^+_1 = {nplus[0]}$')
hbar(hy - 0.12, splits[0][0], splits[0][-1], f'$n_1 = {n[0]}$')
b, t = ax.get_ylim()
ax.set_ylim(b - 0.08, t)

vx = splits[0][-1] + 0.05
vbar(vx, splits[1][0], splits[1][min(kx[1], kxp[1]) - 1], f'$n^-_2 = {nminus[1]}$')
vbar(vx, splits[1][min(kx[1], kxp[1])], splits[1][max(kx[1], kxp[1]) - 1], f'$n^0_2 = {nzero[1]}$')
vbar(vx, splits[1][max(kx[1], kxp[1])], splits[1][-1], f'$n^+_2 = {nplus[1]}$')
vbar(vx + 0.23, splits[1][0], splits[1][-1], f'$n_2 = {n[1]}$')
l, r = ax.get_xlim()
ax.set_xlim(l, r + 0.15)

fig.tight_layout()
fig.savefig(fig.get_label() + '.pdf')
fig.show()
