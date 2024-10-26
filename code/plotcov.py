from matplotlib import pyplot as plt
import lsqfitgp as lgp
import numpy as np

""" makes figure 3 """

n = 100
bartkw = dict(alpha=0.95, beta=2, maxd=10, reset=[2,4,6,8], gamma=1)

fig = plt.figure('plotcov', figsize=[6.59, 3.04])
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')
fig.subplots_adjust(
    left=0.09,
    bottom=0.141,
    right=0.925,
    top=0.907,
    wspace=0.018,
)

x = np.linspace(0, 1, n + 1)
splits = lgp.BART.splits_from_coord(x[:, None])
kernel = lgp.BART(splits=splits, **bartkw)
i = round(len(x) * 30 / 100)
cov = kernel(x, x[i])
ax1.plot(x, cov, color='#e00')
ax1.set_xlabel('x')
ax1.set_ylabel(f'Cov[f(x), f({x[i]:.2g})]')
ax1.set_title('$p=1$')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

x1, x2 = np.meshgrid(x, x)
X = np.stack([x1, x2], axis=-1).view('d,d').squeeze()
splits = lgp.BART.splits_from_coord(X)
kernel = lgp.BART(splits=splits, **bartkw)
cov = kernel(X, X[i, i])
c = np.take(['#e00', 'white'], np.add(*np.meshgrid(np.arange(len(x)) // 2, np.arange(len(x)) // 2)) % 2)
ax2.plot_surface(x1, x2, cov, facecolors=c, linewidth=0, rcount=1000, ccount=1000)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_zlabel(f'Cov[f($x_1$,$x_2$), f({X[i, i]["f0"]:.2g},{X[i, i]["f1"]:.2g})]')
ax2.set_title('$p=2$')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_zlim(0, 1)

fig.show()
fig.savefig(fig.get_label() + '.pdf')
