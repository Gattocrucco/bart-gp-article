from matplotlib import colors as _colors
import colorspacious
import numpy as np
from scipy import interpolate, optimize

""" module to generate doubly perceptually uniform colormaps """

lab_to_rgb = colorspacious.cspace_converter('CAM02-UCS', 'sRGB1')
    
def uniform(colors=['black', '#f55', 'white'], N=256, lrange=(0, 100), return_pos=False):
    """
    Make a perceptually uniform colormap with increasing luminosity.
    
    Parameters
    ----------
    colors : sequence of matplotlib colors
        A sequence of colors. The hue of the colors will be preserved, but
        their luminosity will be changed and their positioning along the
        colormap scale will not be in general by evenly spaced steps.
    N : int
        The number of steps of the colormap. Default 256.
    lrange : sequence
        Two values for the start and end luminosity in range [0, 100].
        Default (0, 100).
    return_pos : bool
        If True, also return the position of the input colors along the scale.
        Default False.
    
    Return
    ------
    cmap : matplotlib.colors.ListedColormap
        A new colormap.
    l01 : sequence of numbers
        The positions along the 0-1 segment of the colors in the user-given
        sequence. Returned only if `return_pos` is True.
    
    See also
    --------
    https://github.com/matplotlib/viscm
    
    Notes
    -----
    The colormap is uniform according to the CAM02-UCS colorspace, in the sense
    that the colorspace distance between the colors specified in the `colors`
    list, after changing their luminosity as needed, is proportional to their
    distance along the 0 to 1 scale of the colormap. The same holds for the
    substeps between these "nodes".
    
    The uniformity is preserved in grayscale, assuming that the conversion is
    done zeroing the chroma parameter of CIECAM02.
    
    The RGB values in the colormap are in the sRGB1 colorspace. The same is
    assumed for the input colors.
    
    Raises
    ------
    The function may fail if there are two consecutive very similar colors in
    the list.
    """
        
    rgb0 = np.array([_colors.to_rgb(color) for color in colors])
    lab0 = colorspacious.cspace_convert(rgb0, 'sRGB1', 'CAM02-UCS')
    
    lmin, lmax = lrange
    assert 0 <= lmin <= 100, lmin
    assert 0 <= lmax <= 100, lmax
    
    if len(lab0) > 2:
        l01 = computel01(lab0, lmin, lmax)
    else:
        l01 = np.array([0, 1])
    
    lab0[:, 0] = lmin + (lmax - lmin) * l01
    abinboundary(lab0)

    dist = np.sqrt(np.sum(np.diff(lab0, axis=0) ** 2, axis=1))
    distrel = dist / np.diff(l01)
    np.testing.assert_allclose(distrel, np.mean(distrel))
    
    kw = dict(axis=0, assume_sorted=True, copy=False)
    newx = np.linspace(0, 1, N)
    lab = interpolate.interp1d(l01, lab0, **kw)(newx)
    
    np.testing.assert_allclose(np.diff(lab[:, 0]), (lmax - lmin) / (N - 1))

    distsq = np.sum(np.diff(lab, axis=0) ** 2, axis=1)
    diff = np.diff(distsq)
    maxbad = 2 + 4 * (len(lab0) - 2)
    bad = np.count_nonzero(np.abs(diff) > 1e-8)
    assert bad <= maxbad, (bad, maxbad)

    rgb = lab_to_rgb(lab)
    rgb = np.clip(rgb, 0, 1)
    
    rt = _colors.ListedColormap(rgb)
    if return_pos:
        rt = (rt, l01)
    return rt

def abinboundary(lab):
    # lab = array of triplets (l, a, b)
    # writes in-place
    
    def rgbok(l, a, b):
        rgb = lab_to_rgb([l, a, b])
        return np.max(np.abs(np.clip(rgb, 0, 1) - rgb)) < 1e-4

    def boundary(x, l, a, b):
        return rgbok(l, a * x, b * x) - 0.5

    for ilab in lab:
        l = ilab[0]
        if l < 0 or l > 100:
            ilab[1:] = 0
        elif not rgbok(*ilab):
            kw = dict(args=tuple(ilab), method='bisect', bracket=(0, 1))
            sol = optimize.root_scalar(boundary, **kw)
            assert sol.converged, sol.flag
            x = sol.root
            ilab[1:] *= x

def computel01(lab0, lmin, lmax):

    x0 = np.ones(len(lab0) - 1)

    def l01transf(x):
        diff = np.logaddexp(0, x)
        l01 = np.pad(np.cumsum(diff), (1, 0))
        return l01 / l01[-1]

    def equations(x):
        l01 = l01transf(x)
        lab = np.copy(lab0)
        lab[:, 0] = lmin + (lmax - lmin) * l01
        abinboundary(lab) # <- this operation prohibits writing down the
                          #    jacobian and makes it non-banded
        diff = np.diff(lab, axis=0)
        diffsq = diff ** 2

        dist = np.sqrt(np.sum(diffsq, axis=1))
        ldist = diff[:, 0]
        eqs = dist[1:] * ldist[:-1] - dist[:-1] * ldist[1:]

        return np.concatenate([eqs, [np.sum(x) - len(x)]])

    initeqs = equations(x0)
    np.testing.assert_allclose(initeqs[-1], 0, atol=1e-10)

    sol = optimize.root(equations, x0, method='hybr')
    assert sol.success, sol.message

    l01 = l01transf(sol.x)
    assert l01[0] == 0, l01[0]
    assert l01[-1] == 1, l01[-1]
    return l01
