import functools

import numpy as np
import numba
from scipy import stats, linalg, optimize

""" module to sample the bart prior """

default_bart_params = dict(m=200, alpha=0.95, beta=2, k=2, nu=3, q=0.9)

def sample_bart_prior(X_data, X_test, y_data, size, *, seed=0, check=True, error=True, weights=None, **bart_params):
    """
    Sample from the prior distribution of the model
    
        f(x) = sum_{j=1}^m g(x; T_j, M_j) + eps(x)
    
    where each g(x; T_j, M_j) is a decision tree with random rules T and random
    leaves M, and eps(x) is a i.i.d. random error.
    
    Parameters
    ----------
    X_data : (n, p) array
        Data, n vectors of p covariates. Used for splitting points.
    X_test : (N, p) array
        Points where the prior is computed.
    y_data : (n,) array
        Outcome. Used for the hyperparameters of the priors on mu_ij|T_j and
        sigma.
    size : int
        Number of generated samples.
    seed : int
        Seed for the random number generator, default 0.
    check : bool
        Check arguments, default True.
    error : bool
        Wether to sample and add the error term. Default True.
    weights : (p,) array, optional
        Unnormalized splitting probabilities along each covariate axis.
        Default uniform.
    
    Keyword arguments
    -----------------
    These are the hyperparameters of the BART. The default values are in the
    global variable `default_bart_params`.
    m : int
        Number of trees.
    alpha, beta : scalar
        Parameters of p(T_j), P(node nonterminal) = alpha (1 + d)^(-beta).
    k : scalar
        Inverse scale for p(mu_ij|T_j).
    nu : int
        Degrees of freedom of the chisquare for the prior on the error term.
    q : scalar
        Order of the quantile of y used to scale the prior on the error term
        sigma2_epsilon = lambda / (chi2_nu/nu), p(sigma2_epsilon < var(y)) = q.
        If q = 1 the error is disabled.
    
    Return
    ------
    y_test : (size, N) array
        Samples from the prior on f(X_test)
    """
    
    # extract BART hyperparameters
    params = dict(default_bart_params)
    params.update(**bart_params)
    m = params['m']
    alpha = params['alpha']
    beta = params['beta']
    k = params['k']
    nu = params['nu']
    q = params['q']
    
    X_data = np.transpose(X_data)
    X_test = np.transpose(X_test)

    if check:   # check all arguments
        p, n = X_data.shape
        assert n >= 2, n
        # for dim in range(p):
        #     u = np.unique(X_data[dim])
        #     assert len(u) >= 2, f'no splits for coordinate {dim} (0-based), unique value = {u.item()}'
        # TODO maybe it works even if there are degenerate coordinates with
        # no splits, but maybe it is still appropriate to warn the user
    
        p2, N = X_test.shape
        assert p == p2, (p, p2)
    
        y_data = np.asarray(y_data)
        n2, = y_data.shape
        assert n == n2, (n, n2)
        
        if weights is not None:
            weights = np.asarray(weights, float)
            assert weights.shape == (p,)
            assert np.all(weights >= 0)
            assert np.isfinite(np.sum(weights))
    
        assert size == int(size), size
        assert size > 0, size
    
        assert m == int(m), m
        assert m > 0, m
    
        assert alpha == float(alpha), alpha
        assert alpha >= 0, alpha
    
        assert beta == float(beta), beta
        assert beta >= 0, beta
    
        assert k == float(k), k
        assert k >= 0
    
        assert nu == int(nu), nu
        assert nu > 0, nu
    
        assert q == float(q), q
        assert 0 <= q <= 1, q
    
        assert seed == int(seed)

    nthreads = numba.get_num_threads()
    seeds = make_seeds(seed, nthreads + 1)
    gen = np.random.default_rng(seeds[0])
    if not error:
        q = 1
    return sample_bart_prior_impl(X_data, X_test, y_data, size, m, alpha, beta, k, nu, q, seeds[1:], gen, weights)
    
def make_seeds(seed, size):
    s = np.random.SeedSequence(seed)
    out = np.empty(size, np.uint32)
    for i in range(len(out)):
        out[i] = s.spawn(1)[0].pool[0]
    return out
    
def sample_bart_prior_impl(X_data, X_test, y_data, size, m, alpha, beta, k, nu, q, seeds, gen, weights):
    p, N = X_test.shape
    y_test = np.zeros((size, N))
    
    # pre-computed/allocated stuff for tree generation
    sort_map = make_sort_map(X_test)
    sp_map = make_sp_map(X_data, X_test)
    sp_slice = root_sp_slice(X_data)
    active_dims = np.zeros(p, bool)
    
    # divide in batches for parallelization
    batchlengths = size // len(seeds) + (np.arange(len(seeds)) < size % len(seeds))
    assert np.sum(batchlengths) == size
    assert np.unique(batchlengths).size <= 2
    batches = np.pad(np.cumsum(batchlengths), (1, 0))
    
    # cycle over samples and trees, generate trees and leaves
    size = int(size)
    m = int(m)
    alpha = float(alpha)
    beta = float(beta) # casting to avoid jit overload
    sample_bart_prior_hellpit(size, m, y_test, sort_map, sp_map, sp_slice, alpha, beta, active_dims, seeds, batches, weights)
    
    # set mean and sdev of p(mu_ij|T_j)
    mu_mu, sigma_mu = mumu_sigmamu(y_data, k, m)
    y_test *= sigma_mu
    y_test += mu_mu
    
    # add error term
    dist = var_epsilon_dist(y_data, nu, q)
    if dist is not None:
        sigma2_eps = dist.rvs((size, 1), gen)
        y_test += np.sqrt(sigma2_eps) * gen.standard_normal((size, N))
    
    return y_test

@numba.jit(nopython=True, cache=True, parallel=True)
def sample_bart_prior_hellpit(size, m, y_test, sort_map, sp_map, sp_slice, alpha, beta, active_dims, seeds, batches, weights):
    for ibatch in numba.prange(len(batches) - 1):
        np.random.seed(seeds[ibatch]) # explicit seeds because numba draws
                                      # them at random for each thread
        
        # copies to avoid parallel modification, not necessary in inner cycle
        cycle_sp_slice = np.copy(sp_slice)
        cycle_active_dims = np.copy(active_dims)
        
        for isamp in range(batches[ibatch], batches[ibatch + 1]):
            for itree in range(m):
                recursive_tree_descent(y_test[isamp], 0, sort_map, sp_map, cycle_sp_slice, alpha, beta, cycle_active_dims, weights)
            # assert np.all(cycle_sp_slice == sp_slice)
            # assert np.all(cycle_active_dims == active_dims)
            ## (!) asserts break parallelization

def var_epsilon_dist(y_data, nu, q):
    hat_sigma2 = np.var(y_data)
    dist = stats.invgamma(a=nu / 2)
    lamda = hat_sigma2 / dist.ppf(q)
    return None if lamda == 0 else stats.invgamma(a=nu / 2, scale=lamda)

def mumu_sigmamu(y_data, k, m):
    ymin = np.min(y_data)
    ymax = np.max(y_data)
    mu_mu = (ymax + ymin) / 2
    sigma_mu = (ymax - ymin) / (2 * k * np.sqrt(m))
    return mu_mu, sigma_mu
    
def make_sort_map(X_test):
    p, k = X_test.shape
    sort_map = np.empty((p, k), int)
    for dim in range(p):
        sort_map[dim] = np.argsort(X_test[dim])
    return sort_map

def make_sp_map(X_data, X_test):
    p, n = X_data.shape
    sp_map = np.full((2, p, n + 1), np.iinfo(int).max)
    for dim in range(p):
        x_data = X_data[dim]
        x_test = X_test[dim]
        x_data = np.unique(x_data)
        x_test = np.sort(x_test)
        split = (x_data[1:] + x_data[:-1]) / 2
        split = np.block([-np.inf, split, np.inf])
        sp_map[0, dim, :len(split)] = np.searchsorted(x_test, split, side='left')
        sp_map[1, dim, :len(split)] = np.searchsorted(x_test, split, side='right')
    return sp_map

def root_sp_slice(X_data):
    p, n = X_data.shape
    sp_slice = np.empty((p, 2), int)
    for dim in range(p):
        x = X_data[dim]
        u = np.unique(x)
        sp_slice[dim] = [0, len(u)]
    return sp_slice

@numba.jit(nopython=True, cache=True)
def recursive_tree_descent(y_test, d, sort_map, sp_map, sp_slice, alpha, beta, active_dims, weights):
    """
    y_test : (N,) float array
        The tree output is accumulated here
    d : int
        node depth (root = 0)
    sort_map : (p, N) int array
        argsort separately for each coordinate of X_test
    sp_map : (2, p, n + 1) int array
        map from splitting points to sorted X_test for each coordinate
    sp_slice : (p, 2) int array
        current slice (as start:end) in the splitting points indices for each
        dimension. index i = split to the left of the ith element in the
        sorted unique p coordinate of X_data
    alpha, beta : scalar
        parameters of termination probability
    active_dims : (p,) bool array
        dimensions used in ancestors' splits
    weights: (p,) array or None
        Splitting probabilities along each axis
    """
    p, N = sort_map.shape
    
    # decide if node should be nonterminal
    pnt = alpha / (1 + d) ** beta
    u = np.random.uniform(0, 1)
    nt = u < pnt
    
    # check if there are available splits
    splittable_dims, = np.nonzero(sp_slice[:, 0] + 1 < sp_slice[:, 1])
    if weights is None:
        can_split = len(splittable_dims) > 0
    else:
        restr_weights = weights[splittable_dims]
        can_split = np.sum(restr_weights) > 0
    
    # split and recurse
    if nt and can_split:
        if weights is None:
            dim_restricted = np.random.randint(0, len(splittable_dims))
        else:
            prob = restr_weights / np.sum(restr_weights)
            (dim_restricted,), = np.random.multinomial(1, prob).nonzero()
        dim = splittable_dims[dim_restricted]
        start, end = sp_slice[dim]
        
        # draw the split
        split_start = start + 1
        split_end = end - 1 + 1
        isplit = np.random.randint(split_start, split_end)

        pa = active_dims[dim]
        active_dims[dim] = True
        
        sp_slice[dim, 1] = isplit
        recursive_tree_descent(y_test, d + 1, sort_map, sp_map, sp_slice, alpha, beta, active_dims, weights)
        sp_slice[dim, 1] = end
        
        sp_slice[dim, 0] = isplit
        recursive_tree_descent(y_test, d + 1, sort_map, sp_map, sp_slice, alpha, beta, active_dims, weights)
        sp_slice[dim, 0] = start
        
        active_dims[dim] = pa

    else:       # generate leaf value and accumulate
        # note: preallocating cum makes no difference
        # note: updating cum at each node instead of computing in each leaf is
        #       slower
        cum = np.zeros(N, np.intp)
        dims, = np.nonzero(active_dims)
        for dim in dims:
            ldata, rdata = sp_slice[dim]
            ltest = sp_map[0, dim, ldata]
            rtest = sp_map[1, dim, rdata]
            indices = sort_map[dim, ltest:rtest]
            cum[indices] += 1
        mask = cum == len(dims)
        y_test[mask] += np.random.normal()
