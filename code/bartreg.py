import warnings
import functools

import lsqfitgp as lgp
import numpy as np
from numpy.lib import recfunctions
from scipy import stats, optimize, linalg, special
from jax import numpy as jnp
from jax.scipy import linalg as jlinalg
import jax
from jax import tree_util
import gvar

""" module for GP regression with the BART kernel """

class bart:
    
    def __init__(self,
        x_train,
        y_train,
        *,
        x_test=None,
        sigest=None,
        sigdf=3,
        sigquant=0.90,
        k=2,
        power=2,
        base=0.95,
        ndpost=1000,
        seed=None,
        kernelkw={},
        nu=None,
        x64=True):
        """
        GP implementation of BayesTree::bart
        
        usequants=True, numcut=inf
        
        The sigma posterior is sampled exactly i.i.d. with the ratio-of-uniforms
        method, or sigma is fixed to its MAP.
        
        Parameters
        ----------
        x_train : (n, p) array or dataframe
            Data covariates.
        y_train : (n,) array
            Data outcome.
        x_test : (n*, p) array or dataframe, optional
            Prediction covariates.
        sigest : scalar, optional
            Scale of the prior on sigma. If not specified, use the OLS
            estimate of the error standard deviation.
        sigdf : scalar
            nu (degrees of freedom of scaled inverse chisquared prior on
            sigma^2)
        sigquant : scalar
            q (sets the scale of the sigma^2 prior such that `sigest` is the
            `q`-th quantile of the prior)
        k : scalar
            k (inverse scale of the prior variance of the latent function)
        power : scalar
            beta (parameter of the tree prior, higher -> less interactions)
        base : scalar
            alpha (parameter of the tree prior, higher -> more interactions)
        ndpost : int
            The number of posterior samples.
        seed : None, int, SeedSequence, Generator
            The seed or random number generator used to sample the posterior.
            If not specified, do not sample the posterior and fix sigma to its
            maximum a posteriori (MAP).
        nu : None or scalar
            If specified, the degrees of freedom of the Student distribution.
            If not specified (default), use a Normal.
        
        Attributes
        ----------
        sigma : scalar or (ndpost,) array
            The MAP or posterior on sigma.
        logsigma_sdev : scalar
            The posterior standard deviation of log(sigma), either estimated
            from the sample or with Laplace.
        yhat_train_mean : (n,) or (ndpost, n) array
            The posterior mean of the latent function at `x_train`, conditional on sigma.
        yhat_train_var : (n,) or (ndpost, n) array
            The posterior variance of the latent function at `x_train`, conditional on sigma.
        y_train_var : (n,) or (ndpost, n) array
            The posterior variance of the predictive outcome at `x_train`, conditional on sigma.
        yhat_test_mean : (n*,) or (ndpost, n*) array
            The posterior mean of the latent function at `x_test`, conditional on sigma.
        yhat_test_var : (n*,) or (ndpost, n*) array
            The posterior variance of the latent function at `x_test`, conditional on sigma.
        y_test_var : (n*,) or (ndpost, n*) array
            The posterior variance of the outcome at `x_test`, conditional on sigma.
        yhat_train_marg_mean : (n,) array
            The marginal posterior mean of the latent function at `x_train`.
        yhat_train_marg_var : (n,) array
            The marginal posterior variance of the latent function at `x_train`.
        y_train_marg_var: (n,) array
            The marginal posterior variance of the predictive outcome at `x_train`. 
        yhat_test_marg_mean : (n*,) array
            The marginal posterior mean of the latent function at `x_test`.
        yhat_test_marg_var : (n*,) array
            The marginal posterior variance of the latent function at `x_test`.
        y_test_marg_var: (n*,) array
            The marginal posterior variance of the outcome at `x_test`.
        sigest : scalar
            The `sigest` argument if specified, else the OLS error standard
            deviation.
        sigdf, sigquant, k, power, base : scalar
            The namesake arguments.

        """

        dtype = jnp.float64 if x64 else jnp.float32
        
        # convert covariate matrices to StructuredArray
        x_train = self._to_structured(x_train, dtype)
        if x_test is None:
            x_test = np.empty_like(x_train, shape=0)
        else:
            x_test = self._to_structured(x_test, dtype)
        assert x_train.dtype == x_test.dtype
        assert x_train.ndim == x_test.ndim == 1
        assert x_train.dtype[0] == dtype
        
        # make sure data is a 1d array
        if hasattr(y_train, 'to_numpy'):
            y_train = y_train.to_numpy()
        assert y_train.shape == x_train.shape
        
        # determine inverse gamma prior on error variance
        if sigest is None:
            # compute least squares residual sdev
            sigest = self._sigest(x_train, y_train)
        self.sigest = sigest
        sigma2_alpha = sigdf / 2
        sigma2_beta = sigest ** 2 / stats.invgamma.ppf(sigquant, a=sigma2_alpha)
        quant = stats.invgamma.ppf(sigquant, a=sigma2_alpha, scale=sigma2_beta)
        np.testing.assert_allclose(quant, sigest ** 2, atol=0, rtol=1e-6)
        
        # determine prior mean and variance
        ymin = np.min(y_train)
        ymax = np.max(y_train)
        mu_mu = (ymax + ymin) / 2
        sigma_mu = (ymax - ymin) / (2 * k)
            # mu_mu and sigma_mu here are for the whole forest instead of for
            # the single tree like the standard notation
        
        # convert coordinates to indices in the tree splitting grid
        splits = lgp.BART.splits_from_coord(x_train)
        def idxstruct(x):
            idx = lgp.BART.indices_from_coord(x, splits)
            return lgp.unstructured_to_structured(idx, names=x.dtype.names)
        i_train = idxstruct(x_train)
        i_test = idxstruct(x_test)

        # define covariance function
        kw = dict(
            maxd=10,
            reset=[2,4,6,8],
        )
        kw.update(kernelkw)
        @jax.jit
        def kernel(x, y, splits, *, _kw=kw):
            kernel = lgp.BART(
                splits=splits,
                indices=True,
                alpha=dtype(base),
                beta=power,
                **_kw,
            )
            kernel *= dtype(sigma_mu ** 2)
            if nu is not None:
                # convert from covariance matrix to scaling matrix
                kernel *= dtype((nu - 2) / nu)
            return kernel(x, y)

        # compute prior covariance matrices
        Kxx = kernel(*np.ix_(i_train, i_train), splits)
        Kxsx = kernel(*np.ix_(i_test, i_train), splits)
        Kxsxs = kernel(*np.ix_(i_test, i_test), splits)
        Kxsxs_diag = jnp.diag(Kxsxs)
        assert Kxx.dtype == dtype
        
        # diagonalize train covariance matrix (Kxx == v @ diag(w) @ v.T)
        w, v = linalg.eigh(Kxx)
        threshold = np.finfo(w.dtype).eps * len(w) * np.max(np.abs(w))
        small = w < threshold
        w = np.where(small, 0, w)

        # laplace approximate p(log(sigma2) | y_train)
        @jax.jit
        def minus_log_post(logs2):
            kw = dict(y=y_train, alpha=sigma2_alpha, beta=sigma2_beta, mu=mu_mu, w=w, v=v, nu=nu)
            return -logs2 - self._log_p_sigma2_given_x(jnp.exp(logs2), **kw)
        x0 = 2 * np.log(sigest)
        minkw = dict(tol=1e-5)
        result = optimize.minimize_scalar(minus_log_post, (x0, x0 + 1), **minkw)
        assert result.success, result.message
        mode_logs2 = result.x
        prec_logs2 = jax.jacfwd(jax.jacfwd(minus_log_post))(mode_logs2)
        assert prec_logs2 > 0
        sdev_logs2 = 1 / np.sqrt(prec_logs2)
        min_logp_logs2 = result.fun

        if seed is None: # fix sigma to MAP

            # sigma posterior
            self.sigma = np.exp(mode_logs2 / 2)
            self.logsigma_sdev = sdev_logs2 / 2

            # compute train posterior mean and variances conditional on sigma
            #
            # E[ŷ_train|y,σ] = μ + Kxx (Kxx + Iσ²)⁻¹ (y - μ) =
            #                = μ + V W V' V (W + Iσ²)⁻¹ V' (y - μ) =
            #                = μ + V W / (W + Iσ²) V' (y - μ)
            #
            # Cov[ŷ_train|y,σ] = Kxx - Kxx (Kxx + Iσ²)⁻¹ Kxx =
            #                  = V W V' - V W V' V (W + Iσ²)⁻¹ V' V W V' =
            #                  = V W V' - V W² / (W + Iσ²) V' =
            #                  = V (W - W² / (W + Iσ²)) V' =
            #                  = V Wσ² / (W + Iσ²) V'
            s2 = np.exp(mode_logs2).astype(dtype)
            iws2 = 1 / (w + s2)
            vtr = v.T @ (y_train - mu_mu)
            self.yhat_train_mean = mu_mu + v @ ((w * iws2)[:, None] * vtr)
            self.yhat_train_var = (v * v) @ (w * s2 * iws2)
            
            # compute test posterior mean and variances conditional on sigma
            #
            # E[ŷ_test|y,σ] = μ + Kxsx (Kxx + Iσ²)⁻¹ (y - μ) =
            #               = μ + Kxsx V (W + Iσ²)⁻¹ V' (y - μ)
            #
            # Cov[ŷ_test|y,σ] = Kxsxs - Kxsx (Kxx + Iσ²)⁻¹ Kxxs =
            #                 = Kxsxs - Kxsx V (W + Iσ²)⁻¹ V' Kxxs =
            #                 = Kxsxs - (Kxsx V) (W + Iσ²)⁻¹ (Kxsx V)'
            Kxsxv = Kxsx @ v
            self.yhat_test_mean = mu_mu + Kxsxv @ (iws2 * vtr)
            self.yhat_test_var = Kxsxs_diag - (Kxsxv * Kxsxv) @ iws2

            # predictive variance (latent mean + i.i.d. error)
            self.y_train_var = self.yhat_train_var + self.sigma ** 2
            self.y_test_var = self.yhat_test_var + self.sigma ** 2

            # fill marginal moments (they are equal to the conditional ones
            # since sigma^2 is fixed)
            self.yhat_train_marg_mean = self.yhat_train_mean
            self.yhat_train_marg_var = self.yhat_train_var
            self.y_train_marg_var = self.y_train_var
            self.yhat_test_marg_mean = self.yhat_test_mean
            self.yhat_test_marg_var = self.yhat_test_var
            self.y_test_marg_var = self.y_test_var

            # convert to Student covariance
            if nu is not None:
                d = (vtr * iws2) @ vtr
                nupost = nu + y_train.size
                factor = (nu + d) / (nupost - 2)
                self.y_train_var *= factor
                self.y_test_var *= factor
                self.y_train_marg_var *= factor
                self.y_test_marg_var *= factor
            
        else: # sample sigma

            # determine bounds for ratio of uniforms method
            umax = 1
            @jax.jit
            def x_c_sqrt_pdf(logs2):
                return (logs2 - mode_logs2) * jnp.exp(-1/2 * (minus_log_post(logs2) - min_logp_logs2))
            result = optimize.minimize_scalar(x_c_sqrt_pdf, (mode_logs2 - sdev_logs2, mode_logs2), **minkw)
            assert result.success, result.message
            assert result.x < mode_logs2
            vmin = result.fun
            result = optimize.minimize_scalar(lambda x: -x_c_sqrt_pdf(x), (mode_logs2, mode_logs2 + sdev_logs2), **minkw)
            assert result.success, result.message
            assert result.x > mode_logs2
            vmax = -result.fun
            
            # sample sigma
            @jax.jit
            @jnp.vectorize
            def pdf(logs2):
                return jnp.exp(-(minus_log_post(logs2) - min_logp_logs2))
            seed = np.random.default_rng(seed)
            logs2 = stats.rvs_ratio_uniforms(pdf, umax, vmin, vmax, size=ndpost, c=mode_logs2, random_state=seed)
            self.sigma = np.exp(logs2 / 2)
            self.logsigma_sdev = np.std(logs2 / 2)
            
            # check that the sampled sigma is roughly compatible with the
            # laplace approximation
            assert abs(mode_logs2 - np.mean(logs2)) < min(sdev_logs2, np.std(logs2))
            
            # compute train posterior mean and variances conditional on sigma
            #
            # E[ŷ_train|y,σ] = μ + Kxx (Kxx + Iσ²)⁻¹ (y - μ) =
            #                = μ + V W V' V (W + Iσ²)⁻¹ V' (y - μ) =
            #                = μ + V W / (W + Iσ²) V' (y - μ)
            #
            # Cov[ŷ_train|y,σ] = Kxx - Kxx (Kxx + Iσ²)⁻¹ Kxx =
            #                  = V W V' - V W V' V (W + Iσ²)⁻¹ V' V W V' =
            #                  = V W V' - V W² / (W + Iσ²) V' =
            #                  = V (W - W² / (W + Iσ²)) V' =
            #                  = V Wσ² / (W + Iσ²) V'
            s2 = np.exp(logs2)[:, None].astype(dtype)
            iws2 = 1 / (w + s2)
            vtr = v.T @ (y_train - mu_mu)
            self.yhat_train_mean = mu_mu + np.einsum('ij,kj,l->ki', v, w * iws2, vtr, optimize='optimal')
            self.yhat_train_var = np.einsum('ij,kj,ji->ki', v, w * s2 * iws2, v.T, optimize='optimal')
            
            # compute test posterior mean and variances conditional on sigma
            #
            # E[ŷ_test|y,σ] = μ + Kxsx (Kxx + Iσ²)⁻¹ (y - μ) =
            #               = μ + Kxsx V (W + Iσ²)⁻¹ V' (y - μ)
            #
            # Cov[ŷ_test|y,σ] = Kxsxs - Kxsx (Kxx + Iσ²)⁻¹ Kxxs =
            #                 = Kxsxs - Kxsx V (W + Iσ²)⁻¹ V' Kxxs =
            #                 = Kxsxs - (Kxsx V) (W + Iσ²)⁻¹ (Kxsx V)'
            Kxsxv = Kxsx @ v
            self.yhat_test_mean = mu_mu + np.einsum('ik,lk,k->li', Kxsxv, iws2, vtr, optimize='optimal')
            self.yhat_test_var = Kxsxs_diag - np.einsum('ij,kj,ji->ki', Kxsxv, iws2, Kxsxv.T, optimize='optimal')
            
            # marginalize sigma
            # E[ŷ] = E[E[ŷ|σ]]
            # Var[ŷ] = E[Var[ŷ|σ]] + Var[E[ŷ|σ]]
            self.yhat_train_marg_mean = np.mean(self.yhat_train_mean, axis=0)
            self.yhat_train_marg_var = np.mean(self.yhat_train_var, axis=0) + np.var(self.yhat_train_mean, axis=0)
            self.yhat_test_marg_mean = np.mean(self.yhat_test_mean, axis=0)
            self.yhat_test_marg_var = np.mean(self.yhat_test_var, axis=0) + np.var(self.yhat_test_mean, axis=0)

            # compute prediction variance (uncertainty on mean + i.i.d. error)
            # Var[ŷ+ε] = E[Var[ŷ+ε|σ]] + Var[E[ŷ+ε|σ]] =
            #          = E[Var[ŷ|σ] + σ²] + Var[E[ŷ|σ]] =
            #          = E[Var[ŷ|σ]] + E[σ²] + Var[E[ŷ|σ]] =
            #          = Var[ŷ] + E[σ²]
            if nu is None:
                ms2 = np.mean(s2)
                self.y_train_var = self.yhat_train_var + s2
                self.y_test_var = self.yhat_test_var + s2
                self.y_train_marg_var = self.yhat_train_marg_var + ms2
                self.y_test_marg_var = self.yhat_test_marg_var + ms2
            else:
                d = (vtr * iws2) @ vtr
                nupost = nu + y_train.size
                factor = (nu + d) / (nupost - 2)
                self.y_train_var = factor[:, None] * (self.yhat_train_var + s2)
                self.y_test_var = factor[:, None] * (self.yhat_test_var + s2)
                self.y_train_marg_var = np.var(self.yhat_train_mean, axis=0) + np.mean(self.y_train_var, axis=0)
                self.y_test_marg_var = np.var(self.yhat_test_mean, axis=0) + np.mean(self.y_test_var, axis=0)

            # stuff for _log_posterior_ru
            self._w = w
            self._v = v
            self._Kxx = Kxx
            self._Kxsx = Kxsx
            self._r = y_train - mu_mu
            self._mu = mu_mu

        # stuff for log_posterior
        self._Kxsxs = Kxsxs
        self._Kxsxv = Kxsxv
        self._vtr = vtr
        self._iws2 = iws2
        self._s2 = s2.squeeze()
        self._ndpost = ndpost

        # arguments
        self.sigdf = sigdf
        self.sigquant = sigquant
        self.k = k
        self.power = power
        self.base = base

        if nu is None:
            self.nupost = None
        else:
            self.nupost = nupost
            self._nud = nu + d
            # delete the latent function properties because it is not well
            # defined for the student
            del self.yhat_train_var
            del self.yhat_test_var
            del self.yhat_train_marg_var
            del self.yhat_test_marg_var

    def log_posterior(self, y_test, seed=None, *, laplace=False, add_error_in_samples=False):
        """
        Estimate the logarithm of the posterior at test points.
        
        Parameters
        ----------
        y_test : (n*,) array
            The outcomes at `x_test`.
        seed : int, SeedSequence, Generator, optional
            Seed or random number generator. If specified, sample the posterior.
        laplace : bool, default False
            If True, and sigma has been fixed to its MAP, use a Laplace
            approximation of the sigma posterior to estimate the covariance
            matrix instead of using the covariance matrix conditional on sigma.
        add_error_in_samples : bool, default False
            If True, add the random error term to the samples.

        Returns
        -------
        log_posterior : scalar
            The logarithm of the posterior probability evaluated in `y_test`.
        estimated_neg_netropy : scalar
            An estimate of the expected value of `log_posterior`.
        yhat_test : (ndpost, n*) array
            Predictive posterior samples at test points (returned only if `seed`
            is given), either with or without error term.
        """
        if self._s2.shape:
            return self._log_posterior_ru(y_test, seed, add_error_in_samples)
        else:
            return self._log_posterior_laplace(y_test, seed, add_error_in_samples, laplace)

    def _log_posterior_ru(self, y_test, seed, add_error_in_samples):

        assert len(y_test) > 0

        # diagonalize the joint [y_train, y_test] prior covariance matrix
        K = np.block([[self._Kxx, self._Kxsx.T], [self._Kxsx, self._Kxsxs]])
        W, V = linalg.eigh(K) # K = (V * W) @ V.T
        W = np.maximum(0, W)
        
        s2 = self._s2[:, None]
        Ws2 = W + s2
        
        # compute the decomposition of the schur complement K/Kxx by using
        # the diagonalizations of K and Kxx because they do not change with
        # sigma^2
        logdet = np.sum(np.log(Ws2), axis=1) - np.sum(np.log(self._w + s2), axis=1)
        Vtr = (y_test - self.yhat_test_mean) @ V[:, -len(y_test):].T
        quad = np.einsum('ij,ij->i', Vtr / Ws2, Vtr, optimize='optimal')
        p = y_test.size
        nu = self.nupost
        if nu == None:
            norm = -1/2 * logdet - p / 2 * np.log(2 * np.pi)
            logp = norm - 1/2 * quad
            elbo = norm - 1/2 * p
        else:
            norm = special.gammaln((nu + p) / 2) - special.gammaln(nu / 2)
            norm -= 1/2 * logdet + p / 2 * np.log(self._nud * np.pi)
            logp = norm - (nu + p) / 2 * np.log1p(quad / self._nud)
            elbo = norm - (nu + p) / 2 * (special.psi((nu + p) / 2) - special.psi(nu / 2))

        # log p(Y=y) = log ∫dσ p(Y=y|∑=σ)p(∑=σ) =
        #            = log E[p(Y=y|∑)] ≈
        #            ≈ log 1/n ∑_i p(Y=y|∑=σ_i) =
        #            = log ∑_i p(Y=y|∑=σ_i) - log n
        log_posterior = special.logsumexp(logp) - np.log(logp.size)

        # E[log p(Y)] = ∫dy p(y) log p(y) =
        #             = ∫dy p(y) log ∫dσ p(y|σ)p(σ) ≥
        #             ≥ ∫dy p(y) ∫dσ p(σ) log p(y|σ) =
        #             = ∫dσ p(σ) ∫dy p(y) log p(y|σ) ≈
        #             ≈ ∫dσ p(σ) ∫dy p(y|σ) log p(y|σ) ≈
        #             ≈ 1/n ∑_i ∫dy p(y|σ_i) log p(y|σ_i)
        estimated_neg_netropy = np.mean(elbo)

        if seed is not None:
            gen = np.random.default_rng(seed)
            samples = gen.standard_normal((self._ndpost, len(K)))
            # now samples ~ i.i.d. standard normal
            
            samples = (samples * np.sqrt(W)) @ V.T
            # now samples ~ errorless centered prior on [y_train, y_test]

            samples_train = samples[:, :len(self._Kxx)]
            samples_test = samples[:, len(self._Kxx):]
            error_train = np.sqrt(s2) * gen.standard_normal(samples_train.shape)
            samples_train += error_train
            # now samples_train ~ centered prior on y_train
            
            matheron = (self._iws2 * (-samples_train @ self._v)) @ self._Kxsxv.T
            samples_test += matheron
            # now samples_test ~ errorless centered posterior on y_test

            if add_error_in_samples:
                error_test = np.sqrt(s2) * gen.standard_normal(samples_test.shape)
                samples_test += error_test
                # now samples_test ~ centered posterior on y_test

            if nu is not None:
                if not add_error_in_samples:
                    raise ValueError("Can't sample from Student without error "
                                     "term. Maybe it would work, but I haven't "
                                     "checked the math.")
                # Normal -> Student
                chisq = gen.chisquare(nu, self._ndpost)
                samples_test *= np.sqrt(self._nud / chisq)[:, None]
            
            samples_test += self.yhat_test_mean
            
            return log_posterior, estimated_neg_netropy, samples_test
        else:
            return log_posterior, estimated_neg_netropy

    def _log_posterior_laplace(self, y_test, seed, add_error_in_samples, laplace):

        dtype = self._Kxsxs.dtype

        nu = self.nupost
        
        cov = np.array(self._Kxsxs - (self._Kxsxv * self._iws2) @ self._Kxsxv.T)

        if laplace:
            # add Cov[E[y|σ]] estimated to first order
            # E[y_test|σ] = μ + Kxsx V (W + Iσ²)⁻¹ V' (y - μ)
            # Std[σ²] ≈ ∂σ²/∂log σ² Std[log σ²] =
            #         = σ² Std[log σ²]
            #         = 2σ² Std[log σ]
            # Cov[E[y_test|σ]]_ij = Cov[Kxsxv_ik iws2_k vtr_k, Kxsxv_jl iws2_l vtr_l] =
            #                     = Kxsxv_ik vtr_k Kxsxv_jl vtr_l Cov[iws2_k, iws2_l] =
            #                     = Kxsxv_ik vtr_k Kxsxv_jl vtr_l ∂iws2_k/∂σ² ∂iws2_l/∂σ² Var[σ²]
            #                     = Kxsxv_ik vtr_k Kxsxv_jl vtr_l iws2_k^2 iws2_l^2 Var[σ²]
            #                     = Kxsxv_ik vtr_k iws2_k^2 Kxsxv_jl vtr_l iws2_l^2 Var[σ²]
            std_sigma2 = 2 * self._s2 * self.logsigma_sdev
            mean_coeff = self._Kxsxv @ (self._iws2 ** 2 * self._vtr)
            cov_mean = np.outer(mean_coeff, mean_coeff) * std_sigma2 ** 2
            if nu is not None:
                # rescale the correction to be a "raw" Student scaling matrix
                # instead of a covariance matrix
                cov_mean *= (nu - 2) / self._nud
            cov += cov_mean.astype(dtype)

        # decompose covariance matrix and compute terms of the probability density
        assert cov.dtype == dtype
        r = y_test - self.yhat_test_mean
        w0, V = linalg.eigh(cov)
        w = w0 + self._s2
        threshold = np.finfo(w.dtype).eps * len(w) * np.max(np.abs(w))
        threshold = np.maximum(threshold, self._s2)
        w = np.maximum(w, threshold)
        l = V * np.sqrt(w)
        logdet = np.sum(np.log(w))
        Vtr = V.T @ r
        quad = (Vtr / w) @ Vtr
        
        p = y_test.size
        if nu == None:
            norm = -1/2 * logdet - p / 2 * np.log(2 * np.pi)
            log_posterior = norm - 1/2 * quad
            neg_entropy = norm - p / 2
        else:
            norm = special.gammaln((nu + p) / 2) - special.gammaln(nu / 2)
            norm -= 1/2 * logdet + p / 2 * np.log(self._nud * np.pi)
            log_posterior = norm - (nu + p) / 2 * np.log1p(quad / self._nud)
            neg_entropy = norm - (nu + p) / 2 * (special.psi((nu + p) / 2) - special.psi(nu / 2))

        if seed is not None:
            if add_error_in_samples:
                w = w0 + self._s2
            else:
                w = w0
            A = V * np.sqrt(np.maximum(w, 0))
            
            gen = np.random.default_rng(seed)
            samples = gen.standard_normal((self._ndpost, self.yhat_test_mean.shape[-1]))
            samples = samples @ A.T
            
            if nu is not None:
                if not add_error_in_samples:
                    raise ValueError("Can't sample from Student without error "
                                     "term. Maybe it would work, but I haven't "
                                     "checked the math.")
                chisq = gen.chisquare(nu, self._ndpost)
                samples *= np.sqrt(self._nud / chisq[:, None])
            
            samples += self.yhat_test_mean
            
            return log_posterior, neg_entropy, samples
        else:
            return log_posterior, neg_entropy

    def coverage(self, y_test, *, CL):
        nu = self.nupost
        if nu is None: # Normal
            dist = stats.norm.ppf((1 + CL) / 2)
            sc2 = self.y_test_var
        else: # Student
            dist = stats.t.ppf((1 + CL) / 2, nu)
            sc2 = self.y_test_var * (nu - 2) / nu
        shift = np.array([-1, 1])[:, None]
        if self._s2.shape:
            shift = shift[:, :, None]
            # in this case it's not really the coverage of a marginal interval,
            # but the coverage averaged over sigma, it's expectation is still CL
        interval = self.yhat_test_mean + shift * dist * np.sqrt(sc2)
        return np.mean((interval[0] <= y_test) & (y_test <= interval[1]))

    @staticmethod
    def _to_structured(x, dtype):
        if hasattr(x, 'columns'):
            x = lgp.StructuredArray.from_dataframe(x)
        elif x.dtype.names is None:
            x = lgp.unstructured_to_structured(x)
        else:
            x = lgp.StructuredArray(x)
        return tree_util.tree_map(lambda c: c.astype(dtype), x)

    @staticmethod
    def _sigest(X, y):
        assert X.size > len(X.dtype)
        X = recfunctions.structured_to_unstructured(X)
        X = np.concatenate([X, np.ones_like(X[:, :1])], axis=1) # add intercept
        beta, _, rank, _ = linalg.lstsq(X.astype('f'), y.astype('f'))
        assert beta.dtype == 'f4'
        res = y - X @ beta
        chi2 = res @ res # compute chi2 manually because linalg.lstsq won't
                         # if X is not full rank
        dof = len(y) - rank
        return np.sqrt(chi2 / dof)

    @staticmethod
    def _log_p_sigma2_given_x(sigma2, *, y, alpha, beta, mu, w, v, nu):
        prior = -(alpha + 1) * jnp.log(sigma2) - beta / sigma2
        vtr = v.T @ (y - mu)
        ws2 = w + sigma2
        logdet = jnp.sum(jnp.log(ws2))
        quad = vtr @ (vtr / ws2)
        if nu is None:
            like = -1/2 * (logdet + quad)
        else:
            like = -1/2 * (logdet + (nu + vtr.size) * jnp.log1p(quad / nu))
        return prior + like

class JaxConfig:
    
    def __init__(self, **opts):
        self.opts = opts
    
    def __enter__(self):
        self.prev = {k: jax.config.read(k) for k in self.opts}
        for k, v in self.opts.items():
            jax.config.update(k, v)
    
    def __exit__(self, *_):
        for k, v in self.prev.items():
            jax.config.update(k, v)

class barteb2:
    
    def __init__(self,
        x_train,
        y_train,
        *,
        x_test=None,
        sigest=None,
        sigdf=3,
        sigquant=0.90,
        k=2,
        power=2,
        base=0.95,
        hyperprior=None,
        fit_mean=False,
        fitkw={},
        kernelkw={},
        x64=True):
        """
        GP implementation of BayesTree::bart
        
        usequants=True, numcut=inf
        
        Hyperparameters are optimized with empirical Bayes, i.e., their
        marginal posterior probability density is maximized.
                
        Parameters
        ----------
        x_train : (n, p) or dataframe
        y_train : (n,)
        x_test : (n*, p) or dataframe
        sigdf :
            nu
        sigquant :
            q
        k :
            k
        power :
            beta
        base :
            alpha
        hyperprior :
            A Gaussian copula, coded as a dictionary of gvars, that represents
            the prior of k, power and base. Hyperparameters which are missing in
            this dictionary are left fixed at their value. The prior on sigma2
            is set by sigdf and sigquant.
        fitkw : dict
            Additional arguments passed to `lsqfitgp.empbayes_fit`.
        
        Attributes
        ----------
        yhat_train_mean : (n,)
        yhat_train_var : (n,)
        yhat_test_mean : (n*,)
        yhat_test_var : (n*,)
        sigest : scalar
        sigma : scalar
        base : scalar
        power : scalar
        k : scalar
        sigdf : scalar
        sigquant : scalar
        fit: lsqfitgp.empbayes_fit

        Methods
        -------
        log_posterior
        posterior_samples

        """
        with JaxConfig(jax_enable_x64=x64):
        
            # convert covariate matrices to StructuredArray
            x_train = self._to_structured(x_train)
            if x_test is None:
                x_test = lgp.StructuredArray(np.empty_like(x_train, shape=0))
            else:
                x_test = self._to_structured(x_test)
            assert x_train.dtype == x_test.dtype
            assert x_train.ndim == x_test.ndim == 1
            assert x_train.size > len(x_train.dtype)
        
            # make sure data is a 1d array
            if hasattr(y_train, 'to_numpy'):
                y_train = y_train.to_numpy()
                y_train = y_train.squeeze() # for dataframes
            y_train = jnp.asarray(y_train)
            assert y_train.shape == x_train.shape
        
            # calculate scale for prior on error variance
            if sigest is None:
                # compute least squares residual sdev
                sigest = self._sigest(x_train, y_train)
            self.sigest = sigest

            # determine prior mean and variance
            ymin = np.min(y_train)
            ymax = np.max(y_train)
            mu_mu = (ymax + ymin) / 2
            k_sigma_mu = (ymax - ymin) / 2
            
            # add prior on sigma2
            hyperprior = gvar.BufferDict(hyperprior)
            hassigma2 = hyperprior.has_dictkey('sigma2')
            if not hassigma2:
                alpha = sigdf / 2
                beta = sigest ** 2 / stats.invgamma.ppf(sigquant, a=alpha)
                bd = lgp.copula.makedict({
                    'sigma2': lgp.copula.invgamma(alpha, beta),
                })
                sigma2_key = next(iter(bd.keys()))
                hyperprior.update(bd)

                # cross-check invgamma parameters
                copula_sigest = stats.norm.ppf(stats.invgamma.cdf(sigest ** 2, a=alpha, scale=beta))
                np.testing.assert_allclose(lgp.copula.invgamma.invfcn(copula_sigest, alpha, beta), sigest ** 2)

            # function to get hyperparameter values
            default_hyp = dict(base=base, power=power, k=k, sigma2=np.nan)
            def get_hyp(name, hp):
                return hp.get(name, default_hyp[name])

            # splitting points and indices
            splits = lgp.BART.splits_from_coord(x_train)
            def toindices(x):
                ix = lgp.BART.indices_from_coord(x, splits)
                return lgp.unstructured_to_structured(ix, names=x.dtype.names)
            i_train = toindices(x_train)
            i_test = toindices(x_test)

            # GP factory
            def makegp(hp, *, i_train, i_test, splits, **_):
                base = jnp.clip(get_hyp('base', hp), 1e-100, 1 - 1e-15)
                power = jnp.clip(get_hyp('power', hp), 1e-100, 1e100)
                k = jnp.clip(get_hyp('k', hp), 1e-100, 1e100)
                sigma2 = jnp.clip(hp['sigma2'], 1e-100, 1e100)
                
                kw = dict(alpha=base, beta=power, maxd=10, reset=[2,4,6,8])
                kw.update(kernelkw)
                kernel = lgp.BART(splits=splits, indices=True, **kw)
                kernel *= (k_sigma_mu / k) ** 2

                if fit_mean:
                    kernel += lgp.Constant() * k_sigma_mu ** 2
                
                gp = (
                    lgp.GP(kernel, checkpos=False, checksym=False, solver='chol')
                    .addx(i_train, 'data_latent')
                    .addcov(sigma2 * jnp.eye(i_train.size), 'noise')
                    .addtransf({'data_latent': 1, 'noise': 1}, 'data')
                    .addx(i_test, 'test')
                )
                
                return gp

            # fit hyperparameters
            info = {'data': y_train - mu_mu}
            initial = gvar.mean(hyperprior)
            if not hassigma2:
                initial[sigma2_key] = copula_sigest
            options = dict(
                verbosity=3,
                raises=False,
                jit=True,
                minkw=dict(method='l-bfgs-b', options=dict(maxls=4, maxiter=100)),
                initial=initial,
                mlkw=dict(epsrel=0),
                forward=True,
            )
            options.update(fitkw)
            gpfactorykw = dict(
                i_train=i_train,
                i_test=i_test,
                splits=splits,
                mu_mu=mu_mu,
            )
            fit = lgp.empbayes_fit(hyperprior, makegp, info, gpfactorykw=gpfactorykw, **options)
            
            # extract hyperparameters from minimization result
            def get_hyp_check(name, l, u):
                hyp = get_hyp(name, fit.pmean)
                if hyp <= l or hyp >= u or np.isnan(hyp):
                    default = default_hyp[name]
                    warnings.warn(f'Hyperparameter {name}={hyp:.3g} not valid, set to default {name}={default:.3g}')
                    return default
                return hyp
            self.sigma = np.sqrt(get_hyp_check('sigma2', 0, np.inf))
            self.base = get_hyp_check('base', 0, 1)
            self.power = get_hyp_check('power', 0, np.inf)
            self.k = get_hyp_check('k', 0, np.inf)

            self.fit = fit

    @functools.cached_property
    def _yhat_mean_cov(self):
        gp = self.fit.gpfactory(self.fit.pmean, **self.fit.gpfactorykw)
        return gp.predfromdata(self.fit.data, ['data_latent', 'test'], raw=True)

    @property
    def yhat_train_mean(self):
        mean, _ = self._yhat_mean_cov
        return self.fit.gpfactorykw['mu_mu'] + mean['data_latent']

    @property
    def yhat_train_var(self):
        _, cov = self._yhat_mean_cov
        return np.diag(cov['data_latent', 'data_latent'])

    @property
    def yhat_test_mean(self):
        mean, _ = self._yhat_mean_cov
        return self.fit.gpfactorykw['mu_mu'] + mean['test']

    @property
    def yhat_test_cov(self):
        _, cov = self._yhat_mean_cov
        return cov['test', 'test']

    @property
    def yhat_test_var(self):
        return np.diag(self.yhat_test_cov)

    def log_posterior(self, y_test):
        """
        Evaluate analytically the posterior at test points.

        Parameters
        ----------
        y_test : (n*,) array
            The test values.

        Returns
        -------
        log_posterior : scalar
            The logarithm of the posterior probability evaluated in y_test.
        neg_entropy : scalar
            The expected value of `log_posterior`.
        """        
        cov = self.yhat_test_cov
        nugget = self.sigma ** 2
        cov = cov.at[jnp.diag_indices_from(cov)].add(nugget)
        l = linalg.cholesky(cov, lower=True)
        r = y_test - self.yhat_test_mean
        lr = linalg.solve_triangular(l, r, lower=True)
        logdet = 2 * np.sum(np.log(np.diag(l)))
        norm = 1/2 * logdet + y_test.size/2 * np.log(2 * np.pi)
        log_posterior = -norm - 1/2 * (lr @ lr)
        neg_entropy = -norm - 1/2 * lr.size
        return log_posterior, neg_entropy

    def posterior_samples(self, seed, nsamples_gp, *, nsamples_hp=None, add_error=False):
        """
        Sample the posterior at test points.

        Parameters
        ----------
        seed : anything accepted by `numpy.random.default_rng`
            Random seed or generator.
        nsamples_gp : int
            Number of samples to draw conditional on each hyperparameters value.
        nsamples_hp : int, optional
            Number of hyperparameters samples to draw. If not specified, fix
            hyperparameters at their MAP.
        add_error : bool, default False
            If True, add the random error term to the samples.

        Returns
        -------
        y_test : (nsamples_hp, nsamples_gp, n*) or (nsamples_gp, n*)
            Predictive posterior at test points, either with or without error.
        """
        gen = np.random.default_rng(seed)

        if nsamples_hp is None:
            hypers = [self.fit.pmean]
            nhp = 1
        else:
            hypers = lgp.raniter(self.fit.pmean, self.fit.pcov, n=nsamples_hp, rng=gen)
            nhp = nsamples_hp
        
        samples = gen.standard_normal((nhp, nsamples_gp, self.yhat_test_mean.size))

        for i, hp in enumerate(hypers):
            if nsamples_hp is None:
                mean = self.yhat_test_mean
                cov = self.yhat_test_cov
            else:
                mean, cov = self._pred(
                    self.fit.gpfactory,
                    hp,
                    self.fit.gpfactorykw,
                    self.fit.data,
                )

            if add_error:
                cov = cov.at[np.diag_indices_from(cov)].add(self.sigma ** 2)
        
            w, V = linalg.eigh(cov)
            A = V * np.sqrt(np.maximum(0, w))
            samples[i] = mean + samples[i] @ A.T

        if nsamples_hp is None:
            samples = samples.squeeze(0)

        return samples

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def _pred(gpfactory, hp, gpfactorykw, data):
        gp = gpfactory(hp, **gpfactorykw)
        return gp.predfromdata(data, 'test', raw=True)       

    @staticmethod
    def _to_structured(x):
        if hasattr(x, 'columns'):
            x = lgp.StructuredArray.from_dataframe(x)
        elif x.dtype.names is None:
            x = lgp.unstructured_to_structured(x)
        else:
            x = lgp.StructuredArray(x)
        return tree_util.tree_map(lambda c: c.astype('f4'), x)

    @staticmethod
    def _sigest(X, y):
        X = recfunctions.structured_to_unstructured(X)
        X = np.concatenate([X, np.ones_like(X[:, :1])], axis=1) # add intercept
        beta, _, rank, _ = linalg.lstsq(X.astype('f'), y.astype('f'))
        assert beta.dtype == 'f4'
        res = y - X @ beta
        chi2 = res @ res # compute chi2 manually because linalg.lstsq won't
                         # if X is not full rank
        dof = len(y) - rank
        return np.sqrt(chi2 / dof)
