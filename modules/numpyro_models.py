import logging
import jax.numpy as np
from jax import lax
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from modules.metrics import Metrics

assert numpyro.__version__.startswith('0.2.4')
numpyro.set_host_device_count(4)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Model(object):
    """
    All models must have a numpyro model and a out of sample forecast method.
    """

    def model(self, **kwargs):
        raise NotImplementedError

    def forecast(self, **kwargs):
        raise NotImplementedError


class HierarchicalDrift(Model):

    def __init__(self, X_dim):
        """
        HierarchicalModel parameters initialisation
        :param rw: Include random walk term ?
        :param X_dim: dict with variable dimensions
        """
        self.X_dim = X_dim
        self.values = list(self.X_dim.values())
        self.n_cov = len(self.values)
        # Seasonality and regression effects
        self.l, self.n_, self.n_items = None, None, None

    def model(self, X, y=None):
        l, n_, n_items = X.shape

        if n_items > 1:
            beta_meta = numpyro.sample('beta_meta', fn=dist.Normal(0, 0.5))
            sigma_meta = numpyro.sample('sigma_meta', fn=dist.HalfNormal(0.4))
        else:
            beta_meta = numpyro.deterministic('beta_meta', value=np.array(0.))
            sigma_meta = numpyro.deterministic('sigma_meta', value=np.array(0.4))
        # Plate over items
        with numpyro.plate('items', n_items):
            dof = numpyro.sample("dof", dist.Uniform(20, 50))
            sigma_sto = numpyro.sample('sigma_sto', fn=dist.HalfNormal(scale=1))
            # Plate over variables
            with numpyro.plate('n_cov', self.n_cov):
                beta = numpyro.sample('beta', fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.),
                                                                              transforms=dist.transforms.AffineTransform(
                                                                                  loc=beta_meta, scale=sigma_meta)))
                sigma = numpyro.sample('sigma', fn=dist.HalfNormal(0.4))
            # Plate over variable dimension
            beta_long = np.repeat(beta, self.values, axis=0)
            sigma_long = np.repeat(sigma, self.values, axis=0)
            with numpyro.plate('covariates', n_):
                beta_covariates = numpyro.sample(name='beta_covariates',
                                                 fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1),
                                                                                 transforms=dist.transforms.AffineTransform(
                                                                                     loc=beta_long, scale=sigma_long)))
            mu = np.matmul(X.transpose((-1, -3, -2)), beta_covariates.T[..., None]).sum(-1).T
            # Constant
            const = numpyro.sample('const', fn=dist.Normal(0, 2))
            C = np.repeat(const[None, ...], repeats=l, axis=0)
            mu += C
            # Stochastic Trend
            sigma_rw = numpyro.sample('sigma_rw', fn=dist.HalfNormal(0.002))
            rw = numpyro.sample('rw', fn=dist.GaussianRandomWalk(sigma_rw, l))
            mu += rw.T
            # Autoregressive component
            alpha = numpyro.sample(name="alpha", fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.),
                                                                                 transforms=dist.transforms.AffineTransform(
                                                                                     loc=0.4, scale=0.3)))
            z_init = numpyro.sample('z_init', fn=dist.Normal(0, 5))
            (z_prev, mu_prev), Z = self.scan_fn(alpha, z_init, np.zeros(shape=(n_items,)), mu)
            # Break detection
            if y is not None:
                brk = numpyro.deterministic('brk', (np.diff(y, n=1, axis=0) == 0).argmin(axis=0))
            else:
                brk = X.shape[0]
            # Inference
            with numpyro.plate('y', l):
                with handlers.mask(np.arange(l)[..., None] < brk):
                    obs = numpyro.sample('obs', fn=dist.StudentT(df=dof, loc=Z, scale=sigma_sto), obs=y)
                    return z_prev, mu_prev

    def scan_fn(self, alpha, z_init, dz_0, dz):
        def _body_fn(carry, x):
            z_prev, x_ = carry
            z_t = np.multiply(alpha, (z_prev - x_)) + x
            z_prev = z_t.reshape(-1, 1)[:, -1]
            return (z_prev, x), z_t

        return lax.scan(_body_fn, (z_init, dz_0), dz)

    def _forecast(self, future, sample, X, z_prev, mu_prev):
        beta = sample['beta_covariates']
        mu = np.matmul(X.transpose((-1, -3, -2)), beta.T[..., None]).sum(-1).T + sample['const']
        rw_prev = sample['rw'][..., -1]
        for t in range(future):
            rw_ = numpyro.sample(f"rw[{t}]", dist.Normal(rw_prev, sample['sigma_rw']))
            mu_ = mu[t] + rw_
            z_ = np.multiply(sample['alpha'], z_prev - mu_prev) + mu_
            yf = numpyro.sample(f"yf[{t}]", dist.StudentT(sample['dof'], z_, sample['sigma_sto']))
            rw_prev, z_prev, mu_prev = rw_, z_, mu_

    def forecast(self, future, rng_key, sample, X_test, X_train, y):
        z_prev, mu_prev = handlers.substitute(self.model, sample)(X_train, y)
        forecast_model = handlers.seed(self._forecast, rng_key)
        forecast_trace = handlers.trace(forecast_model).get_trace(future, sample, X_test, z_prev, mu_prev)
        results = [np.clip(forecast_trace[f"yf[{t}]"]["value"], a_min=1e-30)
                   for t in range(future)]
        return np.stack(results, axis=0)


class HierarchicalLLM(Model):

    def __init__(self, X_dim):
        """
        HierarchicalModel parameters initialisation
        :param rw: Include random walk term ?
        :param X_dim: dict with variable dimensions
        """
        self.X_dim = X_dim
        self.values = list(self.X_dim.values())
        self.n_cov = len(self.values)
        # Seasonality and regression effects
        self.l, self.n_, self.n_items = None, None, None

    def model(self, X, y=None):
        l, n_, n_items = X.shape

        if n_items > 1:
            beta_meta = numpyro.sample('beta_meta', fn=dist.Normal(0, 0.5))
            sigma_meta = numpyro.sample('sigma_meta', fn=dist.HalfNormal(0.4))
        else:
            beta_meta = numpyro.deterministic('beta_meta', value=np.array(0.))
            sigma_meta = numpyro.deterministic('sigma_meta', value=np.array(0.4))
        # Plate over items
        with numpyro.plate('items', n_items):
            sigma_sto = numpyro.sample('sigma_sto', fn=dist.HalfNormal(scale=1))
            # Plate over variables
            with numpyro.plate('n_cov', self.n_cov):
                beta = numpyro.sample('beta', fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.),
                                                                              transforms=dist.transforms.AffineTransform(
                                                                                  loc=beta_meta, scale=sigma_meta)))
                sigma = numpyro.sample('sigma', fn=dist.HalfNormal(0.4))
            # Plate over variable dimension
            beta_long = np.repeat(beta, self.values, axis=0)
            sigma_long = np.repeat(sigma, self.values, axis=0)
            with numpyro.plate('covariates', n_):
                beta_covariates = numpyro.sample(name='beta_covariates',
                                                 fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1),
                                                                                 transforms=dist.transforms.AffineTransform(
                                                                                     loc=beta_long, scale=sigma_long)))
            mu = np.matmul(X.transpose((-1, -3, -2)), beta_covariates.T[..., None]).sum(-1).T
            # Constant
            const = numpyro.sample('const', fn=dist.Normal(0, 100))
            C = np.repeat(const[None, ...], repeats=l, axis=0)
            mu += C
            # Stochastic Trend
            sigma_trend = numpyro.sample('sigma_trend', fn=dist.HalfNormal(0.002))
            rw = numpyro.sample('rw', fn=dist.GaussianRandomWalk(sigma_trend, l))
            # Autoregressive component
            alpha = numpyro.sample(name="alpha", fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.),
                                                                                 transforms=dist.transforms.AffineTransform(
                                                                                     loc=0.0, scale=0.5,
                                                                                     domain=dist.constraints.interval(
                                                                                         -1, 1))))
            (z_prev, mu_prev, rw_prev), Z = self.scan_fn(alpha=alpha,
                                                         mu_0=mu[0],
                                                         rw_0=rw.T[0],
                                                         rw=rw.T[1:],
                                                         mu=mu[1:],
                                                         dz=np.arange(1,l))
            Z = np.concatenate([mu[0].reshape(-1,n_items),Z],axis=0)
            # Break detection
            if y is not None:
                brk = numpyro.deterministic('brk', (np.diff(y, n=1, axis=0) == 0).argmin(axis=0))
            else:
                brk = X.shape[0]
            # Inference
            with numpyro.plate('y', l):
                with handlers.mask(np.arange(l)[..., None] < brk):
                    obs = numpyro.sample('obs', fn=dist.Normal(loc=Z, scale=sigma_sto), obs=y)
                    return z_prev, mu_prev, rw_prev

    def scan_fn(self, alpha, mu_0, rw_0, rw, mu, dz):
        def _body_fn(carry, x):
            z_prev, x_, rw_ = carry
            z_t = np.multiply(alpha, (z_prev - x_)) + mu[x] + rw_ + rw[x]
            z_prev = z_t.reshape(-1, 1)[:, -1]
            return (z_prev, mu[x], rw_ + rw[x]), z_t
        return lax.scan(_body_fn, (mu_0, mu_0, rw_0), dz)

    def _forecast(self, future, sample, X, z_prev, mu_prev, rw_prev):
        beta = sample['beta_covariates']
        mu = np.matmul(X.transpose((-1, -3, -2)), beta.T[..., None]).sum(-1).T + sample['const']
        for t in range(future):
            rw_ = numpyro.sample(f"rw[{t}]", dist.Normal(0.0, sample['sigma_trend']))
            z_ = np.multiply(sample['alpha'], z_prev - mu_prev) + mu[t] + rw_ + rw_prev
            yf = numpyro.sample(f"yf[{t}]", dist.Normal(z_, sample['sigma_sto']))
            rw_prev, z_prev, mu_prev = rw_ + rw_prev, yf, mu[t]

    def forecast(self, future, rng_key, sample, X_test, X_train, y):
        z_prev, mu_prev, rw_prev = handlers.substitute(self.model, sample)(X_train, y)
        forecast_model = handlers.seed(self._forecast, rng_key)
        forecast_trace = handlers.trace(forecast_model).get_trace(future, sample, X_test, z_prev, mu_prev, rw_prev)
        results = [np.clip(forecast_trace[f"yf[{t}]"]["value"], a_min=1e-30)
                   for t in range(future)]
        return np.stack(results, axis=0)


class HierarchicalMeanReverting(Model):

    def __init__(self, X_dim):
        """
        HierarchicalModel parameters initialisation
        :param X_dim: dict with variable dimensions
        """
        self.X_dim = X_dim
        self.values = list(self.X_dim.values())
        self.n_cov = len(self.values)
        # Seasonality and regression effects
        self.l, self.n_, self.n_items = None, None, None

    def model(self, X, y=None):
        l, n_, n_items = X.shape

        if n_items > 1:
            beta_meta = numpyro.sample('beta_meta', fn=dist.Normal(0, 0.5))
            sigma_meta = numpyro.sample('sigma_meta', fn=dist.HalfNormal(0.4))
        else:
            beta_meta = numpyro.deterministic('beta_meta', value=np.array(0.))
            sigma_meta = numpyro.deterministic('sigma_meta', value=np.array(0.4))
        # Plate over items
        with numpyro.plate('items', n_items):
            dof = numpyro.sample("dof", dist.Uniform(1, 50))
            sigma_sto = numpyro.sample('sigma_sto', fn=dist.HalfNormal(scale=1))
            const = numpyro.sample('const', fn=dist.HalfNormal(2))
            C = np.repeat(const[None, ...], repeats=l, axis=0)
            # Plate over variables
            with numpyro.plate('n_cov', self.n_cov):
                beta = numpyro.sample('beta', fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.),
                                                                              transforms=dist.transforms.AffineTransform(
                                                                                  loc=beta_meta, scale=sigma_meta)))
                sigma = numpyro.sample('sigma', fn=dist.HalfNormal(0.4))
            # Plate over variable dimension
            beta_long = np.repeat(beta, self.values, axis=0)
            sigma_long = np.repeat(sigma, self.values, axis=0)
            with numpyro.plate('covariates', n_):
                beta_covariates = numpyro.sample(name='beta_covariates',
                                                 fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1),
                                                                                 transforms=dist.transforms.AffineTransform(
                                                                                     loc=beta_long, scale=sigma_long)))
            mu = np.matmul(X.transpose((-1, -3, -2)), beta_covariates.T[..., None]).sum(-1).T
            mu += C
            # Autoregressive component
            alpha = numpyro.sample(name="alpha", fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.),
                                                                                 transforms=dist.transforms.AffineTransform(
                                                                                     loc=0.5, scale=0.3)))
            _, Z = self.scan_fn(alpha, np.zeros(shape=(n_items,)), mu)
            # Break detection
            if y is not None:
                brk = numpyro.deterministic('brk', (np.diff(y, n=1, axis=0) == 0).argmin(axis=0))
            else:
                brk = X.shape[0]
            # Inference
            with numpyro.plate('y', l):
                with handlers.mask(np.arange(l)[..., None] < brk):
                    obs = numpyro.sample('obs', fn=dist.StudentT(df=dof, loc=Z, scale=sigma_sto), obs=y)
                    return obs

    def scan_fn(self, alpha, z_init, dz):
        def _body_fn(carry, x):
            z_prev = carry
            z_t = np.multiply(alpha, z_prev) + np.multiply((np.ones(alpha.shape) - alpha), x)
            z_prev = z_t.reshape(-1, 1)[:, -1]
            return z_prev, z_t

        return lax.scan(_body_fn, z_init, dz)

    def _forecast(self, future, sample, X, last):
        beta = sample['beta_covariates']
        mu = np.matmul(X.transpose((-1, -3, -2)), beta.T[..., None]).sum(-1).T + sample['const']
        last_ = np.mean(last)
        one = np.ones(last.shape[-1])
        for t in range(future):
            z = np.multiply(sample['alpha'], last_) + np.multiply((one - sample['alpha']), mu[t])
            yf = numpyro.sample(f"yf[{t}]", dist.StudentT(sample['dof'], z, sample['sigma_sto']))
            last_ = z

    def forecast(self, future, rng_key, sample, X, y):
        last = handlers.substitute(self.model, sample)(X, y)
        forecast_model = handlers.seed(self._forecast, rng_key)
        forecast_trace = handlers.trace(forecast_model).get_trace(future, sample, X, last)
        results = [np.clip(forecast_trace[f"yf[{t}]"]["value"], a_min=1e-30)
                   for t in range(future)]
        return np.stack(results, axis=0)


#######################
# Decommissioned models
#######################


def poisson_model_hierarchical(X, X_dim, y=None):
    values = list(X_dim.values())
    n_cov = len(values)
    # Seasonality and regression effects
    l, n_, n_items = X.shape
    if X.shape[-1] > 1:
        beta_meta = numpyro.sample('beta_meta', fn=dist.Normal(0, 0.5))
        sigma_meta = numpyro.sample('sigma_meta', fn=dist.HalfNormal(0.4))
    else:
        beta_meta = numpyro.deterministic('beta_meta', value=np.array(0.))
        sigma_meta = numpyro.deterministic('sigma_meta', value=np.array(0.4))
    # Plate over items
    with numpyro.plate('items', n_items):
        const = numpyro.sample('const', fn=dist.Normal(0, 5.))
        C = np.repeat(const[None, ...], repeats=l, axis=0)
        prob = numpyro.sample('prob', fn=dist.Beta(2., 2.))
        # Plate over variables
        with numpyro.plate('n_cov', n_cov):
            beta = numpyro.sample('beta', fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1),
                                                                          transforms=dist.transforms.AffineTransform(
                                                                              loc=beta_meta, scale=sigma_meta)))
            sigma = numpyro.sample('sigma', fn=dist.HalfNormal(0.3))
        # Plate over variable dimension
        beta_long = np.repeat(beta, values, axis=0)
        sigma_long = np.repeat(sigma, values, axis=0)
        with numpyro.plate('covariates', n_):
            beta_covariates = numpyro.sample(name='beta_covariates',
                                             fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1),
                                                                             transforms=dist.transforms.AffineTransform(
                                                                                 loc=beta_long, scale=sigma_long)))
        mu = np.einsum('ijk,jk->ik', X, beta_covariates)
        # Autoregressive component
        alpha = numpyro.sample(name="alpha", fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.),
                                                                             transforms=dist.transforms.AffineTransform(
                                                                                 loc=0.5, scale=0.1)))
        _, Z = scan_fn_h(alpha, np.zeros(shape=(n_items,)), mu)
        Z += C
        # Break detection
        if y is not None:
            brk = numpyro.deterministic('brk', (np.diff(y, n=1, axis=0) == 0).argmin(axis=0))
        else:
            brk = X.shape[0]
        # Inference
        with numpyro.plate('y', l):
            with handlers.mask(np.arange(l)[..., None] < brk):
                return numpyro.sample('obs', fn=dist.ZeroInflatedPoisson(gate=prob, rate=np.exp(Z) / prob), obs=y)


def poisson_model(X, y=None):
    jitter = 10 ** -25
    prob_1 = numpyro.sample('prob_1', fn=dist.Beta(2, 2))
    beta_0 = numpyro.sample('beta_0', fn=dist.Normal(0, 3))
    sigma_0 = numpyro.sample('sigma_0', fn=dist.HalfCauchy(5))
    beta = numpyro.sample(name="beta",
                          sample_shape=(X.shape[1],),
                          fn=dist.TransformedDistribution(dist.Normal(loc=0.,
                                                                      scale=1),
                                                          transforms=dist.transforms.AffineTransform(loc=beta_0,
                                                                                                     scale=sigma_0)))
    prob_1 = np.clip(prob_1, a_min=jitter)
    if y is not None:
        brk = numpyro.deterministic('brk', np.min(np.nonzero(np.diff(y, n=1))))
        prob = np.where(np.arange(0, X.shape[0]) < brk, 1, prob_1)
        mu_ = np.tensordot(X[brk:, :], beta, axes=(1, 0))
        mu = np.hstack([jitter * np.ones(shape=brk), mu_])
    else:
        mu = np.tensordot(X, beta, axes=(1, 0))
        prob = prob_1
    return numpyro.sample('obs', fn=dist.ZeroInflatedPoisson(gate=prob, rate=mu / prob), obs=y)


def scan_fn(alpha, z_init, dz):
    def _body_fn(carry, x):
        z_prev = carry
        z_t = alpha * z_prev + (np.ones(1) - alpha) * x
        z_prev = z_t[-1]
        return z_prev, z_t

    return lax.scan(_body_fn, z_init, dz)


def poisson_model_mask(X, X_dim, autoregressive, y=None):
    # Seasonality and regression effects
    jitter = 10 ** -25
    prob = numpyro.sample('prob', fn=dist.Beta(2., 2.))
    beta = numpyro.sample('beta', fn=dist.Normal(0., 1.), sample_shape=(len(X_dim),))
    sigma = numpyro.sample('sigma', fn=dist.HalfNormal(0.4), sample_shape=(len(X_dim),))

    def declare_param(i, name, dim):
        if dim == 1:
            return numpyro.deterministic(name=r"beta_{}".format(name), value=beta[i, np.newaxis])
        else:
            return numpyro.sample(name=r"beta_{}".format(name), sample_shape=(dim,), fn=dist.TransformedDistribution(
                dist.Normal(loc=0., scale=1),
                transforms=dist.transforms.AffineTransform(
                    loc=beta[i],
                    scale=sigma[i])))

    var = {r"beta_{}".format(name): declare_param(i, name, dim)
           for i, (name, dim) in enumerate(X_dim.items())}
    beta_m = np.concatenate(list(var.values()), axis=0)
    prob = np.clip(prob, a_min=jitter)
    mu = np.tensordot(X, beta_m, axes=(1, 0))
    # Break detection
    if y is not None:
        brk = numpyro.deterministic('brk', np.min(np.nonzero(np.diff(y, n=1))))
    else:
        brk = X.shape[0]
    # Autoregressive component
    if autoregressive:
        alpha = numpyro.sample(name="alpha", fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.),
                                                                             transforms=dist.transforms.AffineTransform(
                                                                                 loc=0.5, scale=0.15)))
        z_init = numpyro.sample(name='z_init', fn=dist.Normal(loc=0., scale=1.))
        z_last, zs_exp = scan_fn(alpha, z_init, mu)
        Z = zs_exp[:, 0]
    else:
        Z = mu
    # Inference
    l, _ = X.shape
    with numpyro.plate('y', l):
        with handlers.mask(np.arange(l)[..., None] < brk):
            return numpyro.sample('obs', fn=dist.ZeroInflatedPoisson(gate=prob, rate=np.exp(Z) / prob), obs=y)
