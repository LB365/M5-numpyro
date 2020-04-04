from utils import M5Data
import matplotlib.pyplot as plt
from numba import jit
import os
import pandas as pd
import jax.numpy as np
from jax import lax, random, vmap
from jax.nn import softmax
import numpy as onp
import numpyro
from functools import partial
import numpyro.distributions as dist
from numpyro.diagnostics import autocorrelation, hpdi
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, SA
from numpyro.infer import Predictive
from itertools import product
from datetime import datetime

assert numpyro.__version__.startswith('0.2.4')
numpyro.set_host_device_count(4)


def load_training_data(items, covariates=None):
    """
    Load sales for first item and covariates.
    :return:
    """
    if covariates is None:
        covariates = ['month']
    m5 = M5Data()
    sales = m5.get_sales()[items]
    col_snap = [m5.states[x] for x in m5.list_states[items]]
    calendar = m5.calendar_df.index.values[:sales.shape[-1]]
    variables_set = ['price',
                     'christmas',
                     'dayofweek',
                     'dayofmonth',
                     'month',
                     'snap',
                     'event']
    functions = [m5.get_prices,
                 m5.get_christmas,
                 m5.get_dummy_day_of_week,
                 m5.get_dummy_day_of_month,
                 m5.get_dummy_month_of_year,
                 m5.get_snap,
                 m5.get_event]
    _ = dict(zip(variables_set, functions))
    selected_variables = {k: _[k] for k in covariates}
    data = [f() for f in list(selected_variables.values())]
    filtered_data = [x[:sales.shape[-1], :] for x in data]
    training_data = dict(zip(covariates, filtered_data))
    training_data['sales'] = sales.T
    if 'snap' in covariates:
        training_data['snap'] = training_data['snap'][:, col_snap]
    if 'price' in covariates:
        training_data['price'] = training_data['price'][:, items]
    return calendar, training_data


def plot_sales_and_covariate(training_data, calendar):
    fig, ax = plt.subplots(nrows=len(training_data), sharex='row')
    for i, (name, item) in enumerate(training_data.items()):
        ax[i].set_title(name)
        if len(item.shape) <= 1:
            ax[i].plot(calendar, item)
        else:
            for j in range(item.shape[1]):
                ax[i].plot(calendar, item[:, j], label=r'%s %d' % (name, j))
    fig.legend()
    plt.show()


def plot_fit(forecasts, y, calendar):
    n_plots = y.shape[1]
    n_rows = int(onp.sqrt(n_plots)) + 1
    n_cols = int(n_plots // n_rows) + 1
    if y.shape[1] == 1:
        n_rows, n_cols = 1, 1
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
    for i in range(y.shape[1]):
        if y.shape[1] == 1:
            ax.plot(calendar, y, label='sales')
            ax.plot(calendar, forecasts['mean'], label='prediction')
            ax.fill_between(calendar, forecasts['lower'][:,0], forecasts['upper'][:,0], alpha=0.2, color='red')
        else:
            ax[i // n_cols, i % n_cols].plot(calendar, y[:, i], label='sales')
            ax[i // n_cols, i % n_cols].plot(calendar, forecasts['mean'][:, i], label='prediction')
            ax[i // n_cols, i % n_cols].fill_between(calendar, forecasts['lower'][:, i], forecasts['upper'][:, i],
                                                     alpha=0.2, color='red')
    fig.legend()
    plt.show()


def plot_inference(sample):
    n_plots = len(sample)
    n_rows = int(onp.sqrt(n_plots)) + 1
    n_cols = int(n_plots // n_rows) + 1
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
    for i, (key, value) in enumerate(sample.items()):
        iterator = list(product(*[range(x) for x in value.shape[1:]]))
        for t in iterator:
            ax[i // n_cols, i % n_cols].hist(value[(slice(0, value.shape[0]), *t)], bins=100,
                                             label=r'{}{}'.format(key, str(t)))
        ax[i // n_cols, i % n_cols].set_title(r'Parameter: {}'.format(key))
    fig.legend()
    plt.show()


def scan_fn_h(alpha, z_init, dz):
    def _body_fn(carry, x):
        z_prev = carry
        z_t = np.multiply(alpha, z_prev) + np.multiply((np.ones(alpha.shape) - alpha), x)
        z_prev = z_t.reshape(-1, 1)[:, -1]
        return z_prev, z_t

    return lax.scan(_body_fn, z_init, dz)


def poisson_model_hierarchical(X, X_dim, y=None):
    values = list(X_dim.values())
    n_cov = len(values)
    # Seasonality and regression effects
    l, n_, n_items = X.shape
    if X.shape[-1] > 1:
        beta_meta = numpyro.sample('beta_meta', fn=dist.Normal(0, 5))
        sigma_meta = numpyro.sample('sigma_meta', fn=dist.HalfNormal(5))
    else:
        beta_meta = numpyro.deterministic('beta_meta', value=np.array(0.))
        sigma_meta = numpyro.deterministic('sigma_meta', value=np.array(5))
    # Plate over items
    with numpyro.plate('items', n_items):
        prob = numpyro.sample('prob', fn=dist.Beta(2., 2.))
        # Plate over variables
        with numpyro.plate('n_cov', n_cov):
            beta = numpyro.sample('beta', fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1),
                                                                          transforms=dist.transforms.AffineTransform(
                                                                              loc=beta_meta, scale=sigma_meta)))
            sigma = numpyro.sample('sigma', fn=dist.HalfNormal(0.4))
        # Plate over variable dimension
        beta_long = np.repeat(beta, values, axis=0)
        sigma_long = np.repeat(sigma, values, axis=0)
        with numpyro.plate('covariates',n_):
            beta_covariates = numpyro.sample(name='beta_covariates',
                                             fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1),
                                                                             transforms=dist.transforms.AffineTransform(
                                                                                 loc=beta_long, scale=sigma_long)))
        mu = np.einsum('ijk,jk->ik', X, beta_covariates)
        # Autoregressive component
        alpha = numpyro.sample(name="alpha", fn=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.),
                                                                             transforms=dist.transforms.AffineTransform(
                                                                                 loc=0.5, scale=0.1)))
        z_init = numpyro.sample(name='z_init', fn=dist.Normal(loc=0., scale=5.))
        _, Z = scan_fn_h(alpha, z_init, mu)
        # Break detection
        if y is not None:
            brk = numpyro.deterministic('brk', (np.diff(y, n=1, axis=0) == 0).argmin(axis=0))
        else:
            brk = X.shape[0]
        # Inference
        with numpyro.plate('y', l):
            with handlers.mask(np.arange(l)[..., None] < brk):
                return numpyro.sample('obs', fn=dist.ZeroInflatedPoisson(gate=prob, rate=np.exp(Z) / prob), obs=y)


def run_inference(model, inputs, method=None):
    num_samples = 2500
    if method is None:
        kernel = NUTS(model)
    else:
        kernel = SA(model)
    tic = datetime.now()
    mcmc = MCMC(kernel, num_warmup=500, num_samples=num_samples)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, **inputs, extra_fields=('potential_energy',))
    toc = (datetime.now() - tic) / 60
    print(toc)
    print(r'Summary for: {}'.format(model.__name__))
    mcmc.print_summary(exclude_deterministic=False)
    samples = mcmc.get_samples()
    return samples


def posterior_predictive(model, samples, inputs):
    predictive = Predictive(model=model, posterior_samples=samples)
    rng_key = random.PRNGKey(0)
    forecast = predictive(rng_key=rng_key, **inputs)['obs']
    return forecast

class Metrics():

    def __init__(self,**kwargs):
        self.alpha = 0.95
        self._moments = None
        self._hit_rate = None
        self._data = None
        self._trace = None
        for name,value in kwargs.items():
            setattr(self,name,value)

    def moments(self,forecast):
        if self._moments is None:
            mean = onp.mean(forecast, axis=0)
            hpdi_0, hpdi_1 = hpdi(forecast, prob=alpha)
            names = ['lower', 'mean', 'upper']
            values = [hpdi_0, mean, hpdi_1]
            self._moments = dict(zip(names, values))
        return self._moments

    def hit_rate(self,forecast,y):
        m = self.moments(forecast)
        hit = (y > m['lower']) & (y < m['upper'])
        return hit.sum()/hit.size



def expectation_convolution(x, steps):
    x_ = onp.array(x)
    signal_ = onp.arange(steps)
    signal_inv = - onp.arange(steps)[::-1]
    signal = np.append(signal_, signal_inv)
    if len(x_.shape) <= 1:
        computation = [onp.convolve(x_, signal, mode='same').reshape(-1, 1)]
    else:
        rng = x.shape[1]
        computation = [onp.convolve(x_[:, i], signal, mode='same').reshape(-1, 1) for i in range(rng)]
    return onp.concatenate(computation, axis=1)


def log_normalise(x):
    tol = 0.001
    mask = (x.std(axis=0) < tol)
    random_price = onp.random.rand(x.shape[0])
    random_price_mask = onp.dot(random_price.reshape((-1, 1)), mask.reshape((-1, 1)).T)
    X = onp.log(x) + 0 * random_price_mask
    return (X - onp.mean(X, axis=0)) / onp.std(X, axis=0)


def hump(x, n_days):
    hump_signal = np.square(np.concatenate([np.arange(0, n_days), np.arange(0, n_days - 1)[::-1]]))
    if len(x.shape) <= 1:
        computation_ = [onp.convolve(x, hump_signal, mode='same').reshape(-1, 1)]
    else:
        rng = x.shape[1]
        computation_ = [onp.convolve(x[:, i], hump_signal, mode='same').reshape(-1, 1) for i in range(rng)]
    computation = onp.concatenate(computation_, axis=1)
    rescaler = np.max(computation)
    return 1 / rescaler * np.concatenate(computation_, axis=1)


def transform(transformation_function, training_data, t_covariates, *args):
    def _body_transform(name, value, t_covariates, *args):
        if name in t_covariates:
            return transformation_function(value, *args)
        else:
            return value

    return {name: _body_transform(name, value, t_covariates, *args)
            for name, value in training_data.items()}


def main():
    steps = 5
    n_days = 15
    items = range(5)
    variable = ['sales']  # Target variables
    covariates = ['month', 'snap', 'christmas', 'event', 'price']  # List of considered covariates
    ind_covariates = ['price', 'snap']  # Item-specific covariates
    common_covariates = set(covariates).difference(ind_covariates)  # List of non item-specific covariates
    t_covariates = ['event', 'christmas']  # List of transformed covariates
    norm_covariates = ['price']  # List of normalised covariates
    hump_covariates = ['month']  # List of convoluted covariates
    calendar, training_data = load_training_data(items=items, covariates=covariates)

    training_data = transform(expectation_convolution, training_data, t_covariates, steps)
    training_data = transform(log_normalise, training_data, norm_covariates)
    training_data = transform(hump, training_data, hump_covariates, n_days)

    plot_sales_and_covariate(training_data, calendar)
    y = np.array(training_data[variable[0]])
    X_i = np.stack([training_data[x] for x in ind_covariates], axis=1)
    X_i_dim = dict(zip(ind_covariates, [1 for x in ind_covariates]))
    X_c = np.repeat(np.hstack([training_data[i] for i in common_covariates])[:, :, np.newaxis], repeats=len(items),
                    axis=2)
    X_c_dim = dict(zip(common_covariates, [training_data[x].shape[-1] for x in common_covariates]))
    X = np.concatenate([X_c, X_i], axis=1)
    #Aggregation
    X = np.median(X,axis=-1)[...,None]
    y = np.sum(y,axis=-1)[...,None]
    X_dim = {**X_c_dim, **X_i_dim}
    inputs = {'X': X,
              'X_dim': X_dim,
              'y': y}
    samples = run_inference(poisson_model_hierarchical, inputs)

    plot_inference(samples)

    inputs.pop('y')
    trace = posterior_predictive(poisson_model_hierarchical, samples, inputs)
    forecasts = moments(trace)
    plot_fit(forecasts, y, calendar)


if __name__ == '__main__':
    main()

#######################
# Decommissioned models
#######################

# def poisson_model(X, y=None):
#     jitter = 10 ** -25
#     prob_1 = numpyro.sample('prob_1', fn=dist.Beta(2, 2))
#     beta_0 = numpyro.sample('beta_0', fn=dist.Normal(0, 3))
#     sigma_0 = numpyro.sample('sigma_0', fn=dist.HalfCauchy(5))
#     beta = numpyro.sample(name="beta",
#                           sample_shape=(X.shape[1],),
#                           fn=dist.TransformedDistribution(dist.Normal(loc=0.,
#                                                                       scale=1),
#                                                           transforms=dist.transforms.AffineTransform(loc=beta_0,
#                                                                                                      scale=sigma_0)))
#     prob_1 = np.clip(prob_1, a_min=jitter)
#     if y is not None:
#         brk = numpyro.deterministic('brk', np.min(np.nonzero(np.diff(y, n=1))))
#         prob = np.where(np.arange(0, X.shape[0]) < brk, 1, prob_1)
#         mu_ = np.tensordot(X[brk:, :], beta, axes=(1, 0))
#         mu = np.hstack([jitter * np.ones(shape=brk), mu_])
#     else:
#         mu = np.tensordot(X, beta, axes=(1, 0))
#         prob = prob_1
#     return numpyro.sample('obs', fn=dist.ZeroInflatedPoisson(gate=prob, rate=mu / prob), obs=y)

# def scan_fn(alpha, z_init, dz):
#     def _body_fn(carry, x):
#         z_prev = carry
#         z_t = alpha * z_prev + (np.ones(1) - alpha) * x
#         z_prev = z_t[-1]
#         return z_prev, z_t
#     return lax.scan(_body_fn, z_init, dz)
#
#
# def poisson_model_mask(X, X_dim, autoregressive, y=None):
#     # Seasonality and regression effects
#     jitter = 10 ** -25
#     prob = numpyro.sample('prob', fn=dist.Beta(2., 2.))
#     beta = numpyro.sample('beta', fn=dist.Normal(0., 1.), sample_shape=(len(X_dim),))
#     sigma = numpyro.sample('sigma', fn=dist.HalfNormal(0.4), sample_shape=(len(X_dim),))
#     def declare_param(i,name,dim):
#         if dim == 1:
#             return numpyro.deterministic(name=r"beta_{}".format(name), value=beta[i,np.newaxis])
#         else:
#             return numpyro.sample(name=r"beta_{}".format(name), sample_shape=(dim,),fn=dist.TransformedDistribution(
#                 dist.Normal(loc=0., scale=1),
#                 transforms=dist.transforms.AffineTransform(
#                 loc=beta[i],
#                 scale=sigma[i])))
#
#     var = {r"beta_{}".format(name): declare_param(i,name,dim)
#            for i, (name, dim) in enumerate(X_dim.items())}
#     beta_m = np.concatenate(list(var.values()), axis=0)
#     prob = np.clip(prob, a_min=jitter)
#     mu = np.tensordot(X, beta_m, axes=(1, 0))
#     # Break detection
#     if y is not None:
#         brk = numpyro.deterministic('brk', np.min(np.nonzero(np.diff(y, n=1))))
#     else:
#         brk = X.shape[0]
#     # Autoregressive component
#     if autoregressive:
#         alpha = numpyro.sample(name="alpha",fn=dist.TransformedDistribution(dist.Normal(loc=0.,scale=1.),
#                                transforms=dist.transforms.AffineTransform(loc=0.5,scale=0.15)))
#         z_init = numpyro.sample(name='z_init', fn=dist.Normal(loc=0.,scale=1.))
#         z_last, zs_exp = scan_fn(alpha, z_init, mu)
#         Z = zs_exp[:,0]
#     else:
#         Z = mu
#     # Inference
#     l,_ = X.shape
#     with numpyro.plate('y',l):
#         with handlers.mask(np.arange(l)[..., None] < brk):
#             return numpyro.sample('obs', fn=dist.ZeroInflatedPoisson(gate=prob, rate=np.exp(Z) / prob), obs=y)
