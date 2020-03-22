from utils import M5Data
import matplotlib.pyplot as plt
from numba import jit
import os
import pandas as pd
from dbnomics import fetch_series
import jax.numpy as np
from jax import lax, random, vmap
from jax.nn import softmax
import numpy as onp
import numpyro

numpyro.set_host_device_count(4)
import numpyro.distributions as dist
from numpyro.diagnostics import autocorrelation, hpdi
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive

assert numpyro.__version__.startswith('0.2.4')


def load_training_data(covariates=None):
    """
    Load sales for first item and covariates.
    :return:
    """
    if covariates is None:
        covariates = ['month']
    m5 = M5Data()

    sales = m5.get_sales()[0]

    calendar = m5.calendar_df.index.values[:sales.shape[0]]

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
    filtered_data = [x[:sales.shape[0], :] for x in data]
    training_data = dict(zip(covariates, filtered_data))
    training_data['sales'] = sales
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
    fig, ax = plt.subplots()
    ax.plot(calendar, y, label='sales')
    ax.plot(calendar, forecasts['mean'], label='prediction')
    ax.fill_between(calendar, forecasts['lower'], forecasts['upper'], alpha=0.2, color='red')
    fig.legend()
    plt.show()


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


def poisson_model_mask(X, X_dim, y=None):
    jitter = 10 ** -25
    prob = numpyro.sample('prob', fn=dist.Beta(2, 2))
    beta = numpyro.sample('beta', fn=dist.Normal(0.2, 0.2), sample_shape=(len(X_dim),))
    sigma = numpyro.sample('sigma', fn=dist.HalfCauchy(0.5), sample_shape=(len(X_dim),))
    var = {r"beta_{}".format(name): numpyro.sample(name=r"beta_{}".format(name),
                                                   sample_shape=(dim,),
                                                   fn=dist.TransformedDistribution(dist.Normal(loc=0.,
                                                                                               scale=1),
                                                                                   transforms=dist.transforms.AffineTransform(
                                                                                       loc=beta[i],
                                                                                       scale=sigma[i])))
           for i, (name, dim) in enumerate(X_dim.items())}
    beta_m = np.concatenate(list(var.values()), axis=0)
    prob = np.clip(prob, a_min=jitter)
    if y is not None:
        brk = numpyro.deterministic('brk', np.min(np.nonzero(np.diff(y, n=1))))
    else:
        brk = X.shape[0]
    mu = np.tensordot(X, beta_m, axes=(1, 0))
    with handlers.mask(np.arange(X.shape[0])[..., None] < brk):
        return numpyro.sample('obs', fn=dist.ZeroInflatedPoisson(gate=prob, rate=mu / prob), obs=y)


def run_inference(model, inputs):
    num_samples = 5000
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=num_samples)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, **inputs, extra_fields=('potential_energy',))
    print(r'Summary for: {}'.format(model.__name__))
    mcmc.print_summary()
    samples = mcmc.get_samples()
    return samples


def posterior_predictive(model, samples, inputs):
    predictive = Predictive(model=model, posterior_samples=samples)
    rng_key = random.PRNGKey(0)
    forecast = predictive(rng_key=rng_key, **inputs)['obs']
    return forecast


def moments(forecast, alpha=None):
    if alpha is None:
        alpha = 0.95
    mean = np.mean(forecast, axis=0)
    hpdi_0, hpdi_1 = hpdi(forecast, prob=alpha)
    names = ['lower', 'mean', 'upper']
    values = [hpdi_0, mean, hpdi_1]
    return dict(zip(names, values))


def expectation_convolution(x, steps):
    x_ = onp.array(x)
    signal = onp.arange(0, steps)
    computation = [onp.convolve(x_[:, i], signal,mode='same').reshape(-1, 1) for i in range(x.shape[1])]
    return onp.concatenate(computation, axis=1)

def convolute(training_data,t_covariates,steps):
    def f_convolute(name,value,steps,t_covariates):
        if name in t_covariates:
            return expectation_convolution(value, steps)
        else:
            return value
    return  {name: f_convolute(name,value,steps,t_covariates)
                     for name, value in training_data.items()}


def main():
    steps = 2
    variable = 'sales'
    covariates = ['month', 'christmas', 'event']
    t_covariates = ['christmas', 'event']
    calendar, training_data = load_training_data(covariates=covariates)

    training_data = convolute(training_data,t_covariates,steps)

    plot_sales_and_covariate(training_data, calendar)
    y = np.array(training_data[variable])
    training_data.pop(variable)
    X = np.hstack(list(training_data.values()))
    X_dim = dict(zip(covariates, [x.shape[1] for x in training_data.values()]))
    inputs = {'X': X,
              'X_dim': X_dim,
              'y': y}
    samples_mask = run_inference(poisson_model_mask, inputs)
    inputs.pop('y')
    trace_mask = posterior_predictive(poisson_model_mask, samples_mask, inputs)
    forecasts_mask = moments(trace_mask)
    plot_fit(forecasts_mask, y, calendar)

    pass


if __name__ == '__main__':
    main()
