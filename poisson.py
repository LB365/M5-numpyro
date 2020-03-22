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
import statsmodels.api as sm

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

    variables_set = ['price',
                     'christmas',
                     'weekday',
                     'monthday',
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
    return training_data


def plot_sales_and_covariate(training_data):
    fig, ax = plt.subplots(nrows=len(training_data))
    for i, (name, item) in enumerate(training_data.items()):
        x_axis = range(0, training_data['sales'].shape[0])
        ax[i].set_title(name)
        if len(item.shape) <= 1:
            ax[i].plot(x_axis, item, 'o')
        else:
            for j in range(item.shape[1]):
                ax[i].plot(x_axis, item[:, j], label=r'%s %d' % (name, j))
    fig.legend()
    plt.show()


def poisson_model(X, y=None):
    jitter = 10 ** -25
    prob_1 = numpyro.sample('prob_1', fn=dist.Beta(2, 2))
    beta_0 = numpyro.sample('beta_0',fn=dist.Normal(0,3))
    sigma_0 = numpyro.sample('sigma_0',fn=dist.HalfCauchy(1))
    beta = numpyro.sample(name="beta",
                          sample_shape=(X.shape[1],),
                          fn=dist.TransformedDistribution(dist.Normal(loc=0.,
                                                                      scale=1),
                                                          transforms=dist.transforms.AffineTransform(loc=beta_0,
                                                                                                     scale=sigma_0)))
    prob_1 = np.clip(prob_1, a_min=jitter)
    if y is not None:
        brk = np.min(np.nonzero(np.diff(y, n=1)))
        prob = np.where(np.arange(0,X.shape[0])<brk,1,prob_1)
        mu_ = np.tensordot(X[brk:,:], beta, axes=(1, 0))
        mu = np.hstack([jitter*np.ones(shape=brk),mu_])
    else:
        mu = np.tensordot(X, beta, axes=(1, 0))
        prob = prob_1
    return numpyro.sample('obs', fn=dist.ZeroInflatedPoisson(gate=prob, rate=mu / prob), obs=y)


def run_inference(model, inputs):
    num_samples = 5000
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=num_samples)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, **inputs, extra_fields=('potential_energy',))
    mcmc.print_summary()
    samples = mcmc.get_samples()
    return samples


def main():
    variable = 'sales'
    covariates = ['month']
    training_data = load_training_data(covariates = covariates)
    plot_sales_and_covariate(training_data)
    y = np.array(training_data[variable])
    training_data.pop(variable)
    X = np.hstack(list(training_data.values()))
    inputs = {'X': X,
              'y': y}
    run_inference(poisson_model, inputs)

    pass


if __name__ == '__main__':
    main()
