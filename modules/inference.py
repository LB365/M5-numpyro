import logging
import jax.numpy as np
from jax import lax, random, vmap
from jax.nn import softmax
import numpy as onp
import numpyro
from functools import partial
import numpyro.distributions as dist
from numpyro.diagnostics import autocorrelation, hpdi
from numpyro import handlers
from numpyro.util import fori_loop
from numpyro.infer.util import init_to_prior, init_to_median
from numpyro.infer import MCMC, NUTS, SVI, SA
from numpyro.contrib.autoguide import (AutoContinuousELBO,
                                       AutoLaplaceApproximation,
                                       AutoDiagonalNormal,
                                       AutoBNAFNormal,
                                       AutoMultivariateNormal,
                                       AutoLowRankMultivariateNormal)
from numpyro.optim import Adam
from numpyro.infer import Predictive

logger = logging.getLogger()


def run_inference(model, inputs, method=None):
    if method is None:
        # NUTS
        num_samples = 5000
        logger.info('NUTS sampling')
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=300, num_samples=num_samples)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, **inputs, extra_fields=('potential_energy',))
        logger.info(r'MCMC summary for: {}'.format(model.__name__))
        mcmc.print_summary(exclude_deterministic=False)
        samples = mcmc.get_samples()
    else:
        #SVI
        logger.info('Guide generation...')
        rng_key = random.PRNGKey(0)
        guide = AutoDiagonalNormal(model=model)
        logger.info('Optimizer generation...')
        optim = Adam(0.05)
        logger.info('SVI generation...')
        svi = SVI(model, guide, optim, AutoContinuousELBO(), **inputs)
        init_state = svi.init(rng_key)
        logger.info('Scan...')
        state, loss = lax.scan(lambda x,i: svi.update(x), init_state, np.zeros(2000))
        params = svi.get_params(state)
        samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
        logger.info(r'SVI summary for: {}'.format(model.__name__))
        numpyro.diagnostics.print_summary(samples, prob=0.90, group_by_chain=False)
    return samples


def posterior_predictive(model, samples, inputs):
    inputs_ = {k:inputs[k] for k in set(inputs.keys()).difference(['y'])}
    predictive = Predictive(model=model, posterior_samples=samples)
    rng_key = random.PRNGKey(0)
    forecast = predictive(rng_key=rng_key, **inputs_)['obs']
    return forecast

def predict(model, samples, y_test, X_test, X_train, y_train):
    rng_keys = random.split(random.PRNGKey(3), samples["beta"].shape[0])
    forecast_marginal = vmap(lambda rng_key, sample: model.forecast(
        y_test.shape[0], rng_key, sample, X_test, X_train, y_train))(rng_keys, samples)
    return forecast_marginal
