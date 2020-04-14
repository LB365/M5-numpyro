from numba import jit
import os
import matplotlib.pyplot as plt
import pandas as pd
from dbnomics import fetch_series
import jax.numpy as np
from jax import lax, random, vmap
from jax.nn import softmax
import numpy as onp
import numpyro; numpyro.set_host_device_count(4)
import numpyro.distributions as dist
from numpyro.diagnostics import autocorrelation, hpdi
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
import statsmodels.api as sm
assert numpyro.__version__.startswith('0.2.4')


def ar_signal(coefs, sample, epsilon):
    order = len(coefs)
    y = [epsilon[i] for i in range(0,order)]
    for i in range(order,sample):
        y_next = epsilon[i]
        for j in range(0,order):
            y_next += coefs[::-1][j] * y[i - j - 1]
        y.append(y_next)
    return y

def scan_fn(n_coefs,beta,obs_init,obs):

    def _body_fn(carry, x):
        beta, z_prev = carry
        z_mean = np.append(z_prev,x)
        z_t = np.dot(beta,z_mean)
        z_prev = z_mean[-(n_coefs-1):]
        return (beta, z_prev), z_t

    return lax.scan(_body_fn, (beta, obs_init), obs[n_coefs-1:-1])


def ar_k(n_coefs, obs=None,X=None):

    beta = numpyro.sample(name="beta",
                          sample_shape=(n_coefs,),
                          fn=dist.TransformedDistribution(dist.Normal(loc=0.,
                                                                      scale=1),
                                                          transforms=dist.transforms.AffineTransform(loc=0,
                                                                                                     scale=1,
                                                                                                     domain=dist.constraints.interval(-1,1))))
    tau = numpyro.sample(name="tau", fn=dist.HalfCauchy(scale=1))

    z_init = numpyro.sample(name='z_init',fn=dist.Normal(0,1),sample_shape=(n_coefs,))

    obs_init = z_init[:n_coefs-1]

    (beta, obs_last), zs_exp = scan_fn(n_coefs,beta,obs_init,obs)
    Z_exp = np.concatenate((z_init, zs_exp), axis=0)

    Z = numpyro.sample(name="Z", fn=dist.Normal(loc=Z_exp, scale=tau), obs=obs)

    return Z_exp, obs_last

def _forecast(future, sample, Z_exp, n_coefs):
    beta = sample['beta']
    z_exp = Z_exp[-n_coefs:]
    for t in range(future):
        mu = np.dot(beta, z_exp[-n_coefs:])
        yf = numpyro.sample("yf[{}]".format(t), dist.Normal(mu,sample['tau']))
        z_exp = np.append(z_exp,yf)


def forecast(future, rng_key, sample, y, n_obs):
    Z_exp,obs_last = handlers.substitute(ar_k, sample)(n_obs, y)
    forecast_model = handlers.seed(_forecast, rng_key)
    forecast_trace = handlers.trace(forecast_model).get_trace(future, sample, Z_exp, n_obs)
    results = [np.clip(forecast_trace["yf[{}]".format(t)]["value"], a_min=1e-30)
               for t in range(future)]
    return np.stack(results, axis=0)

def plot_inference(param,sample):

    fig,ax = plt.subplots(nrows=len(param),sharex=True)

    for i,(key, value) in enumerate(sample.items()):

        if len(value.shape) > 1:
            for j in range(0,value.shape[1]):
                ax[i].hist(sample[key][:,j],bins=100,label=r'{}[{}]'.format(key,str(j)))
                ax[i].axvline(param[key][j],color='black')
                ax[i].legend()
        else:
            ax[i].hist(sample[key], bins=100)
            ax[i].axvline(param[key],color='black')

        ax[i].set_title(r'Parameter: {}'.format(key))

    plt.show()


def main():

    # Ground truth values

    ground_truth = {'beta':   [0.5, 0.5, 0.0, -0.1],
                    'z_init': [1, 2, 0, 1],
                    'tau':0.2}

    sample = 2000
    init = len(ground_truth['z_init'])
    adj_sample = sample-init
    epsilon = onp.concatenate([onp.array(ground_truth['z_init']),
                               ground_truth['tau']*onp.random.randn(adj_sample)],axis=0)

    X = onp.random.randn(sample)

    obs = onp.array(ar_signal(ground_truth['beta'],
                              sample,
                              epsilon))

    test_sample = 1000

    y_train, y_test = np.array(obs[:sample-test_sample], dtype=np.float32), obs[sample-test_sample:]

    data_ = {'obs': y_train,
             'X':X,
             'n_coefs': 4}

    # Inference

    num_samples = 5000
    nuts_kernel = NUTS(ar_k)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=num_samples)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, **data_, extra_fields=('potential_energy',))
    mcmc.print_summary()
    samples = mcmc.get_samples()

    plot_inference(ground_truth,samples)

    rng_keys = random.split(random.PRNGKey(3), samples["tau"].shape[0])
    forecast_marginal = vmap(lambda rng_key, sample: forecast(
        y_test.shape[0], rng_key, sample, y_train,data_['n_coefs']))(rng_keys, samples)

    y_pred = np.mean(forecast_marginal, axis=0)
    sMAPE = np.mean(np.abs(y_pred - y_test) / (y_pred + y_test)) * 200
    msqrt = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print("sMAPE: {:.2f}, rmse: {:.2f}".format(sMAPE, msqrt))

    plt.figure(figsize=(8, 4))
    plt.plot(range(sample),obs)
    t_future = range(sample-test_sample,sample)
    hpd_low, hpd_high = hpdi(forecast_marginal,prob=0.95)
    plt.plot(t_future, y_pred, lw=2)
    plt.fill_between(t_future, hpd_low, hpd_high, alpha=0.3)
    plt.title("Forecasting AR model (90% HPDI)")
    plt.show()

    pass


if __name__ == '__main__':

    main()