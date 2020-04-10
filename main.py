import logging
logging.basicConfig(level=logging.INFO)
from modules.transform import log_normalise,cluster,expectation_convolution,transform,hump
from modules.numpyro_models import normal_model_hierarchical,poisson_model_hierarchical
from modules.inference import run_inference,posterior_predictive
from modules.metrics import Metrics
from modules.plots import plot_fit,plot_inference,plot_parameter_by_inference,plot_sales_and_covariate
from modules.utils import load_training_data
import jax.numpy as np
from jax import lax, random, vmap
from jax.nn import softmax
import numpy as onp
import numpyro
import pyro
import torch
from pyro.contrib.forecast import ForecastingModel, Forecaster, backtest, eval_crps
from modules.pyro_models import Model1, Model2
from pyro.infer.reparam import LocScaleReparam, StableReparam
from pyro.ops.tensor_utils import periodic_cumsum, periodic_repeat, periodic_features
from pyro.ops.stats import quantile
import matplotlib.pyplot as plt
logger = logging.getLogger()

def jax_to_torch(x):
    data_ = onp.asarray(x)
    return torch.from_numpy(data_)

def load_input(clusters=None):
    assert numpyro.__version__.startswith('0.2.4')
    numpyro.set_host_device_count(4)
    logger.info('Main starting')
    steps = 5
    n_days = 15
    items = range(0,500)
    variable = ['sales']  # Target variables
    covariates = ['month', 'snap', 'christmas', 'event', 'price', 'trend']  # List of considered covariates
    ind_covariates = ['price', 'snap']  # Item-specific covariates
    common_covariates = set(covariates).difference(ind_covariates)  # List of non item-specific covariates
    t_covariates = ['event', 'christmas']  # List of transformed covariates
    norm_covariates = ['price','trend']  # List of normalised covariates
    hump_covariates = ['month']  # List of convoluted covariates
    logger.info('Loading data')
    calendar, training_data = load_training_data(items=items, covariates=covariates)
    training_data = transform(expectation_convolution, training_data, t_covariates, steps, False)
    training_data = transform(log_normalise, training_data, norm_covariates)
    training_data = transform(hump, training_data, hump_covariates, n_days)
    y = np.array(training_data[variable[0]])
    X_i = np.stack([training_data[x] for x in ind_covariates], axis=1)
    X_i_dim = dict(zip(ind_covariates, [1 for x in ind_covariates]))
    X_c = np.repeat(np.hstack([training_data[i] for i in common_covariates])[..., np.newaxis],
                    repeats=len(items),
                    axis=2)
    X_c_dim = dict(zip(common_covariates, [training_data[x].shape[-1] for x in common_covariates]))
    X = np.concatenate([X_c, X_i], axis=1)
    # Aggregation
    if clusters is not None:
        y,X,clusters = cluster(y,X,1)
    X_dim = {**X_c_dim, **X_i_dim}
    return {'X': X,
            'X_dim': X_dim,
            'y': y}

def main(pyro_backend=None,clusters=None):
    inputs = load_input(clusters)
    logger.info('Inference')
    if pyro_backend is None:
        samples_hmc = run_inference(model=normal_model_hierarchical,inputs=inputs)
        inputs.pop('y')
        trace = posterior_predictive(normal_model_hierarchical, samples_hmc, inputs)
        metric_data = {'trace':trace,
                       'actual':y,
                       'alpha':0.95}
        m = Metrics(**metric_data)
        forecasts = m.moments
        hit_rate = m.hit_rate
        plot_fit(forecasts,hit_rate, y, calendar)
        print(r'Hit rate={0:0.2f}'.format(hit_rate))
    else:
        covariates, covariate_dim, data = inputs.values()
        data, covariates = map(jax_to_torch,[data,covariates])
        data = torch.log(1+data.double())
        assert pyro.__version__.startswith('1.3.1')
        pyro.enable_validation(True)
        T0 = 0  # begining
        T2 = data.size(-2)  # end
        T1 = T2 - 1000  # train/test split
        pyro.set_rng_seed(1)
        pyro.clear_param_store()
        time = torch.arange(float(T2)) / 365
        # covariates = torch.stack([time], dim=-1)
        # covariates = torch.cat([time.unsqueeze(-1),
        #                         periodic_features(T2, 12)], dim=-1)
        covariates = covariates.squeeze(-1)
        forecaster = Forecaster(Model1(), data[:T1], covariates[:T1], learning_rate=0.05,num_steps=2000)
        samples = forecaster(data[:T1], covariates, num_samples=1000)
        p10, p50, p90 = quantile(samples, [0.1, 0.5, 0.9]).squeeze(-1)
        crps = eval_crps(samples, data[T1:])
        print(samples.shape, p10.shape)
        plt.figure(figsize=(9, 3))
        plt.plot(data, 'k-', label='truth')
        plt.fill_between(torch.arange(T1, T2), p10, p90, color="red", alpha=0.3)
        plt.plot(torch.arange(T1, T2), p50, 'r-', label='forecast')
        plt.title("Forecast (CRPS = {:0.3g})".format(crps))
        plt.ylabel("#")
        plt.legend(loc="best")
        plt.show()
        plt.savefig('figures/pyro_forecast.png')



if __name__ == '__main__':
    main(pyro_backend=True,clusters=True)