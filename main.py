import logging
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import numpyro
from numpyro.diagnostics import hpdi
import pyro
import torch
from modules.transform import log_normalise, cluster, expectation_convolution, transform, hump
from modules.numpyro_models import HierarchicalDrift, HierarchicalMeanReverting, HierarchicalLLM
from modules.inference import run_inference, posterior_predictive, predict
from modules.metrics import Metrics
from modules.plots import plot_fit, plot_inference, plot_parameter_by_inference, plot_sales_and_covariate, plot_predict
from modules.utils import load_training_data
from jax import lax, random, vmap
from jax.nn import softmax
from pyro.contrib.forecast import ForecastingModel, Forecaster, backtest, eval_crps, HMCForecaster
from modules.pyro_models import *
from pyro.infer.reparam import LocScaleReparam, StableReparam
from pyro.ops.tensor_utils import periodic_cumsum, periodic_repeat, periodic_features
from pyro.ops.stats import quantile

numpyro.set_host_device_count(4)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def jax_to_torch(x):
    data_ = onp.asarray(x)
    return torch.from_numpy(data_)


def load_input():
    assert numpyro.__version__.startswith('0.2.4')
    logger.info('Main starting')
    steps = 2
    n_days = 15
    items = range(200)
    variable = ['sales']  # Target variables
    covariates = ['month', 'snap', 'christmas', 'event', 'trend', 'dayofweek',
                  'thanksgiving']  # List of considered covariates
    ind_covariates = ['snap']  # Item-specific covariates
    common_covariates = set(covariates).difference(ind_covariates)  # List of non item-specific covariates
    t_covariates = ['event', 'christmas']  # List of transformed covariates
    norm_covariates = []  # List of normalised covariates
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
    X = np.concatenate([X_i, X_c], axis=1)
    # Aggregation
    y, X, clusters = cluster(y, X, 2)
    X_dim = {**X_i_dim, **X_c_dim}
    return {'X': X,
            'X_dim': X_dim,
            'y': np.log(1 + y)}, calendar


def main():
    inputs, calendar = load_input()
    logger.info('Inference')
    T1 = 1000
    T2 = inputs['X'].shape[0]
    X_train, y_train, X_test, y_test = inputs['X'][:T1], inputs['y'][:T1], inputs['X'][T1:], inputs['y'][T1:]
    inputs_train = {'X': X_train, 'y': y_train}
    Model = HierarchicalDrift(X_dim=inputs['X_dim'])
    samples = run_inference(model=Model.model, inputs=inputs_train)
    trace = posterior_predictive(Model.model, samples, inputs_train)
    # In sample forecast
    metric_data = {'trace': trace, 'actual': y_train, 'alpha': 0.95}
    in_sample = Metrics(**metric_data)
    in_sample_forecasts = in_sample.moments
    logger.info(r'In sample hit rate={0:0.2f}'.format(in_sample.hit_rate))
    plot_fit(in_sample_forecasts, in_sample.hit_rate, y_train, calendar[:T1])
    # Out of sample forecast
    logger.info('Out-of-sample forecast')
    forecasts = predict(model=Model,
                        samples=samples,
                        y_test=y_test,
                        X_test=X_test,
                        y_train=y_train,
                        X_train=X_train)
    metric_data = {'trace': forecasts, 'actual': y_test, 'alpha': 0.95}
    out_of_sample = Metrics(**metric_data)
    out_of_sample_forecasts = out_of_sample.moments
    plot_predict(out_of_sample_forecasts, in_sample_forecasts, y_test, y_train, calendar)
    logger.info(r'Out of sample hit rate={0:0.2f}'.format(out_of_sample.hit_rate))


if __name__ == '__main__':
    main()
