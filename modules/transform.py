import pandas as pd
import jax.numpy as np
import numpy as onp
import numpyro
from functools import partial
from itertools import product
from datetime import datetime
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

def expectation_convolution(x, steps, two_sided):
    x_ = onp.array(x)
    signal_ = onp.arange(steps)
    if two_sided:
        signal_inv = - onp.arange(steps)[::-1]
        signal = np.append(signal_, signal_inv)
    else:
        signal = signal_
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
    X = onp.log(x) + random_price_mask
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

def cluster(y, X, n_clusters):
    if n_clusters != 1 :
        y_ = onp.array(y,dtype='float64')
        # mask = onp.clip((onp.diff(y, n=1, axis=0) == 0).argmin(axis=0),a_max=2 * (y.shape[0] // 3),a_min=0)
        # for i in range(mask.size):
        #     y_[range(mask[i]),i] = np.nan
        corr = pd.DataFrame(y_).corr(method='kendall')
        model = SpectralCoclustering(n_clusters=n_clusters)
        model.fit(corr)
        clusters = [model.get_indices(i)[0] for i in range(n_clusters)]
        def fn_by_cluster(x,fn,weights=None):
            if weights is not None:
                return np.concatenate([fn(x[..., rng], axis=-1, weights=weights[i])[..., np.newaxis]
                                       for i, rng in enumerate(clusters)], axis=-1)
            else:
                return np.concatenate([fn(x[..., rng], axis=-1)[..., np.newaxis]
                                       for i, rng in enumerate(clusters)],axis=-1)

        fn_market_share = lambda x: [np.sum(x[...,rng],axis=0)/np.sum(x[...,rng]) for rng in clusters]

        y_ = fn_by_cluster(y,np.sum)
        #Compute within-group market shares
        y_weights = fn_market_share(y)
        X_ = fn_by_cluster(X,np.average,y_weights)
        return y_, X_, clusters
    else:
        return np.sum(y,axis=-1)[...,np.newaxis],np.mean(X,axis=-1)[...,np.newaxis],1