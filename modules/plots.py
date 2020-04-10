import matplotlib.pyplot as plt
import jax.numpy as np
import numpy as onp
from itertools import product
import seaborn as sns

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


def plot_fit(forecasts, hit_rate, y, calendar):
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
    fig.text(60, .025, r'hit_rate={0:.1f}'.format(hit_rate))
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

def plot_parameter_by_inference(sample_svi,sample_hmc,parameter):
    method = ['NUTS', 'SVI']
    _ = sample_hmc[parameter], sample_svi[parameter]
    samples_dict = dict(zip(method, _))
    n_plots = _[0].size // _[0].shape[0]
    fig, ax = plt.subplots(nrows=n_plots, sharex=True)
    if n_plots > 1:
        iterator = list(product(*[range(x) for x in _[0].shape[1:]]))
        for i, t in enumerate(iterator):
            for method, sample in samples_dict.items():
                s = np.array(sample[(slice(0, sample.shape[0]), *t)])
                sns.kdeplot(s, ax=ax[i], label=r'{}: {}{}'.format(method, parameter, str(t)))
                ax[i].axvline(onp.mean(s), color='black')
                ax[i].legend()
                ax[i].set_title(r'Parameter: {}'.format(parameter))
    else:
        for method, sample in samples_dict.items():
            s = np.array(sample[(slice(0, sample.shape[0]), 0)])
            sns.kdeplot(s, ax=ax, label=r'{}: {}'.format(method, parameter))
            ax.axvline(onp.mean(s), color='black')
        ax.set_title(r'Parameter: {}'.format(parameter))
        ax.legend()
    plt.show()
