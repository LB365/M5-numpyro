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
            ax.plot(calendar, forecasts['mean'][:,0], label='prediction')
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

def plot_predict(moments,in_sample_forecasts,y_test,y_train,calendar):
    exp_1 = lambda x: np.exp(x) - 1
    y = exp_1(np.concatenate([y_train,y_test],axis=0))
    l_, m_, h_ = [exp_1(x) for x in in_sample_forecasts.values()]
    hpd_low, y_pred, hpd_high = [exp_1(x) for x in moments.values()]
    range_test = np.arange(calendar.shape[0]) >= y_train.shape[0]
    range_train = np.arange(calendar.shape[0]) < y_train.shape[0]
    n_plots = y_test.shape[-1]
    if n_plots > 3:
        n_cols = int(np.sqrt(n_plots))
        n_rows = n_cols + 1 * ((n_plots % n_cols)>0)
    else:
        n_cols, n_rows = n_plots, 1
    fig, axes = plt.subplots(nrows=n_rows,ncols=n_cols, figsize=(9, 10), sharex=True)
    if n_plots == 1:
        axes.plot(calendar, y[:, 0], color='black')
        axes.plot(calendar[range_train], m_[:, 0])
        axes.fill_between(calendar[range_train], l_[:, 0], h_[:, 0], color="green", alpha=0.2)
        axes.plot(calendar[range_test], y_pred[:, 0], lw=2, color="red")
        axes.fill_between(calendar[range_test], hpd_low[:, 0], hpd_high[:, 0], color="red", alpha=0.3)
    else:
        for i, ax in enumerate(axes.flatten()[:n_plots]):
            ax.plot(calendar, y[:, i],color='black')
            ax.plot(calendar[range_train], m_[:, i])
            ax.fill_between(calendar[range_train], l_[:, i], h_[:, i], color="green", alpha=0.2)
            ax.plot(calendar[range_test], y_pred[:, i], lw=2, color="red")
            ax.fill_between(calendar[range_test], hpd_low[:, i], hpd_high[:, i], color="red", alpha=0.3)
    fig.legend(labels=('ground truth','in-sample prediction','out-of-sample prediction',
                       f'in-sample {0.95} % interval', f'out-of-sample {0.95} % interval'))
    plt.show()