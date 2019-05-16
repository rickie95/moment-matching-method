from scipy import optimize, stats
import numpy as np
import matplotlib.pyplot as plt


def compute_missing_value_intervals(values):
    values = np.argwhere(np.isnan(values)).reshape((1, -1))[0]
    intervals = []
    sub = []
    sub.append(values[0])
    for i in range(1, len(values)):
        if values[i] - values[i - 1] > 1:
            sub.append(values[i - 1])
            intervals.append(sub)
            sub = []
            sub.append(values[i])
    sub.append(values[-1])
    intervals.append(sub)
    l1 = 0
    l2 = 0
    l3 = 0

    for e in intervals:
        if (e[1] - e[0]) >= 2:
            l1 += 1
        if (e[1] - e[0]) == 1:
            l2 += 1
        if (e[1] - e[0]) == 0:
            l3 += 1

    return intervals, l1, l2, l3


def create_bounds(values, feasible, missing_intervals):
    max_value = np.nanmax(values)
    min_value = np.nanmin(values)
    upper_bounds = np.copy(values)
    lower_bounds = np.copy(values)
    for interval in missing_intervals:
        upper_bounds[interval[0]:interval[1] + 1] = max_value
        lower_bounds[interval[0]:interval[1] + 1] = min_value
    return optimize.Bounds(lower_bounds, upper_bounds, feasible)


def moments(x, order):
    return np.sum(np.power(x, order))/x.shape[0]


def autocorrelation(x, lag, k, mean):
    # [ x x x x 0 0 0 ] * [ 0 0 0 x x x x ]
    x_pad = np.append(x - mean, np.zeros(lag))
    x_lag = np.append(np.zeros(lag), x - mean)
    return np.dot(x_pad, x_lag) / stats.moment(x_pad, k)


def shaking(serie, missing_intervs, upper, lower):  # , lower_bound, upper_bound):  # Return a time series pertubated in his null values
    for interval in missing_intervs:
        values = stats.uniform.rvs(loc=-(upper/lower), scale=2*(upper/lower), size=(interval[1] - interval[0]))
        serie[interval[0]:interval[1]:1] += values
    return serie


def plot(original, predicted, xlabel=None, ylabel=None, filename=None):
    plt.figure()
    plt.plot(original)
    plt.plot(predicted)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()