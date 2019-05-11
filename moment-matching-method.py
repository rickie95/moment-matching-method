from scipy import signal, stats, optimize, fftpack
from functools import reduce
import numpy as np
import csv
import random
import matplotlib.pyplot as plt


original = None

def compute_missing_value_intervals(values):
    values = np.argwhere(np.isnan(values)).reshape((1, -1))[0]
    intervals = []
    sub = []
    sub.append(values[0])
    for i in range(1, len(values)):
        if values[i] - values[i-1] > 1:
            sub.append(values[i-1])
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


def create_bounds(values, feasible):
    max_value = np.nanmax(values)
    min_value = np.nanmin(values)
    print("max " + str(max_value) + " min " + str(min_value))
    upper_bounds = np.empty(values.size)
    lower_bounds = np.empty(values.size)
    for i in range(values.size):
        if values[i] is not np.nan:
            upper_bounds[i] = values[i]
            lower_bounds[i] = values[i]
        else:
            upper_bounds[i] = max_value
            lower_bounds[i] = min_value
    return optimize.Bounds(lower_bounds, upper_bounds, feasible)


def moment_matching_method(actual_values, moment=None, autocorrelation=None, interpolated_values=None,
                           max_iteration=50, max_neighborhood_tentatives=2, keep_feasible=False):
    '''
    Requires:
        a time series (with missing values as None) as vector
        the interpolated version of the time se
        max iterations allowed (Default: 100)
        number of points generated for each neighborhood (Default: 2)
    :return: the optimum as vector
    '''

    x_0, missing_intervals = solve_auxiliary_problem(interpolated_values, actual_values)
    q_max = max_neighborhood_tentatives
    best_opt = x_0
    best_opt_value = objective_function(best_opt, moment, autocorrelation, x_0)
    bounds = create_bounds(y, keep_feasible)
    upper_b = np.nanmax(actual_values)
    lower_b = np.nanmin(actual_values)
    i = 1
    q = 1
    print("prewhile")
    while i < max_iteration:
        x = shaking(best_opt, missing_intervals, upper_b, lower_b) #, best_opt.min(), best_opt.max())  # Perturb solution within his range
        local_opt, local_opt_value = local_search(x, moment, autocorrelation, x_0, bounds)
        print("Iteration # %3d/%3d of %3d: value %5.5f \n" % (i,q, max_iteration, local_opt_value))
        plt.plot(x)
        plt.plot(original)
        plt.xlabel(str("Iterazione # "+ str(i)))
        plt.show()
        if local_opt_value < best_opt_value:
            best_opt = local_opt
            best_opt_value = local_opt_value
            q = 1
        else:
            q += 1

        if q > q_max:
            q = 1

        i += 1

    return best_opt


def shaking(serie, missing_intervs, upper, lower):  # , lower_bound, upper_bound):  # Return a time series pertubated in his null values
    for interval in missing_intervs:
        values = stats.uniform.rvs(loc=-(upper/lower), scale=2*(upper/lower), size=(interval[1] - interval[0]))
        serie[interval[0]:interval[1]:1] += values
    return serie


def local_search(candidate, moment, autocorrelation, starting_point, bounds):
    result = optimize.minimize(objective_function, candidate, args=(moment, autocorrelation, starting_point),
                               method="L-BFGS-B", bounds=bounds, options={'maxiter': 10000})  # Return an optimization object
    if result.success is False:
        print(result.message)

    w = fftpack.rfft(result.x[200:320])
    spectrum = w ** 2
    plt.plot(spectrum)
    plt.show()
    pepe = (spectrum.max() / 5)
    cutoff_idx = spectrum > pepe
    w2 = w.copy()
    #w2[cutoff_idx] = 0
    #result.x[200:320] = fftpack.irfft(w2)
    return result.x, result.fun


def objective_function(x, target_moment, target_autocorrelation, p_0):
    m = target_moment
    p = target_autocorrelation
    lambda_coeff = np.full(m.shape[0], 5000)
    mu_coeff = np.full(p.shape[0], 4000)
    k = len(lambda_coeff)
    j = len(mu_coeff)

    mean = np.mean(x)

    def moments(x, order):
        return np.sum(np.power(x, order))/x.shape[0]

    def autocorrelation(x, lag, k, mean):
        # [ x x x x 0 0 0 ] * [ 0 0 0 x x x x ]
        x_pad = np.append(x - mean, np.zeros(lag))
        x_lag = np.append(np.zeros(lag), x)
        return np.dot(x_pad, x_lag) / stats.moment(x_pad, k)

    def f(x):
        x_pre = x[1:]
        x_post = x[:x.shape[0]-1]
        return np.sum(np.power(x_pre - x_post, 2))

    def moments_term(x):
        res = 0
        for i in range(k):
            mms = moments(x, i + 1)
            res += lambda_coeff[i] * (mms / m[i] - 1)**2
        return res

    def autocorrelation_term(x, k, mean):
        res = 0
        for i in range(j):
            autocrr = autocorrelation(x, i, k, mean)
            res += mu_coeff[i] * ((autocrr / p[i] - 1)**2)
        return res

    result = f(x) / f(p_0) + moments_term(x) + autocorrelation_term(x, k, mean)
    return result


def solve_auxiliary_problem(x, y):
    # Return the starting point x_0. This problem has an analytical solution, see Appendix A
    # None values in y are considered missing values

    # First of all, missing intervals are retrieved

    missing_intervals, L1, L2, L3 = compute_missing_value_intervals(y)

    def sum_of_known_values():  # SUM {i in S} y_i
        return np.nansum(y)

    def sum_lambda():
        summation = 0
        for interval in missing_intervals: # fixme optimization
            summation += len(interval) * (y[interval[0] - 1] + y[interval[-1] + 1])
        return summation

    def lambda_denominator(r):  # a simplified version of tau_n
        return (1/3 * r*r*r - 2 * r*r - 19/3 * r - 2)/(4*N) - 2*L2/N - L3/N

    N = x.size
    c = N * np.mean(x) - sum_of_known_values()
    lambda_numerator = 2 * c - sum_lambda()

    for interval in missing_intervals:  # Computes the interpolated values for each interval
        interval_length = interval[1] - interval[0] + 1
        interpolated_values = np.empty(interval_length)
        r = interval_length
        p = interval[0]

        lambda_coeff = lambda_numerator / lambda_denominator(r)

        if interval_length >= 3:  # Case 1.a & b: an interval, with more than 3 missing values
            interpolated_values[0] = (r + 1)/(r + 2) * y[p - 1] - 1/(r + 2) * y[p + r] + \
                                     (- r*r - 3*r - 2)/(4*N*(r + 2)) * lambda_coeff
            x[p] = interpolated_values[0]
            for j in range(1, interval_length - 1):
                interpolated_values[j] = (j + 1) * x[p] - j*y[p - 1] + j*(j + 1)/(4*N) * lambda_coeff

            interpolated_values[r - 1] = r/2 * x[p] - (r - 1)/2 * y[p - 1] + 1/2 * y[p + r] + (r*r - r - 2)/(8*N) * lambda_coeff

        if interval_length == 2:  # Case 2.a & b: a 2-value interval
            interpolated_values[0] = 2/3 * y[p - 1] + 1/3 * y[p + 2] - 1/(2*N) * lambda_coeff
            interpolated_values[1] = 2/3 * y[p + 2] + 1/3 * y[p - 1] - 1/(2*N) * lambda_coeff

        if interval_length == 1:  # Case 2.c & d: an isolated missing value
            interpolated_values[0] = 1/2 * y[p - 1] + 1/2 * y[p + 1] - 1/(4*N) * lambda_coeff

        #  Insert the new values into the x vector
        for i in range(len(interpolated_values)):
            x[p + i] = interpolated_values[i]

    # Finally, return the whole vector
    return x, missing_intervals


def autocorrelation_pippo(x, lag, k, mean):
    res = 0
    for i in range(len(x) - lag):
        res += (x[i] - mean) * (x[i + lag] - mean)
    return res / stats.moment(x, k)


y = np.empty(500)
i = 0

with open('dati.csv', 'rb') as file:
    for i in range(500):
        row = file.readline()
        y[i] = float(row)
        i += 1
        if i == 500:
            break

x = np.copy(y)
original = np.copy(x)

x_mean = np.mean(x)


def moments_ll(x, order):
    return np.sum(np.power(x, order)) / x.shape[0]

def autocorrelation_(x, lag, k, mean):
    # [ x x x x 0 0 0 ] * [ 0 0 0 x x x x ]
    x_pad = np.append(x - mean, np.zeros(lag))
    x_lag = np.append(np.zeros(lag), x)
    return np.dot(x_pad, x_lag) / stats.moment(x_pad, k)


momento_k = np.empty(3)
momento_k[0] = moments_ll(x, 1)
momento_k[1] = moments_ll(x, 2)
momento_k[2] = moments_ll(x, 3)


autocorr_ = np.empty(3)
autocorr_[0] = autocorrelation_(x, 1, 3, x_mean)
autocorr_[1] = autocorrelation_(x, 2, 3, x_mean)
autocorr_[2] = autocorrelation_(x, 3, 3, x_mean)


for i in range(200, 320):
    x[i] = y[199]
    y[i] = None

plt.figure()

print("Partenza")
solution = moment_matching_method(y, moment=momento_k, autocorrelation=autocorr_, interpolated_values=x)

with open('soluzione.csv', 'w') as file:
    for i in range(500):
        file.write(str(str(solution[i]) + "\n"))

print("ok")


