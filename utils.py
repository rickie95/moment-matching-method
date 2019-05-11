import numpy as np
from scipy import optimize


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