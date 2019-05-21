from utils import *
import os, datetime, sys

original = None
today = datetime.datetime.now()
results_dir = "results_" + str(today.year) + "_" + str(today.month) + "_" + str(today.day) + "__" \
              + str(today.hour) + "_" + str(today.minute)

save_plots = False


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

    print("Solving auxiliary problem...")
    x_0, missing_intervals = solve_auxiliary_problem(interpolated_values, actual_values)
    print("Starting point found.")
    q_max = max_neighborhood_tentatives
    best_opt = x_0
    best_opt_value = objective_function(best_opt, moment, autocorrelation, x_0)
    bounds = create_bounds(actual_values, keep_feasible, missing_intervals)
    upper_b = np.nanmax(actual_values)
    lower_b = np.nanmin(actual_values)

    fname = os.path.join(results_dir, str("starting_solution.png")) if save_plots is True else None
    plot(actual_values[200:800], x_0[200:800], str("Starting solution"), filename=fname)

    i = 1
    q = 1
    print("Proceeding with the iteration loop.")
    while i < max_iteration:
        local_opt = shaking(best_opt, missing_intervals, upper_b, lower_b, i)  # Perturb solution within his range
        local_opt, local_opt_value = local_search(local_opt, moment, autocorrelation, x_0, bounds)
        print("Iteration # %3d/%3d of %3d: value %5.8f \n" % (i, q, max_iteration, local_opt_value))

        fname = os.path.join(results_dir, str("iteration_" + str(i)+".png")) if save_plots is True else None
        plot(actual_values[200:800], local_opt[200:800], str("Iteration # " + str(i)), filename=fname)

        if local_opt_value < best_opt_value:
            best_opt = local_opt
            best_opt_value = local_opt_value
            q = 1
        else:
            q += 1
            print("Perturbing solution..")
        if q > q_max:
            q = 1

        i += 1

    return best_opt


def local_search(candidate, moment, autocorrelation, starting_point, bounds):
    result = optimize.minimize(objective_function, candidate, args=(moment, autocorrelation, starting_point),
                               method="L-BFGS-B", bounds=bounds, options={'maxiter': 10000})
    # Return an optimization object
    if result.success is False:
        print(result.message)

    return result.x, result.fun


def objective_function(x, target_moment, target_autocorrelation, p_0):
    m = target_moment
    p = target_autocorrelation
    lambda_coeff = np.full(m.shape[0], 5000)
    mu_coeff = np.full(p.shape[0], 4000)
    k = len(lambda_coeff)
    j = len(mu_coeff)
    mean = np.mean(x)

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
        for interval in missing_intervals:  # fixme optimization
            summation += (interval[1] - interval[0] + 1) * (y[interval[0] - 1] + y[interval[1] + 1])
        return summation

    def lambda_denominator(r):  # a simplified version of tau_n
        return (1/3 * r*r*r - 2 * r*r - 19/3 * r - 2)/(4*N - 2*L2/N - L3/N)

    N = x.size
    c = N * np.mean(x) - sum_of_known_values()
    lambda_numerator = 2 * c - sum_lambda()

    for interval in missing_intervals:  # Computes the interpolated values for each interval
        interval_length = interval[1] - interval[0] + 1
        interpolated_values = np.empty(interval_length)
        r = interval_length
        p = interval[0]

        lambda_coeff = lambda_numerator / lambda_denominator(r)
        print("Lambda: " + str(lambda_coeff))

        if interval_length >= 3:  # Case 1.a & b: an interval, with more than 3 missing values
            interpolated_values[0] = (r + 1)/(r + 2) * y[p - 1] - 1/(r + 2) * y[p + r] + \
                                     (- r*r - 3*r - 2)/(4*N*(r + 2)) * lambda_coeff
            x[p] = interpolated_values[0]
            for j in range(1, interval_length - 1):
                interpolated_values[j] = (j + 1) * x[p] - j*y[p - 1] + j*(j + 1)/(4*N) * lambda_coeff

            interpolated_values[r - 1] = r/2 * x[p] + (r - 1)/2*y[p-1] - 1/2 * y[p + r] + (r*r - r - 2)/(8*N) * lambda_coeff

        if interval_length == 2:  # Case 2.a & b: a 2-value interval
            interpolated_values[0] = 2/3 * y[p - 1] + 1/3 * y[p + 2] - 1/(2*N) * lambda_coeff
            interpolated_values[1] = 2/3 * y[p + 2] + 1/3 * y[p - 1] - 1/(2*N) * lambda_coeff

        if interval_length == 1:  # Case 2.c & d: an isolated missing value
            interpolated_values[0] = 1/2 * y[p - 1] + 1/2 * y[p + 1] - 1/(4*N) * lambda_coeff

        #  Insert the new values into the x vector
        for offset in range(len(interpolated_values)):
            x[p + offset] = interpolated_values[offset]

    # Finally, return the whole vector
    return x, missing_intervals


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "save-plots":
            os.mkdir(results_dir)
            save_plots = True
            print("I'm gonna save plots on a dedicated folder")

    num_of_samples = 3000
    y = np.empty(num_of_samples)

    with open('coke.csv', 'rb') as file:
        for i in range(num_of_samples):
            row = file.readline()
            y[i] = float(row)
            i += 1
            if i == num_of_samples:
                break

    x = np.copy(y)
    original = np.copy(x)
    x_mean = np.mean(x)

    momento_k = np.empty(3)
    momento_k[0] = moments(x, 1)
    momento_k[1] = moments(x, 2)
    momento_k[2] = moments(x, 3)


    autocorr_ = np.empty(12)
    for i in range(autocorr_.size):
        autocorr_[i] = autocorrelation(x, i, 3, x_mean)

    missing_interval = (500, 600)

    for i in range(missing_interval[0], missing_interval[1]):
        x[i] = y[missing_interval[0] - 1]
        y[i] = None

    print("\n=== Starting script ====")
    print("\nNumber of samples avaiable: " + str(num_of_samples))

    try:
        solution = moment_matching_method(y, moment=momento_k, autocorrelation=autocorr_, interpolated_values=x, keep_feasible=False)
    except Exception as ex:
        print("Interrupted")

    print("The method has finished.")

    filename = os.path.join(results_dir, "final_solution.png") if save_plots is True else None
    plot(original,solution, str("Final solution"), filename=filename)

    with open(os.path.join(results_dir, "solution.csv"), 'w') as file:
        for i in range(num_of_samples):
            file.write(str(str(solution[i]) + "\n"))

    print("Done.")


if __name__ == "__main__":
    main()