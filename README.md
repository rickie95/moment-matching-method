# moment-matching-method
A Python 3 + Scipy implementation of Carrizosa et al. work [1].

The method is a variant of Variable Neighborhood Search for time series interpolation: performs a local search then selects a neighbor of the current solution, if the new point is better than the old one then continues searching with the newest. Otherwise selects another neighborhood and so on.
It's provided an objective function to be optimized: statistic moments and an autocorrelation term are included in the O.F. 

The method use a simplified problem in order to achieve a "good" starting point.

```python
while i < i_max:
        # 1) Shaking Step, generate a random x from the i-th neighborhood of x_0.
        x = get_neighboor( x_0 , neighborhoods[i])
        # 2) Local Search, use a local method (like L-BFGS-B) starting from x and reach x_local, a local optimum
        x_local = local_search(x)
        # 3) Neighborhood change: if local optimum is better than the actual one, then
        if f(x_local) < f(x):
            x_0 = x_local
            q = 1
        else:
            q = q + 1
            
        if q > q_max:
            q = 1
            i = i + 1
```

[1] <b>Time series interpolation via global optimization of moments fitting</b>. Emilio Carrizosa, Alba V. Olivares-Nadal, Pepa Ramírez-Cobo. <i>European Journal of Operational Research</i>. Elsevier. 1 October 2013    [Link](https://www.sciencedirect.com/science/article/abs/pii/S0377221713003068)
