Simple methods for sampling

# How do we sample from a distribution? Why would we want to?

We often know a distribution analytically, but can't sample from it. Sampling from a distribution allows us to perform numerical analysis of many of its properties. Most commonly, we are interested in its integrals or moments, which may be hard to acquire analytically. This kind of problem shows up all the time in Bayesian inference, in which we might look derive the posterior distribution for our favorite parameters, but it may have an unpleasant analytical form.

MCMC provides a general (but costly) solution. For simpler, especially lower dimensional distributions, there are some less complex methods we can use. It's useful to have these at your disposal when MCMC is overkill, and can be quick solutions to simple sampling problems.

# A running example: an asymmetric, unimodal, unnormalized density function

$f(X) = x (1-x)^3$

# Sampling from a one-dimensional distribution: Inverse transform sampling of the CDF

https://en.wikipedia.org/wiki/Inverse_transform_sampling

- Compute the CDF, $F$
- Invert it to get $F^{-1}$
- Sample $u$ uniformly on the unit interval
- Compute $F^{-1}(u)$

## An analytical solution with Sympy

```python
from sympy import symbols, integrate, latex
x, a, b = symbols('x a b')
f = (x**(a-1))*((1-x)**(b-1))
F = integrate(f, (x, 0, 1), meijerg=True)
const = F.evalf(subs={a:2, b:4}) # normalizing constant; don't use meijerg if x is the only variable
print(latex(F))
```

## A numerical solution with Scipy

# Low-dimensional dimensions: Grid sampling

- Select grid bounds and resolution
- Evaluate $f$ over the grid
- Weighted sample according to $f$

```python
import numpy as np
from scipy.stats import norm 
from matplotlib import pyplot as plt
import seaborn as sns

x_plot = np.linspace(-10, 10, 5000)
log_p = norm(0, 1).logpdf(x_plot) + 500
p_sample = np.exp(log_p - np.logaddexp.reduce(log_p))
sns.distplot(np.random.choice(x_plot, p=p_sample, size=10000))
plt.plot(x_plot, norm(0, 1).pdf(x_plot))
plt.show()
```

# Unimodal distributions, however many dimensions they have: Laplace's approximation

- Find maximum
- Compute inverse hessian
- Construct normal approximation


## An interesting connection between the Bayesian and Frequentist worlds: Laplace and the Fisher information

# Comparing these methods

|Method|Dimensionality|Summary|
|-|-|-|
|Inverse transform sampling|1D only|Invert the CDF and sample from the uniform distribution on [0, 1]|
|Grid sampling|Low|Compute a subset of PDF values, and treat the distribution as discrete|
|Laplace approximation|Any, as long as it's unimodal|Find the maximum, and fit a normal approximation around it|
