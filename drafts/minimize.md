_?_

# We often want to tailor our model to the situation

We commonly want to fit a model that can't be expressed as a basis-expanded linear regression

Example of diminishing returns/dose response - satiation, Cat treats, marketing spend, organic growth;

Lots of relationships between two real variables go something like this: "As x increases, so does y. But the next x causes less increase than the last one"

The problem - I can't do `from statsmodels import gompertz_regression`

In this case, we can use scipy to find the best parameters and their SEs

# ?

```python
from scipy.stats import norm, sem
from scipy.optimize import minimize
import numpy as np

x = norm(0, 1).rvs(1000)

def neg_log_likelihood(param_vector):
  mu, log_sd = param_vector
  return -np.sum(norm(mu, np.exp(log_sd)).logpdf(x))
  
result = minimize(neg_log_likelihood, [10, 10])

print(np.sqrt(np.diag(result.hess_inv)), sem(x))
```

```python
from pydataset import data
from matplotlib import pyplot as plt
import seaborn as sns

trees = data('Sitka')

trees['Time'] -= trees['Time'].min() # Time starts at t=0
trees['Time'] /= trees['Time'].max() # And goes until t = 1

plt.scatter(trees['Time'], trees['size'])
plt.show()

def gompertz(a, b, c, t):
  return a * np.exp(-b*np.exp(-c*t))
  
plt.scatter(trees['Time'], trees['size'])

from scipy.stats import norm

def gompertz(a, b, c, t):
  return a * np.exp(-b*np.exp(-c*t))
  
def fit(t, y):
  def neg_log_likelihood(v):
    a, b, c, log_s = v
    expected_y = gompertz(a, b, c, t)
    l = norm(expected_y, np.exp(log_s)).logpdf(y)
    return -np.sum(l)
  return minimize(neg_log_likelihood, [1, 1, 1, 1])
  
result = fit(trees['Time'], trees['size'])

a_mle, b_mle, c_mle, log_s_mle = result.x
s_mle = np.exp(log_s_mle)

plt.scatter(trees['Time'], trees['size'])
t_plot = np.linspace(trees['Time'].min(), trees['Time'].max())
y_plot = gompertz(a_mle, b_mle, c_mle, t_plot)
low_pred = y_plot - 2 * s_mle
high_pred = y_plot + 2 * s_mle
plt.plot(t_plot, y_plot)
plt.fill_between(t_plot, low_pred, high_pred, alpha=.1)
plt.show()

print(result)
print('Standard errors', np.sqrt(np.diag(result.hess_inv)))
```

$y_t \sim N(ae^{-be^{-ct}}, \sigma)$
