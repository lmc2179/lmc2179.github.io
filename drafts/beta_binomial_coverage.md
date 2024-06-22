# Getting started with Bayesian inference: The beta-binomial model

* What do we think about the rate before seeing the data?
* What do we think about the rate after seeing the data?
* What might the next observation look like?

### Prior

Examples of the beta distribution - it's a handy way of summarizing our prior knowledge because it's flexible. plus other reasons

### Posterior and calculating an update

Conjugate prior - wiki

Why does this work? Basically the algebra works out to look like another beta

### Posterior-predictive

# What prior should you pick?

### Option 1: From experts or from your data

### Option 2: Flat prior

https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-5/issue-none/Neutral-noninformative-and-informative-conjugate-beta-and-gamma-prior-distributions/10.1214/11-EJS648.full

# An example experiment analysis

formulate prior

Observe data

# Appendix: Comparing coverage

```python
import numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

def sim_coverage(a, b, n_sim, k, rate, confidence_level):
  x = np.random.binomial(k, rate, size=n_sim)
  posteriors = beta(x+a, (k-x)+b)
  lower, upper = posteriors.interval(confidence_level)
  lower, upper = np.array(lower), np.array(upper)
  covered = (lower <= rate) & (upper >= rate)
  lower_error = rate - lower
  upper_error = rate - upper
  return np.mean(covered), np.mean(lower_error), np.mean(upper_error)


def sim_coverage_freq(a, b, n_sim, k, rate, confidence_level, method='normal'):
  x = np.random.binomial(k, rate, size=n_sim)
  lower, upper = zip(*[proportion_confint(s, k, 1.-confidence_level, method=method) for s in x])
  lower, upper = np.array(lower), np.array(upper)
  covered = (lower <= rate) & (upper >= rate)
  lower_error = rate - lower
  upper_error = rate - upper
  return np.mean(covered), np.mean(lower_error), np.mean(upper_error)

a = 1.
b = 1.
n_sim = 100
confidence_level = 0.95

k_values = np.arange(1, 101, 9)
rates = np.linspace(0, 1, 20)

test_values = [(k, r) for r in rates for k in k_values]

coverages, lower_error, upper_error = zip(*[sim_coverage(a, b, n_sim, k, r, confidence_level) for k, r in tqdm(test_values)])
#coverages, lower_error, upper_error = zip(*[sim_coverage_freq(a, b, n_sim, k, r, confidence_level, method='agresti_coull') for k, r in tqdm(test_values)])
coverages, lower_error, upper_error = np.array(coverages), np.array(lower_error), np.array(upper_error)
k_plot, r_plot = zip(*test_values)
k_plot, r_plot = np.array(k_plot), np.array(r_plot)

plt.tricontourf(r_plot, k_plot, coverages, levels=np.round(np.linspace(0, 1, 20), 2))
plt.colorbar()
plt.show()


plt.tricontourf(r_plot, k_plot, lower_error)
plt.colorbar()
plt.show()

plt.tricontourf(r_plot, k_plot, upper_error)
plt.colorbar()
plt.show()

sns.regplot(r_plot, coverages, lowess=True)
plt.show()

sns.regplot(k_plot, coverages, lowess=True)
plt.show()
```
