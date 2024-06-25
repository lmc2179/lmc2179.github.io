# Getting started with Bayesian inference: The beta-binomial model

The bayesian update process is intuitively appealing

* What do we think about the rate before seeing the data?
* What do we think about the rate after seeing the data?
* What might the next observation look like?

A qualitative example: an experiment

Bayes rule gives us the mechanics of the update

### Prior

Examples of the beta distribution - it's a handy way of summarizing our prior knowledge because it's flexible. plus other reasons

### Posterior and calculating an update

Conjugate prior - wiki

Why does this work? Basically the algebra works out to look like another beta

### Posterior-predictive

# What prior should you pick?

### Option 1: From experts or from your data

what kind of rates have you seen previously? are any excluded by the "laws of physics"? subjective elicitation

link to MATCH here

http://optics.eee.nottingham.ac.uk/match/uncertainty.php

### Option 2: Flat priors

|Prior         |Name    |Notes|
|--------------|--------|-|
|Beta(1, 1)    |Uniform |-|
|Beta(1/2, 1/2)|Jeffreys|-|
|Beta(1/3, 1/3)|Neutral |-|
|Beta(0, 0)    |Haldane |-|

https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-5/issue-none/Neutral-noninformative-and-informative-conjugate-beta-and-gamma-prior-distributions/10.1214/11-EJS648.full

# Posterior analysis of an experiment

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

# Appendix: Fit your own prior to quantiles

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta

# Step 1: Specify your desired quantiles and their corresponding probabilities
desired_quantiles = [0.025, 0.05, .3]  # Example: 25th and 75th percentiles
probabilities = [.25, 0.5, .75]        # Example: probabilities for those quantiles

# Step 2: Define the objective function
def objective(params):
    a, b = params
    quantiles = beta.ppf(probabilities, a, b)
    return np.sum((quantiles - desired_quantiles)**2)

# Step 3: Use an optimization algorithm to find the best alpha and beta
initial_guess = [1, 1]
result = minimize(objective, initial_guess, bounds=[(0.01, None), (0.01, None)])

# Extract the optimized parameters
alpha, beta_ = result.x

# Print the optimized parameters
print(f"Optimized alpha: {alpha}")
print(f"Optimized beta: {beta_}")

# Verify the quantiles
quantiles = beta.ppf(probabilities, alpha, beta_)
print(f"Quantiles at specified probabilities: {quantiles}")
print(f"Desired quantiles: {desired_quantiles}")

```