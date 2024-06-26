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

What is the prior, exactly?

### Option 1: Informative prior

From experts or from your data

what kind of rates have you seen previously? are any excluded by the "laws of physics"? subjective elicitation

link to MATCH here

http://optics.eee.nottingham.ac.uk/match/uncertainty.php

you should err on the side of being more open-minded, ie a flatter prior

appendix if you want to roll your own

### Option 2: Flat priors

|Prior         |Name    |Notes|
|--------------|--------|-|
|Beta(1, 1)    |Uniform |-|
|Beta(1/2, 1/2)|Jeffreys|-|
|Beta(1/3, 1/3)|Neutral |-|
|Beta(0, 0)    |Haldane |-|

https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-5/issue-none/Neutral-noninformative-and-informative-conjugate-beta-and-gamma-prior-distributions/10.1214/11-EJS648.full

### Which one should I pick?

> it can notoriously difficult to choose among noninformative priors; and, even more importantly, seemingly noninformative distributions can sometimes have strong and undesirable implications, as I have found in my own experience (Gelman, 1996, 2006). As a result I have become a convert to the cause of weakly informative priors, which attempt to let the data speak while being strong enough to exclude various “unphysical” possibilities which, if not blocked, can take over a posterior distribution in settings with sparse data—a situation which is increasingly present as we continue to develop the techniques of working with complex hierarchical and nonparametric models.

~[Andrew Gelman, _Bayes, Jeffreys, Prior Distributions and the Philosophy of Statistics_](http://www.stat.columbia.edu/~gelman/research/published/jeffreys.pdf)

imo, the right choice is a prior which is very weakly informative. the prior is a constraint on the inference; some constraints are justified from first principles. for example, the minimum and maximum values are often known beforehand. and uninformative priors have weird properties when the sample size is small, ie 50/50 median

# An example analysis: Statistical quality control

The usual examples here are coin flipping and a standard RCT or A/B test; they have been done very thoroughly elsewhere and so I'll skip them. See the puppy book if you want a good account of those.

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