---
layout: post
title: "Speed up your analysis of giant datasets with sufficient statistics"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

Applying 

# Lots of rows means lots of waiting

In the 

Sometimes it's more convenient to work with summaries of subgroups in the data than with the raw data itself

- Combining multiple datasets, as in [Meta-analysis](https://en.wikipedia.org/wiki/Meta-analysis)
- A large number of datapoints but a small number of parameters
- Observational analysis from data matched in strata

# Sufficient statistics

https://en.wikipedia.org/wiki/Sufficient_statistic
https://web.ma.utexas.edu/users/gordanz/notes/likelihood_color.pdf
# Binomial outcomes: Easy with statsmodels

https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html

# Continuous outcomes: Some assembly required

WLS implementation in statsmodels doesn't work here

## Digression: Sufficient statistic for the normal distribution

https://en.wikipedia.org/wiki/Sufficient_statistic#Normal_distribution

$$\mathcal{L}(\mu, \sigma | \bar{y}, s^2, n) = (2 \pi \sigma^2)^{\frac{n}{2}} exp \left(\frac{n - 1}{2 \sigma^2} s^2 \right ) exp \left (-\frac{n}{2 \sigma^2} (\mu - \bar{y})^2 \right)$$

$$ln \mathcal{L}(\mu, \sigma | \bar{y}, s^2, n) = -\frac{n}{2} ln(2 \pi \sigma^2) - \left( \frac{n-1}{2 \sigma^2}s^2 - \frac{n}{2\sigma^2} (\mu - \bar{y})^2 \right) $$

```python
import numpy as np
from scipy.optimize import minimize

def logpdf_sufficient(mu, sigma_sq, sample_mean, sample_var, n):
  return -(n/2) * np.log(2*np.pi*sigma_sq) - (((n-1) / (2*sigma_sq)) * sample_var) - ((n / (2*sigma_sq)) * (mu - sample_mean)**2) 

data = np.random.normal(0, 1, 1000)
n = len(data)
m, v = np.mean(data), np.var(data)

def neg_log_likelihood(p):
  return -logpdf_sufficient(p[0], np.exp(p[1]), m, v, n)

result = minimize(neg_log_likelihood, np.array([5, -1]))
cov = result.hess_inv

se_mean = np.sqrt(cov[0][0])
```
