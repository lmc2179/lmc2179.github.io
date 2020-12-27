---
layout: post
title: "Confidence intervals for ratios with the Jackknife"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*Practical questions often revolve around ratios of interest - open rates, costs per impression, percentage increases - but the statistics of ratios is more complex than you might realize. The sample ratio is biased, and its standard error is surprisingly hard to pin down. We'll see that despite this, we can use the jackknife (a predecessor of the better-known bootstrap) to handle both of these problems.*

# Ratios are everywhere

Something that might surprise students of statistics embarking on their first job is that quite a lot of practical questions are not framed in terms of the difference, $X - Y$, but rather the ratio, $\frac{X}{Y}$. Despite the fact that it seems very natural to ask questions about relative changes, a lot of initial statistics education focuses on the difference because it is easier to deal with. It is easy to find the standard error of $X - Y$ if we know the standard errors of $X$ and $Y$ and their correlation; we simply use the fact that variances add in this situation, perhaps with a covariance term. If you attempt to find an explanation of the standard error of $\frac{X}{Y}$ though, you suddenly encounter [a bewildering amount of calculus](http://www.stat.cmu.edu/~hseltman/files/ratio.pdf), and a stomach-churning number of Taylor expansions. That's unfortunate, because in my work I see ratios all the time, like:

- Open rates: $\text{Open rate} = \frac{\text{Opened}}{\text{Sent}}$
- Revenue per action: $\text{Revenue per action} = \frac{\text{Total Revenue received}}{\text{Total Actions performed}}$
- Cost per impression: $\text{Cost per impression} = \frac{\text{Total spend}}{\text{Impression count}}$
- Percent increase: $\text{Lift} = \frac{\text{Total new metric}}{\text{Total old metric}}$

The reason for this is that ... .

Introduce a synthetic example, Pareto/Binomial

```
from scipy.stats import binom, pareto, sem
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

numerator_dist = pareto(2)
denominator_dist = binom(2*10, 0.1)

def gen_data(n):
  return numerator_dist.rvs(n), denominator_dist.rvs(n)

def naive_estimate(n, d):
  return np.sum(n) / np.sum(d)

def jackknife_estimate(n, d):
  pass

naive_sampling_distribution_n_5 = [naive_estimate(n, d) for n, d in [gen_data(5) for _ in range(1000)]] # This is biased
jackknife_sampling_distribution_n_5 = [naive_estimate(n, d) for n, d in [gen_data(5) for _ in range(10000)]] # This is not (?)
```

# But the "obvious" ratio estimate is biased, and its standard errors can be tricky

The naive estimator is biased though this is less of an issue with large sample sizes

The variance of a corrected estimator is not obvious

The "cookbook" taylor series solution produces good/bad estimates (?)

# Detour: The Jackknife as a method for correcting bias and computing standard errors

The idea behind the jackknife standard error

the idea behind the jackknife standard error

formulas

this is an early bootstrap

the jackknife is conservative

# Putting it together: Ratio analysis with the jackknife

formula

code example

show it's not biased and the variance is right

you can even do it in SQL - https://www.db-fiddle.com/f/vtVZzMKdNsDpQH9G3L7XwL/3

# Appendix: Some other approaches

Taylor series/delta method

Fieller?

Percentile bootstrap and Bayesian bootstrap

# More reading

Q's paper, review paper

Blog post

CASI

# Material

Naive estimate is biased https://en.wikipedia.org/wiki/Ratio_estimator
The sampling distribution can be nuts: https://en.wikipedia.org/wiki/Ratio_distribution

Probably the easiest thing for both bias correction and confidence intervals is the jackknife, and you can do it in SQL

- Open rates
- Cost per item
- Percentage difference between matched pairs

Do some simulations for an example with a funny-looking sampling distribution; show that the naive estimate is biased; jackknife estimates the variance correctly

Uncorrelated parto/nbinom distribution - revenue per...session?

$\text{test}$

```python
from scipy.stats import binom, pareto

numerator_dist = pareto(2)
denominator_dist = binom(2*10, 0.1)

def gen_data(n):
  return numerator_dist.rvs(n), denominator_dist.rvs(n)

def naive_estimate(n, d):
  return np.sum(n) / np.sum(d)
  
def jackknife_estimate(n, d):
  pass

naive_sampling_distribution_n_5 = [naive_estimate(n, d) for n, d in [gen_data(5) for _ in range(10000)]] # This is biased
jackknife_sampling_distribution_n_5 = [naive_estimate(n, d) for n, d in [gen_data(5) for _ in range(10000)]] # This is not (?)

```

# Other stuff

http://www.stat.cmu.edu/~hseltman/files/ratio.pdf

https://arxiv.org/pdf/0710.2024.pdf

https://stats.stackexchange.com/questions/16349/how-to-compute-the-confidence-interval-of-the-ratio-of-two-normal-means

https://i.stack.imgur.com/vO8Ip.png

https://statisticaloddsandends.wordpress.com/2019/02/20/what-is-the-jackknife/amp/

http://www.math.ntu.edu.tw/~hchen/teaching/LargeSample/references/Miller74jackknife.pdf

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1015.9344&rep=rep1&type=pdf

https://www.researchgate.net/publication/220136520_Bootstrap_confidence_intervals_for_ratios_of_expectations - Bootstrap confidence intervals for ratios of expectations

```python
# Fieller vs Taylor method; assume independence
from scipy.stats import norm, sem
import numpy as np

n_sim = 1000

m_x = 1000
m_y = 1500

s_x = 1900
s_y = 1900

n_x = 100
n_y = 100

true_ratio = m_y / m_x

total_success_taylor = 0
total_success_fieller = 0

z = 1.96

for i in range(n_sim):
  x = np.clip(norm(m_x, s_x).rvs(n_x), 0, np.inf)
  y = np.clip(norm(m_y, s_y).rvs(n_y), 0, np.inf)
  
  point_ratio = np.mean(y) / np.mean(x)
  se_ratio = np.sqrt(((sem(x)**2 / np.mean(x)**2)) + (sem(y)**2 / np.mean(y)**2))

  l_taylor, r_taylor = point_ratio - z * se_ratio, point_ratio + z * se_ratio
  total_success_taylor += int(l_taylor < true_ratio < r_taylor)
  
  #t_unbounded = (np.mean(x)**2 / sem(x)**2) + (((np.mean(y) * sem(x)**2))**2 / (sem(x)**2 * sem(x)**2 * sem(y)**2))
  
  fieller_num_right = np.sqrt((np.mean(x)*np.mean(y))**2 - (np.mean(x)**2 - z*sem(x)) - (np.mean(y) - z*sem(y)**2))
  fieller_num_left = np.mean(x)*np.mean(y)
  fieller_denom = np.mean(x)**2 - z * sem(x)**2
  l_fieller = (fieller_num_left - fieller_num_right) / fieller_denom
  r_fieller = (fieller_num_left + fieller_num_right) / fieller_denom
  total_success_fieller += int(l_fieller < true_ratio < r_fieller)
print(1.*total_success_taylor / n_sim)
print(1.*total_success_fieller / n_sim)
```
