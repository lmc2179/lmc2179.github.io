---
layout: post
title: "Confidence intervals for ratios with the Jackknife and Bootstrap"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*Practical questions often revolve around ratios of interest - open rates, costs per impression, percentage increases - but the statistics of ratios is more complex than you might realize. The sample ratio is biased, and its standard error is surprisingly hard to pin down. We'll see that despite this, we can use the bootstrap (or its older sibling, the jackknife) to handle both of these problems. Along the way, we'll learn a little about how these methods work and when they're useful.*

# Ratios are everywhere

Something that might surprise students of statistics embarking on their first job is that quite a lot of practical questions are not framed in terms of the difference, $X - Y$, but rather the ratio, $\frac{X}{Y}$. Despite the fact that it seems very natural to ask questions about relative changes, a lot of initial statistics education focuses on the difference because it is easier to deal with. It is easy to find the standard error of $X - Y$ if we know the standard errors of $X$ and $Y$ and their correlation; we simply use the fact that variances add in this situation, perhaps with a covariance term. If you attempt to find an explanation of the standard error of $\frac{X}{Y}$ though, you suddenly encounter [a bewildering amount of calculus](http://www.stat.cmu.edu/~hseltman/files/ratio.pdf), and a stomach-churning number of Taylor expansions. This eventually yields an approximation for the standard error, but it's not always clear when this applies. That lack of clarity is unfortunate, because in my work I see ratios all the time, like:

- Open rates: $\text{Open rate} = \frac{\text{Opened}}{\text{Sent}}$
- Revenue per action: $\text{Revenue per action} = \frac{\text{Total Revenue received}}{\text{Total Actions performed}}$
- Cost per impression: $\text{Cost per impression} = \frac{\text{Total spend}}{\text{Impression count}}$
- Percent increase: $\text{Lift} = \frac{\text{Total new metric}}{\text{Total old metric}}$

Synthetic advertising example: Cost per widget sold

```
from scipy.stats import binom, pareto, sem
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils import resample

cost_dist = pareto(2)
widget_dist = binom(2*10, 0.1)

def gen_data(n):
  a, b = cost_dist.rvs(n), widget_dist.rvs(n)
  while sum(b) == 0:
    a, b = cost_dist.rvs(n), widget_dist.rvs(n)
  return a, b

def naive_estimate(n, d):
  return np.sum(n) / np.sum(d)

def jackknife_estimate(n, d):
  total_n, total_d = np.sum(n), np.sum(d)
  k = len(n)
  r = naive_estimate(n, d)
  r_i = (total_n - n) / (total_d - d) 
  if np.isinf(np.mean(r_i)): # This happens when there is exactly one non-zero entry in the entire dataset
    print(n, d, r_i)
    return r
  return k * r - (k-1)*np.mean(r_i)
  
def corrected_estimate(n, d): # This fails when there are zeroes
  return naive_estimate(n, d) - (np.cov(n/d, d)) / np.mean(d)

def bootstrap_estimate(n, d, n_bootstrap=100):
  r = naive_estimate(n, d)
  bootstrap_bias = np.mean([naive_estimate(*resample(n, d)) for _ in range(n_bootstrap)]) - r
  if np.isinf(bootstrap_bias):
    bootstrap_bias = 0
  return r - bootstrap_bias

def jackknife_se(n, d):
  total_n, total_d = np.sum(n), np.sum(d)
  k = len(n)
  r_i = (total_n - n) / (total_d - d) 
  if np.any(np.isinf(r_i)): # This happens when there is exactly one non-zero entry in the entire dataset
    return 0
  return np.sqrt(((k-1)/k) * np.sum(np.power(r_i - np.mean(r_i), 2)))
  #return np.sqrt((k - 1) * np.var(r_i)) # Equivalent, and this is how it's written in some sources

def bootstrap_se(n, d, n_bootstrap=100):
  return np.std([naive_estimate(*resample(n, d)) for _ in range(n_bootstrap)])

datasets = [gen_data(5) for _ in range(10000)]

naive_sampling_distribution_n_5 = [naive_estimate(n, d) for n, d in datasets] # This is biased
jackknife_sampling_distribution_n_5 = [jackknife_estimate(n, d) for n, d in datasets] # This is less biased
bootstrap_sampling_distribution_n_5 = [bootstrap_estimate(n, d) for n, d in datasets] # This is least biased

sns.distplot(naive_sampling_distribution_n_5)
sns.distplot(jackknife_sampling_distribution_n_5)
sns.distplot(bootstrap_sampling_distribution_n_5)
plt.show()

jackknife_se_samples_n_5 = np.array([jackknife_se(n, d) for n, d in datasets]) # This is not the right SD, but maybe the coverage is correct?
bootstrap_se_samples_n_5 = np.array([bootstrap_se(n, d) for n, d in datasets]) # This is also not great, though it at least agrees with the above
```

Paired vs unpaired observations?

# But the "obvious" ratio estimate is biased, and its standard errors can be tricky

The naive estimator is biased though this is less of an issue with large sample sizes

The variance of a corrected estimator is not obvious

The "cookbook" taylor series/fieller solution produces good/bad estimates (?) is it asymptotic?

https://arxiv.org/pdf/0710.2024.pdf

# The Jackknife as a method for correcting bias and computing standard errors

The idea behind the jackknife standard error

the idea behind the jackknife standard error

formulas

this is an early bootstrap

the jackknife is conservative

# An even better method: Percentile bootstrap

the jackknife is a good first approximation

and doesn't deal with asymmetry correctly

Percentile bootstrap and Bayesian bootstrap

# Putting it together: Ratio analysis with the jackknife and the Bootstrap

formula

code example

show it's not biased and the variance is right

you can even do it in SQL - https://www.db-fiddle.com/f/vtVZzMKdNsDpQH9G3L7XwL/3

# Appendix: Some other approaches

Taylor series/delta method

Fieller?

BCa bootstrap

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
