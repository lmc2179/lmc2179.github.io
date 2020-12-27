# Ratios are everywhere

- Open rates: $\text{Open rate} = \frac{\text{Opened}}{\text{Sent}}$
- Revenue per action: $\text{Revenue per action} = \frac{\text{Total Revenue}}{\text{Total Actions}}$
- Cost per impression: $\text{Cost per impression} = \frac{\text{Total spend}}{\text{Impression count}}$
- Percent increase: $\text{Lift} = \frac{\text{Total new}}{\text{Total old}}$

Introduce a synthetic example, Pareto/Binomial

# But the "obvious" ratio estimate is biased, and standard errors can be tricky

The naive estimator is biased

The variance of a corrected estimator is not obvious

# Detour: The Jackknife as a method for correcting bias and computing standard errors

The idea behind the jackknife standard error

the idea behind the jackknife standard error

formulas

this is an early bootstrap

# Putting it together: Ratio analysis with the jackknife

formula

code example

show it's not biased and the variance is right

you can even do it in SQL - https://www.db-fiddle.com/f/vtVZzMKdNsDpQH9G3L7XwL/0

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
