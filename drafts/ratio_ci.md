# It's really easy to mess up estimates of ratios

Naive estimate is biased https://en.wikipedia.org/wiki/Ratio_estimator
The sampling distribution can be nuts: https://en.wikipedia.org/wiki/Ratio_distribution

Probably the easiest thing for both bias correction and confidence intervals is the jackknife, and you can do it in SQL

- Open rates
- Cost per item
- Percentage difference between matched pairs

Do some simulations for an example with a funny-looking sampling distribution; show that the naive estimate is biased; jackknife estimates the variance correctly

Uncorrelated parto/nbinom distribution - revenue per...session?

$\text{test}$

# Other stuff

http://www.stat.cmu.edu/~hseltman/files/ratio.pdf

https://arxiv.org/pdf/0710.2024.pdf

https://stats.stackexchange.com/questions/16349/how-to-compute-the-confidence-interval-of-the-ratio-of-two-normal-means

https://i.stack.imgur.com/vO8Ip.png

https://statisticaloddsandends.wordpress.com/2019/02/20/what-is-the-jackknife/amp/

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
