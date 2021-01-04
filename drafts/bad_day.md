What will my "real bad day" look like? How much do I need to keep in reserve to stay safe in that case? Looking forward, what observations are unusually bad? What is the size of the bottom - top difference? How can I establish "normal bounds" so I can know when things are not normal?

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html

I. Quantiles of a sample

1. Jackknife and many flavors of Bootstrap

Noted as an example for the bootstrap in Shalizi's article

https://www.americanscientist.org/article/the-bootstrap

Which bootstrap works best? Is there a pretty way of writing the jackknife estimate

Bootstrap diagnostic - https://www.cs.cmu.edu/~atalwalk/boot_diag_kdd_final.pdf

```
from scipy.stats import norm, sem, skew, t, lognorm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats.mstats import mquantiles

TRUE_VALUE = lognorm(s=1).ppf(.01)

def gen_data(n):
  return lognorm(s=1).rvs(n)
  
datasets = [gen_data(1000) for _ in range(1000)] # what happens to each of these methods as we vary the sample size

def percentile_bootstrap_estimate(x, alpha, n_sim=2000):
  s = np.percentile(x, 1)
  boot_samples = [np.percentile(np.random.choice(x, len(x)), 1) for _ in range(n_sim)]
  q = 100* (alpha/2.)
  return x, np.percentile(boot_samples, q),np.percentile(boot_samples, 100.-q)
  
percentile_bootstrap_simulation_results = pd.DataFrame([percentile_bootstrap_estimate(x, .05) for x in tqdm(datasets)], columns=['point', 'lower', 'upper'])
percentile_bootstrap_simulation_results['covered'] = (percentile_bootstrap_simulation_results['lower'] < TRUE_VALUE) & (percentile_bootstrap_simulation_results['upper'] > TRUE_VALUE)
print(percentile_bootstrap_simulation_results.covered.mean())

def standard_bootstrap_estimate(x, alpha, n_sim=2000):
  s = np.percentile(x, 1)
  k = len(x)
  boot_samples = [np.percentile(np.random.choice(x, len(x)), 1) for _ in range(n_sim)]
  se = np.std(boot_samples)
  t_val = t(k-1).interval(1.-alpha)[1]
  return s, s - t_val * se, s + t_val * se
  
standard_bootstrap_simulation_results = pd.DataFrame([standard_bootstrap_estimate(x, .05) for x in tqdm(datasets)], columns=['point', 'lower', 'upper'])
standard_bootstrap_simulation_results['covered'] = (standard_bootstrap_simulation_results['lower'] < TRUE_VALUE) & (standard_bootstrap_simulation_results['upper'] > TRUE_VALUE)
print(standard_bootstrap_simulation_results.covered.mean())

def bca_bootstrap_estimate(x, alpha, n_sim=2000):
  k = len(x)
  r = np.percentile(x, 1)
  r_i = (np.sum(x) - x)/(k-1)
  d_i = r_i - np.mean(r_i)
  boot_samples = [np.percentile(np.random.choice(x, len(x)), 1) for _ in range(n_sim)]
  p0 =  np.sum(boot_samples < r) / n_sim
  z0 = norm.ppf(p0)
  a = (1./6) * (np.sum(d_i**3) / (np.sum(d_i**2))**(3./2.))
  alpha_half = (alpha/2.)
  p_low = norm.cdf(z0 + ((z0 + norm.ppf(alpha_half)) / (1. - a*(z0 + norm.ppf(alpha_half)))))
  p_high = norm.cdf(z0 + ((z0 + norm.ppf(1.-alpha_half)) / (1. - a*(z0 + norm.ppf(1.-alpha_half)))))
  return r, np.percentile(boot_samples, p_low*100.), np.percentile(boot_samples, p_high*100.)
  
bca_bootstrap_simulation_results = pd.DataFrame([bca_bootstrap_estimate(x, .05) for x in tqdm(datasets)], columns=['point', 'lower', 'upper'])
bca_bootstrap_simulation_results['covered'] = (bca_bootstrap_simulation_results['lower'] < TRUE_VALUE) & (bca_bootstrap_simulation_results['upper'] > TRUE_VALUE)
print(bca_bootstrap_simulation_results.covered.mean())
```

2. Exact methods ("Binomial idea")

https://staff.math.su.se/hoehle/blog/2016/10/23/quantileCI.html

https://stats.stackexchange.com/questions/99829/how-to-obtain-a-confidence-interval-for-a-percentile

```
from scipy.stats import norm, binom, pareto

MU = 0
S = 1

gen_dist = norm(MU, S)

gen_dist = pareto(2)

n = 100

q = 0.95
alpha = .05

TRUE_QUANTILE = gen_dist.ppf(q)

l = int(binom(n, p=q).ppf(alpha/2))
u = int(binom(n, p=q).ppf(1.-alpha/2) + 1) # ???? Check against R's qbinom

n_sim = 10000
results = 0
lower_dist = []
upper_dist = []

for _ in range(n_sim):
  data = sorted(gen_dist.rvs(n))
  if data[l] <= TRUE_QUANTILE <= data[u]:
    results += 1
  lower_dist.append(data[l])
  upper_dist.append(data[u])
    
lower_dist = np.array(lower_dist)
upper_dist = np.array(upper_dist)
    
print(results / n_sim)
```

3. Easy mode - Asymptotic estimate

Cookbook estimate: http://www.tqmp.org/RegularArticles/vol10-2/p107/p107.pdf

Looks like https://stats.stackexchange.com/a/99833/29694 where we assume the data is normally distributed

Eq 25

4. Methods from Wasserman's all of nonparametric statistics

https://web.stanford.edu/class/ee378a/books/book2.pdf
Use ebook p. 25 to estimate upper and lower bounds on the CDF, then invert them at `q`.

Example 2.17 - 

3.7 Theorem

II. For when you want to relate covariates to quantiles - Conditional quantiles

5. Quantile regression

https://www.statsmodels.org/dev/examples/notebooks/generated/quantile_regression.html - Linear models of quantiles by picking a different loss function

https://jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf - An ML approach


III. This isn't magic

look out for small samples and edge cases
