What will my "real bad day" look like? How much do I need to keep in reserve to stay safe in that case?

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html

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
  
datasets = [gen_data(1000) for _ in range(1000)]

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

2. This other thing

https://stats.stackexchange.com/questions/99829/how-to-obtain-a-confidence-interval-for-a-percentile

3. Easy mode

Cookbook estimate: http://www.tqmp.org/RegularArticles/vol10-2/p107/p107.pdf
