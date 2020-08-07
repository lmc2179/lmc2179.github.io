---
layout: post
title: "An alternative to the FWER for multiple comparisons: The False-Discovery Rate"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*We've [previously explored](https://lmc2179.github.io/posts/fwer.html) one common method for dealing with testing multiple simultaneous hypotheses, methods that control the Family-wise error rate. However, we realized that the FWER can be quite conservative. The False-Discovery rate is a powerful alternative to the FWER, which is often used in cases where hundreds or thousands of simultaneous hypotheses are tested. What is the FDR, and how is it different from the FWER? How do we control it? What price do we pay for doing so*

# What happens to the FWER when we test hundreds or thousands of hypotheses?

# The goal of FDR control: Make sure few of your findings are spurious

https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf, ch.15, epage 289

That matrix that shows up everywhere

# FDR control when the hypotheses are independent: Benjamini-Hochberg

http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf

# FDR control under arbitrary dependence: Benjamini–Yekutieli

However the power is lower
http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_yekutieli_ANNSTAT2001.pdf

# An example with independence

```python
from sklearn.datasets import make_regression
from statsmodels.api import OLS
import numpy as np
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

def simulate_fdr():
  n_tests = 1000
  n_sample = 5000
  n_true_pos = 100

  #count_discoveries = 0
  #count_false_discoveries = 0
  #count_discoveries_bh = 0
  #count_false_discoveries_bh = 0

  alpha = .05

  X, y, coef = make_regression(n_samples=n_sample, n_features=n_tests, n_informative=n_true_pos, bias=0, coef=True, noise=1)
  p = OLS(y, X).fit().pvalues
  reject = p <= alpha
  true_null = (coef == 0)
  count_discoveries = np.sum(reject)
  count_false_discoveries = np.sum(reject & true_null)

  reject_bh = multipletests(p, method='fdr_bh', alpha=alpha)[0]
  count_discoveries_bh = np.sum(reject_bh)
  count_false_discoveries_bh = np.sum(reject_bh & true_null)

  naive_fdr = count_false_discoveries / count_discoveries
  bh_fdr = count_false_discoveries_bh / count_discoveries_bh
  return naive_fdr, bh_fdr
  
simulations = [simulate_fdr() for _ in tqdm(range(1000))]

naive_fdr_dist, bh_fdr_dist = zip(*simulations)

sns.distplot(naive_fdr_dist, label='Naive method')
sns.distplot(bh_fdr_dist, label='BH method')
plt.legend()
plt.title('FDR distribution from 1000 simulations')
plt.axvline(.05, linestyle='dotted')
plt.show()
```

# An example with dependence: Pairwise comparisons

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html

```python
from scipy.stats import ttest_ind
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

#TRUE_MU = [0] * 40 + list(range(1, 20))
TRUE_MU = [0]*60 + [1]
k = len(TRUE_MU)
TRUE_SIGMA = 2

alpha = .05
n_sample = 10000

count_discoveries = 0
count_false_discoveries = 0

test_indices = [(i, j) for i in range(0, k) for j in range(i+1, k)]

X = [norm(m, TRUE_SIGMA).rvs(n_sample) for m in TRUE_MU]
p = np.array([ttest_ind(X[i], X[j])[1] for i, j in test_indices])

true_null = np.array([TRUE_MU[i] == TRUE_MU[j]  for i, j in test_indices])
reject = (p <= alpha)

count_discoveries = sum(reject)
count_false_discoveries = sum(reject & true_null)

reject_bh = multipletests(p, method='fdr_bh', alpha=alpha)[0]
count_discoveries_bh = np.sum(reject_bh)
count_false_discoveries_bh = np.sum(reject_bh & true_null)

reject_by = multipletests(p, method='fdr_by', alpha=alpha)[0]
count_discoveries_by = np.sum(reject_bh)
count_false_discoveries_by = np.sum(reject_by & true_null)

print(count_false_discoveries / count_discoveries)
print(count_false_discoveries_bh / count_discoveries_bh)
print(count_false_discoveries_by / count_discoveries_by)
```
