---
layout: post
title: "How did my treatment affect the distribution of my outcomes? A/B testing with quantiles and their confidence intervals in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: distributional_effects.png
---

# Distributional effects are often overlooked but provide a deeper understanding

## The group averages and average treatment effect hide a lot of information

```python
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import poisson, skellam, nbinom, randint, geom

for dist in [poisson(100), skellam(1101, 1000), randint(0, 200), geom(1./100)]:
  plt.plot(np.arange(0, 400), dist.pmf(np.arange(0, 400)))
plt.xlim(0, 400)  
plt.ylabel('PMF')
plt.title('Four distributions with a mean of 100')
plt.show()
```

## What might we do differently if we thought about the distributional effects?

* If an experiment negatively affected some group, we can consider mitigation
* Do we want ot make dist shape a goal, ie min service level

# An example

```python
from scipy.stats import norm, sem
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from scipy.stats.mstats import mjci
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
```

```python
def z_a_over_2(alpha):
  return norm(0, 1).ppf(1.-alpha/2.)

plt.title('Distribution of revenue per customer')
sns.distplot(data_control, label='Control')
sns.distplot(data_treatment, label='Treatment')
plt.ylabel('Density')
plt.xlabel('Revenue ($)')
plt.legend()
plt.show()
```

```python
print(np.mean(data_treatment) - np.mean(data_control))
print(z_a_over_2(.05) * np.sqrt(sem(data_treatment)**2 + sem(data_control)**2))
```

```python
Q = np.linspace(0.05, .95, 20)

plt.boxplot(data_control, positions=[0], whis=[0, 100])
plt.boxplot(data_treatment, positions=[1], whis=[0, 100])
plt.xticks([0, 1], ['Control', 'Treatment'])
plt.title('Box and Whisker plot')
plt.show()
```

```python
plt.title('Quantiles of revenue per customer')
plt.xlabel('Quantile')
control_quantiles = np.quantile(data_control, Q)
treatment_quantiles = np.quantile(data_treatment, Q)
plt.plot(Q, control_quantiles, label='Control')
plt.plot(Q, treatment_quantiles, label='Treatment')
plt.legend()
plt.show()
```

```python
plt.title('Quantile difference (Treatment - Control)')
plt.xlabel('Quantile')
quantile_diff = treatment_quantiles - control_quantiles
control_se = mjci(data_control, Q)
treatment_se = mjci(data_treatment, Q)
diff_se = np.sqrt(control_se**2 + treatment_se**2)
diff_lower = quantile_diff - z_a_over_2(.05 / len(Q)) * diff_se
diff_upper = quantile_diff + z_a_over_2(.05 / len(Q)) * diff_se
plt.plot(Q, quantile_diff, color='orange')
plt.fill_between(Q, diff_lower, diff_upper, alpha=.5)
plt.axhline(0, linestyle='dashed', color='grey', alpha=.5)
plt.show()
```

# Outro: Other ideas and alternatives

* Hetereogeneous effect analysis/subgroup analysis: Possible introduction of mitigation strategy
* Conditional variance modeling: Also a way of understanding change in the shape (conditional kurtosis? don't think anyone ever does that)
* Many variables: Quantile regression is a good framework

# DGP

```python
sample_size = 1000
data_control = np.random.normal(0, 1, sample_size)**2
data_treatment = np.concatenate([np.random.normal(0, 0.01, round(sample_size/2)), np.random.normal(0, 2, round(sample_size/2))])**2
```
