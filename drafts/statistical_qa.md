Finite population correction

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import sem

x = np.array([1]*3000 + [0]*3000)

n_population = len(x)

n_sample = 1000

n_sim = 10000

fpc = np.sqrt((n_population - n_sample) / (n_population - 1))

samples = [np.random.choice(x, size=n_sample, replace=False) for _ in range(n_sim)]
sampled_means = [np.mean(s) for s in samples]
sampled_sems = [sem(s) for s in samples]
sampled_corrected_sems = [sem(s) * fpc for s in samples]

actual_sem = np.std(sampled_means)

sns.distplot(sampled_sems)
sns.distplot(sampled_corrected_sems)
plt.axvline(actual_sem)
plt.show()

sns.distplot(sampled_sems - actual_sem)
sns.distplot(sampled_corrected_sems - actual_sem)
plt.axvline(np.mean(sampled_corrected_sems - actual_sem))
plt.axvline(np.mean(sampled_sems - actual_sem))
plt.show()

```

Plot (n / N) vs standard error inflation (Inverse of the correction factor?)
Swap X with a different distribution
Is this exact for the binomial distribution!? Note that the correct SE is correct event for tiny sample sizes

Bayesian view: Predictive posterior for the unobserved values for binomial (exact I think), else bootstrap unknown values
