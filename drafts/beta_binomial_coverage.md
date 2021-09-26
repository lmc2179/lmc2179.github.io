

```python
import numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt

def sim_coverage(a, b, n_sim, k, rate, confidence_level):
  x = np.random.binomial(k, rate, size=n_sim)
  posteriors = beta(x+a, (k-x)+b)
  lower, upper = posteriors.interval(confidence_level)
  lower, upper = np.array(lower), np.array(upper)
  covered = (lower <= rate) & (upper >= rate)
  return np.mean(covered)

a = 1
b = 1
n_sim = 1000
confidence_level = 0.95

k_values = np.arange(1, 101, 9)
rates = np.linspace(0, 1, 10)

test_values = [(k, r) for r in rates for k in k_values]

coverages = [sim_coverage(a, b, n_sim, k, r, confidence_level) for k, r in test_values]
k_plot, r_plot = zip(*test_values)

plt.tricontourf(r_plot, k_plot, coverages, levels=np.round(np.linspace(0, 1, 20), 2))
plt.colorbar()
plt.show()
```
