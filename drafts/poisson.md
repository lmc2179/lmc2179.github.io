

Demo of poisson-gamma distribution with "flat" prior. Note coverage is good with a single sample unless rate is near one

```
from scipy.stats import gamma, poisson
import numpy as np


n_sim = 1000

n_sample_size = 1

test_rates = [0.5, 1, 2, 3, 4, 5, 10]

coverage_rates = []

for true_rate in test_rates:
    correct_coverage = 0
    a,b = .00001, .00001
    
    for _ in range(n_sim):
        data = poisson(true_rate).rvs(n_sample_size)
        low, high = gamma(a + np.sum(data), scale = 1./(b + len(data))).interval(.95)
        if low < true_rate < high:
            correct_coverage += 1
    
    coverage_rates.append(correct_coverage / n_sim)
    
plt.plot(test_rates, coverage_rates, marker='o')`
```
