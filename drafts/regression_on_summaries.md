

```python
import numpy as np
from scipy.optimize import minimize

def logpdf_sufficient(mu, sigma_sq, sample_mean, sample_var, n):
  return -(n/2) * np.log(2*np.pi*sigma_sq) - (((n-1) / (2*sigma_sq)) * sample_var) - ((n / (2*sigma_sq)) * (mu - sample_mean)**2) 

data = np.random.normal(0, 1, 1000)
n = len(data)
m, v = np.mean(data), np.var(data)

def neg_log_likelihood(p):
  return -logpdf_sufficient(p[0], np.exp(p[1]), m, v, n)

result = minimize(neg_log_likelihood, np.array([5, -1]))
cov = result.hess_inv

se_mean = np.sqrt(cov[0][0])
```
