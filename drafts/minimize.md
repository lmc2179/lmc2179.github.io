_?_

# We often want to tailor our model to the situation

Example of diminishing returns - satiation process? Cat treats, marketing spend, organic growth

Lots of relationships between two real variables go something like this: "As x increases, so does y. But the next x causes less increase than the last one"

The problem - I can't do `from statsmodels import gompertz_regression`

In this case, we can use scipy to find the best parameters and their SEs

# ?

```python
from scipy.stats import norm, sem
from scipy.optimize import minimize
import numpy as np

x = norm(0, 1).rvs(1000)

def neg_log_likelihood(param_vector):
  mu, log_sd = param_vector
  return -np.sum(norm(mu, np.exp(log_sd)).logpdf(x))
  
result = minimize(neg_log_likelihood, [10, 10])

print(np.sqrt(np.diag(result.hess_inv)), sem(x))
```
