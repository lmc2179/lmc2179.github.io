```
from scipy.optimize import LinearConstraint, minimize

n_vars = 2

def f(v):
  x1, x2 = v
  return (x1-1)**2 + (x2-0.5)**2

linear_constraint = LinearConstraint([[1]*n_vars], [1], [1])

res = minimize(f, [1./n_vars]*n_vars,  constraints=[linear_constraint])
```

https://rdrr.io/cran/Synth/man/basque.html

https://economics.mit.edu/files/11870

https://en.wikipedia.org/wiki/Synthetic_control_method

And do a block bootstrap to get SEs for each year during the test period

Blocks: https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy/6811241#6811241
