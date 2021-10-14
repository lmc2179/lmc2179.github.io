```
from scipy.optimize import LinearConstraint, minimize

n_vars = 2

def f(v):
  x1, x2 = v
  return (x1-1)**2 + (x2-0.5)**2

linear_constraint = LinearConstraint([[1]*n_vars], [1], [1])

res = minimize(f, [1./n_vars]*n_vars,  constraints=[linear_constraint])
```

```
from scipy.optimize import LinearConstraint, minimize
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/master/causal-inference-for-the-brave-and-true/data/smoking.csv')

piv = df[['year', 'state', 'cigsale']].pivot(index='year', columns='state')['cigsale']

i = 31

X, y = piv.drop(i, axis=1), piv[i]

#def f(w):
#  if np.any(w < 0):
#    return np.inf
#  y_pred = np.dot(X, w)
#  err = np.sum((y - y_pred)**2)
#  print(err)
#  return err
#
#n_vars = X.shape[1]
#
#sum_to_one_constraint = LinearConstraint([[1]*n_vars], [1], [1])
#positive_constraint = LinearConstraint([[1]*n_vars], [0], [np.inf])
#bounds = [(0, 1) for _ in range(n_vars)]
#
#x0 =  ([1./n_vars]*n_vars)
#x0 = (np.random.dirichlet([1.]*n_vars))
#
#res = minimize(f, x0,  constraints=[sum_to_one_constraint])
```

https://rdrr.io/cran/Synth/man/basque.html

https://mixtape.scunning.com/synthetic-control.html#synthetic-control

https://economics.mit.edu/files/11870

https://en.wikipedia.org/wiki/Synthetic_control_method

And do a block bootstrap to get SEs for each year during the test period

Blocks: https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy/6811241#6811241

Consider also the case of just two series, one intervened on and one not: https://bcallaway11.github.io/did/articles/multi-period-did.html
ie diff-in-diff with multiple periods

Worth mentioning interrupted time series too?

https://economics.mit.edu/files/11859 <-- One of the key papers and a nice explanation of why they use the convex combination

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3192710
