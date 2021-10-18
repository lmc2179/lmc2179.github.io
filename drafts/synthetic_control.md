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

i = 30

X, y = piv.drop(i, axis=1), piv[i]

from functools import partial

def loss_w(W, X, y):
    return np.sqrt(np.mean((y - X.dot(W))**2))

from scipy.optimize import fmin_slsqp

def get_w(X, y):
    
    w_start = [1/X.shape[1]]*X.shape[1]

    weights = fmin_slsqp(partial(loss_w, X=X, y=y),
                         np.array(w_start),
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights
    
w_fit = get_w(X, y)

plt.plot(y.index, y)
plt.plot(y.index, np.dot(X, w_fit))
plt.show()
```

```
from scipy.optimize import LinearConstraint, minimize
import pandas as pd
from scipy.special import softmax

df = pd.read_csv('https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/master/causal-inference-for-the-brave-and-true/data/smoking.csv')

piv = df[['year', 'state', 'cigsale']].pivot(index='year', columns='state')['cigsale']

i = 30

X, y = piv.drop(i, axis=1), piv[i]


def f(w_raw):
  w = softmax(w_raw)
  y_pred = np.dot(X, w)
  err = np.sum((y - y_pred)**2)
  print(err)
  return err

n_vars = X.shape[1]

x0 = np.zeros(n_vars)

res = minimize(f, x0)
```

What if we use the softmax function?

https://matheusfacure.github.io/python-causality-handbook/Debiasing-with-Orthogonalization.html

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
