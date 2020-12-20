You should include pretreatment measurements of the outcome and unit covariates in your A/B tests, but please be sure they're pre-treatment

We'll show it with some simulations of experiments with binary treatments and continuous normally distributed outcomes:
1. No extra variable, y ~ t
2. One extra variable, y ~ t vs y ~ t + x
3. One unobserved extra variable but an observed ancestor, y ~ t vs y ~ t + x and y ~ t vs y ~ t + z (where z is the parent of x)
4. One irrelevant extra variable
5. One incorrectly-controlled-for extra variable (DO NOT CONDITION ON POST-TREATMENT VARIABLES, conditioning on a variable which is a result of x)
n = 100, alpha = 1, delta=0.5

Maybe this is just one dataset and we just get better at seeing it

https://pubmed.ncbi.nlm.nih.gov/26921693/

# Including pretreatment measurements gives us more efficient estimates of treatment effects in A/B tests

## Example

## Simulations

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.api import formula as smf

def generate_data():
  baseline = 1
  treatment_effect = 1
  residual_sd = 0.5
  t = np.array([0]*50 + [1]*50)
  y = np.concatenate((np.random.normal(baseline, residual_sd, 50), np.random.normal(baseline + treatment_effect, residual_sd, 50)))
  df = pd.DataFrame({'t': t, 'y': y})
  return df

n_sim = 10
  
sim_results = []
  
for i in range(n_sim):
  sim_df = generate_data()
  model = smf.ols('y ~ t', sim_df)
  result = model.fit()
  sim_results.append(result)
```

## Why this works - smaller residual size

# What happens if we pick a bad pretreatment measurement?



# Even a really good pretreatment measurement doesn't solve the problem of a missing confounder

```python
import numpy as np
from statsmodels.api import formula as smf
import pandas as pd
from scipy.special import expit
from matplotlib import pyplot as plt
import seaborn as sns

n = 1000

s_c = 1
s_p = 1
s_y = 1

a_p = 0
b_cp = 1
a_y = 0
b_ty = 1
b_cy = 1

c = np.random.normal(0, s_c, n)
t = np.random.binomial(1, expit(c))
p = np.random.normal(a_p + b_cp*c, s_p)
y = np.random.normal(a_y + b_ty*t + b_cy*c, s_y)

df = pd.DataFrame({'c': c, 't': t, 'p': p, 'y': y})

print(smf.ols('y ~ t', df).fit().summary())
print(smf.ols('y ~ t + c', df).fit().summary())
print(smf.ols('y ~ t + p', df).fit().summary())

plt.scatter(df[df['t'] == 1]['c'], df[df['t'] == 1]['y'])
plt.scatter(df[df['t'] == 0]['c'], df[df['t'] == 0]['y'])
plt.show()

plt.scatter(df[df['t'] == 1]['p'], df[df['t'] == 1]['y'])
plt.scatter(df[df['t'] == 0]['p'], df[df['t'] == 0]['y'])
plt.show()
```

Can we say that it's an improvement over the naive analysis? How much?
