# A pretreatment measurement gives us more efficient estimates of treatment effects

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
