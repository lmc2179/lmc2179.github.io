# PIs are good but OLS doesn't always get it right

# Conditional quantiles for PIs

# Quantile regression example

## Looking at the data

```python
import numpy as np
from scipy.stats import skewnorm
import pandas as pd

n = 250
x = np.linspace(.1, 1, n)
y = 1 + x + skewnorm(np.arange(len(x))+.01, scale=x).rvs()

df = pd.DataFrame({'x': x, 'y': y})

plt.scatter(df['x'], df['y'])
plt.show()

from statsmodels.api import formula as smf
high_model = smf.quantreg('y ~ x', df).fit(q=.95)
mid_model = smf.quantreg('y ~ x', df).fit(q=.5)
low_model = smf.quantreg('y ~ x', df).fit(q=.05)

plt.scatter(df['x'], df['y'])
plt.plot(df['x'], high_model.predict(df))
plt.plot(df['x'], mid_model.predict(df))
plt.plot(df['x'], low_model.predict(df))
plt.show()

index_plot = np.arange(len(df))
plt.vlines(index_plot, low_model.predict(df), high_model.predict(df))
plt.scatter(index_plot, df['y'])
plt.show()

covered = (df['y'] >= low_model.predict(df)) & (df['y'] <= high_model.predict(df))
print(np.average(covered))

```

## Fitting the model

Evidence of heteroskedasticity: non-zero slopes for high and low

Evidence of asymmetry? high - mid == mid - low

## Checking the model

coverage

consider other models?

## Generating simulated data

# Other ideas

Other uses of quantile regression

Other ways of doing PIs
