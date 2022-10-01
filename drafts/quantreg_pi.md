# PIs are good but OLS doesn't always get it right

OLS PI fails when there is heteroskedasticity

Log is not a panacea

# Conditional quantiles for PIs

Intuition: change loss to get conditional quantile

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
```

but ols doesn't work

```python
from statsmodels.api import formula as smf

ols_model = smf.ols('y ~ x', df).fit()
pred = ols_model.get_prediction(df)

high, low = zip(*pred.conf_int(.9))

plt.scatter(df['x'], df['y'])
plt.plot(df['x'], high)
plt.plot(df['x'], pred.predicted_mean)
plt.plot(df['x'], low)
plt.show()
```

because it assumes constant, symmetric noise

## Fitting the model

Evidence of heteroskedasticity: non-zero slopes for high and low

```python
high_model = smf.quantreg('y ~ x', df).fit(q=.95)
mid_model = smf.quantreg('y ~ x', df).fit(q=.5)
low_model = smf.quantreg('y ~ x', df).fit(q=.05)

plt.scatter(df['x'], df['y'])
plt.plot(df['x'], high_model.predict(df))
plt.plot(df['x'], mid_model.predict(df))
plt.plot(df['x'], low_model.predict(df))
plt.show()
```

Evidence of asymmetry? high - mid == mid - low

## Checking the model

```python
covered = (df['y'] >= low_model.predict(df)) & (df['y'] <= high_model.predict(df))
print(np.average(covered))

sns.regplot(df['x'], covered, x_bins=5)
plt.axhline(.9, linestyle='dotted', color='black')
plt.show()
```

consider other models?

## Generating simulated data

# Other ideas

Other uses of quantile regression

Other ways of doing PIs
