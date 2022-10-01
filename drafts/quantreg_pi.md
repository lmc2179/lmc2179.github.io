# PIs are good but OLS doesn't always get it right

## PIs are useful

point forecasts aren't true

biz usage: forecast ranges for planning

Data from previous year:
X = previous spend at location, y = spend during upcoming high season

```python
plt.scatter(df['x'], df['y'])
plt.show()
```

## OLS doesn't handle this well

OLS PI fails when there is heteroskedasticity

example where ols doesn't work; link last post

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

Log is not a panacea, link other post

## Conditional quantiles for PIs

Conditional quantile is kind of like conditional mean from ols

Start with OLS, change loss to get conditional quantile

# Quantile regression example

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

Evidence of heteroskedasticity? differing slopes of high and low

Evidence of asymmetry? high - mid == mid - low

## Checking the model

Coverage

```python
covered = (df['y'] >= low_model.predict(df)) & (df['y'] <= high_model.predict(df))
print(np.average(covered))
```

Coverage CI

Coverage plot

```python
sns.regplot(df['x'], covered, x_bins=5)
plt.axhline(.9, linestyle='dotted', color='black')
plt.show()
```

What other models might we have considered? splines

# Other ideas

Other uses of quantile regression: seeing distributional impact when there are many covariates

Other ways of doing PIs: link to shalizi

# Appendix: DGP

```python
import numpy as np
from scipy.stats import skewnorm
import pandas as pd

n = 250
x = np.linspace(.1, 1, n)
y = 1 + x + skewnorm(np.arange(len(x))+.01, scale=x).rvs()

df = pd.DataFrame({'x': x, 'y': y})
```
