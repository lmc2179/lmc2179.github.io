# PIs are good but OLS doesn't always get it right

## PIs are useful

point forecasts aren't true

biz usage: forecast ranges for planning

As is so often the case, it's useful to consider a specific example.

Data from previous year:
X = previous spend at location, y = spend during upcoming high season

```python
from matplotlib import pyplot as plt
import seaborn as sns

plt.scatter(df['off_season_revenue'], df['on_season_revenue'])
plt.xlabel('Off season revenue at location')
plt.ylabel('On season revenue at location')
plt.title('Comparison between on and off season revenue at store locations')
plt.show()
```

-scatterplot-

## OLS doesn't handle this well

OLS PI fails when there is heteroskedasticity

example where ols doesn't work; link last post

$y ~ \alpha + \beta x + N(0, \sigma)$

```python
from statsmodels.api import formula as smf

ols_model = smf.ols('on_season_revenue ~ off_season_revenue', df).fit()
predictions = ols_model.predict(df)
resid_sd = np.std(ols_model.resid)

high, low = predictions + 1.645 * resid_sd, predictions - 1.645 * resid_sd

plt.scatter(df['off_season_revenue'], df['on_season_revenue'])
plt.plot(df['off_season_revenue'], high, label='OLS 90% high PI')
plt.plot(df['off_season_revenue'], pred.predicted_mean, label='OLS prediction')
plt.plot(df['off_season_revenue'], low, label='OLS 90% low PI')
plt.legend()
plt.xlabel('Off season revenue at location')
plt.ylabel('On season revenue at location')
plt.title('OLS prediction intervals')
plt.show()
```

because it assumes constant, symmetric noise

Log is not a panacea, link other post

## Conditional quantiles for PIs

Conditional quantile is kind of like conditional mean from ols

Start with OLS, change loss to get conditional quantile

$\mathbb{E}[y \mid x]$

$\mathbb{Q}[y \mid x]$

# Quantile regression example

## Fitting the model

Evidence of heteroskedasticity: non-zero slopes for high and low

```python
high_model = smf.quantreg('on_season_revenue ~ off_season_revenue', df).fit(q=.95)
mid_model = smf.quantreg('on_season_revenue ~ off_season_revenue', df).fit(q=.5)
low_model = smf.quantreg('on_season_revenue ~ off_season_revenue', df).fit(q=.05)

plt.scatter(df['off_season_revenue'], df['on_season_revenue'])
plt.plot(df['off_season_revenue'], high_model.predict(df))
plt.plot(df['off_season_revenue'], mid_model.predict(df))
plt.plot(df['off_season_revenue'], low_model.predict(df))
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

df = pd.DataFrame({'off_season_revenue': x, 'on_season_revenue': y})
```
