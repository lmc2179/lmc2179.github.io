# PIs are good but OLS doesn't always get it right

## Most useful forecasts include a range of likely outcomes

How much will our new inventory cost; how much revenue will we make; 

point forecasts aren't true or interesting

biz usage: forecast ranges for planning resource allocation (capacity planning, investing)

As is so often the case, it's useful to consider a specific example. Let's imagine Purrberry

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
![Scatterplot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/quantreg_pi/Figure_1.png)

expected value increases with x, so does variance; noise looks asymmetric

## Our usual tool, OLS, doesn't always handle this well

OLS PI fails when there is heteroskedasticity

example where ols doesn't work; link last post

$y ~ \alpha + \beta x + N(0, \sigma)$

estimate of sigma; yes we're sweeping a little uncertainty under the rug

```python
from statsmodels.api import formula as smf

ols_model = smf.ols('on_season_revenue ~ off_season_revenue', df).fit()
predictions = ols_model.predict(df)
resid_sd = np.std(ols_model.resid)

high, low = predictions + 1.645 * resid_sd, predictions - 1.645 * resid_sd

plt.scatter(df['off_season_revenue'], df['on_season_revenue'])
plt.plot(df['off_season_revenue'], high, label='OLS 90% high PI')
plt.plot(df['off_season_revenue'], predictions, label='OLS prediction')
plt.plot(df['off_season_revenue'], low, label='OLS 90% low PI')
plt.legend()
plt.xlabel('Off season revenue at location')
plt.ylabel('On season revenue at location')
plt.title('OLS prediction intervals')
plt.show()
```

![OLS plot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/quantreg_pi/Figure_2.png)

because it assumes constant, symmetric noise

Log is not a panacea, link other post

## The idea: create prediction intervals based on the conditional quantiles

looks for the region where 90% of the observations will fall for a given store

Conditional quantile is kind of like conditional mean from ols

Start with OLS, change loss to get conditional quantile

$\mathbb{E}[y \mid x]$

$\mathbb{Q}[y \mid x]$

# Quantile regression in action

## Fitting the model

Evidence of heteroskedasticity: non-zero slopes for high and low

```python
high_model = smf.quantreg('on_season_revenue ~ off_season_revenue', df).fit(q=.95)
mid_model = smf.quantreg('on_season_revenue ~ off_season_revenue', df).fit(q=.5)
low_model = smf.quantreg('on_season_revenue ~ off_season_revenue', df).fit(q=.05)

plt.scatter(df['off_season_revenue'], df['on_season_revenue'])
plt.plot(df['off_season_revenue'], high_model.predict(df), label='95% Quantile')
plt.plot(df['off_season_revenue'], mid_model.predict(df), label='50% Quantile (Median)')
plt.plot(df['off_season_revenue'], low_model.predict(df), label='5% Quantile')
plt.legend()
plt.xlabel('Off season revenue at location')
plt.ylabel('On season revenue at location')
plt.title('Quantile Regression prediction intervals')
plt.show()
```

![Quantreg scatterplot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/quantreg_pi/Figure_3.png)

Evidence of heteroskedasticity? differing slopes of high and low

Evidence of asymmetry? high - mid == mid - low

## Checking the model

Coverage

```python
from scipy.stats import sem
covered = (df['on_season_revenue'] >= low_model.predict(df)) & (df['on_season_revenue'] <= high_model.predict(df))
print('In-sample coverage rate: ', np.average(covered))
print('Coverage SE: ', sem(covered))
```
```
In-sample coverage rate:  0.896
Coverage SE:  0.019345100974843932
```

Coverage CI

Model selection metric

Coverage plot

```python
sns.regplot(df['off_season_revenue'], covered, x_bins=4)
plt.axhline(.9, linestyle='dotted', color='black')
plt.title('Coverage by revenue group')
plt.xlabel('Off season revenue at location')
plt.ylabel('Coverage')
plt.show()
```

![Coverage plot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/quantreg_pi/Figure_4.png)

What other models might we have considered? splines

# Some other perspectives on quantile regression and prediction intervals

Other uses of quantile regression: seeing distributional impact when there are many covariates

Other ways of doing PIs: link to shalizi

# Appendix: How the data was generated

The feline fashion visionaries at Purrberry are, regrettably, entirely fictional for the time being. The data from this example was generated using the below code, which creates skew normal distributed noise:

```python
import numpy as np
from scipy.stats import skewnorm
import pandas as pd

n = 250
x = np.linspace(.1, 1, n)
gen = skewnorm(np.arange(len(x))+.01, scale=x)
gen.random_state = np.random.Generator(np.random.PCG64(abs(hash('predictions'))))
y = 1 + x + gen.rvs()

df = pd.DataFrame({'off_season_revenue': x, 'on_season_revenue': y})
```
