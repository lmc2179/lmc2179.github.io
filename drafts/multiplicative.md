# When do we log transform the response variable? Model assumptions, multiplicative models and time series decompositions

*TL;DR - Sometimes, analysts will recommend a log transformation of the outcome variable to "make the residuals look normal". In some cases this is just papering over other issues, but sometimes this kind of transformation genuinely improves the inference or produces a better fitting model. In what cases does this happen? Why does the log transformation work the way it does?*

https://xkcd.com/451/

# One common reason: Because log-transformed data fits the model assumptions than any other transformation

A commonly cited justification for log transforming the response variable is that the OLS assumptions are being violated, and the transformation will remedy this. These arguments often go something like:

- My residuals are non-normal because they are skewed or have outliers; a log transform makes them more symmetric.
- My residuals show evidence of heteroskedasticity; log transforming the response makes the residual variance seem constant.
- My dependent variable is constrained to be positive, but in an OLS model the outcome variable can be negative.

A log transformation of the response variable may *sometimes* resolve these issues, and is worth considering. However, each of these problems has other potential solutions:

- Asymmetric residuals could be resolved by a different non-linear transformation of the outcome; the log transform is not special. A square root, for example, may have the same effect. You might also consider a model with an asymmetric or heavy-tailed residual distribution.
- Heteroskedasticity can be accounted for by making the non-constant variance part of your model. In the linear model framework, [WLS](http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/24/lecture-24--25.pdf) is a common solution.
- A dependent variable which is definitionally positive can be accounted for with a GLM other than OLS, like a Negative-binomial model or Gamma model. 

The point here is _not_ that a log transformation can't solve these problems - it sometimes can! Rather, the point is that it will not _always_ solve these problems. It's worth looking at an example where the OLS assumptions are violated but the log transform doesn't help. 

https://stats.stackexchange.com/a/3530/29694
http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/07/lecture-07.pdf

## Log transformations do not automatically fix model assumption problems

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.api import formula as smf
from scipy.stats import norm

df = pd.DataFrame({'x': [0] * 100 + [1] * 100, 'y': np.random.normal([10] * 100 + [20] * 100, [2]*100 + [1]*100)})

model = smf.ols('y ~ x', df)
fit = model.fit()

plt.scatter(df['x'], df['y'] - fit.predict(df))
plt.show()

log_model = smf.ols('np.log(y) ~ x', df)
log_fit = log_model.fit()

plt.scatter(df['x'], log_fit.resid)
plt.show()

```

# Another reason: Because you'd like a model where coefficients combine multiplicatively instead of additively

An attempt to correct bad OLS assumptions isn't the only reason we might log transform the response variable. Fitting a model like this will change the way that the coefficients combine in the predicted value of y. 

This is sometimes called a [log-linear model](https://en.wikipedia.org/wiki/Log-linear_model#:~:text=A%20log%2Dlinear%20model%20is,(possibly%20multivariate)%20linear%20regression) - the logarithm of y is a linear function of X. 

Example: Treatment effect multiplies instead of adding

# A common use case: Multiplicative time series decomposition

There's a use case for a multiplicative combinations which is common enough that it's worth walking through it here. 

https://en.wikipedia.org/wiki/Decomposition_of_time_series

To see the difference between these two models in action, we're going to look at a [classic time series dataset of monthly airline passenger counts from 1949 to 1960](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv). Plotting the dataset, we see some common features of time series data: there are clear seasonal trends, and a steady increase year over year.

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.api import formula as smf

df = pd.read_csv('airline.csv')

df['year'] = df['Month'].apply(lambda x: int(x.split('-')[0]))
df['month_number'] = df['Month'].apply(lambda x: int(x.split('-')[1]))

additive_model = smf.ols('Passengers ~ year + C(month_number)', df)
additive_fit = additive_model.fit()

multiplicative_model = smf.ols('np.log(Passengers) ~ year + C(month_number)', df)
multiplicative_fit = multiplicative_model.fit()

plt.plot(df['Passengers'])
plt.plot(additive_fit.fittedvalues)
plt.plot(np.exp(multiplicative_fit.fittedvalues))
plt.show()

plt.plot(additive_fit.fittedvalues - df['Passengers'])
plt.plot(np.exp(multiplicative_fit.fittedvalues) - df['Passengers'])
plt.show()
```

