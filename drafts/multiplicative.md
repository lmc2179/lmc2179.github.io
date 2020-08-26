# When do we log transform the response variable? Model assumptions, multiplicative models and time series decompositions

*TL;DR - Sometimes, analysts will recommend a log transformation of the outcome variable to "make the residuals look normal". In some cases this is just papering over other issues, but sometimes this kind of transformation genuinely improves the inference or produces a better fitting model. In what cases does this happen? Why does the log transformation work the way it does?*

https://xkcd.com/451/

# One common reason: Because log-transformed data fits the model assumptions than any other transformation

https://stats.stackexchange.com/a/3530/29694
http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/07/lecture-07.pdf

It's worth pointing out that this is _not_ always the right move.

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

Example: Treatment effect multiplies instead of adding

# A common use case: Multiplicative time series decomposition

There's a use case where multiplicative combinations are common enough that it's worth walking through it here. 

https://en.wikipedia.org/wiki/Decomposition_of_time_series

https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv

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
