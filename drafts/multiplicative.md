Log transformations, multiplicative models, and time series decompositions

https://xkcd.com/451/

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
