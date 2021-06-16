```python
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

df = pd.read_csv('airline.csv')

df['year'] = df['Month'].apply(lambda x: int(x.split('-')[0]))
df['month_number'] = df['Month'].apply(lambda x: int(x.split('-')[1]))
df['t'] = df.index.values

ar_model = AutoReg(endog=df.Passengers, lags=20, trend='ct')
ar_fit = ar_model.fit()

plt.plot(df.t, df.Passengers)
plt.plot(df.t, ar_fit.predict(end=df.t.max()))
plt.show() # Now do OOS
```
