```python
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

df = pd.read_csv('airline.csv')

df['year'] = df['Month'].apply(lambda x: int(x.split('-')[0]))
df['month_number'] = df['Month'].apply(lambda x: int(x.split('-')[1]))
df['t'] = df.index.values

train_cutoff = 80

train_df = df[df['t'] <= train_cutoff]
test_df = df[df['t'] > train_cutoff]

ar_model = AutoReg(endog=train_df.Passengers, lags=20, trend='ct')
ar_fit = ar_model.fit()

plt.plot(df.t, df.Passengers, label='Observed')
plt.plot(train_df.t, ar_fit.predict(start=train_df.t.min(), end=train_df.t.max()), linestyle='dashed', label='In-sample prediction')
plt.plot(test_df.t, ar_fit.predict(start=test_df.t.min(), end=test_df.t.max()), linestyle='dashed', label='Out-of-sample prediction')
plt.show()
```
