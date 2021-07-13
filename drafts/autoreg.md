```python
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from patsy import dmatrix, build_design_matrices

df = pd.read_csv('airline.csv')

df['log_passengers'] = np.log(df.Passengers)

df['year'] = df['Month'].apply(lambda x: int(x.split('-')[0]))
df['month_number'] = df['Month'].apply(lambda x: int(x.split('-')[1]))
df['t'] = df.index.values

train_cutoff = 120

train_df = df[df['t'] <= train_cutoff]
test_df = df[df['t'] > train_cutoff]

dm = dmatrix('C(month_number)-1', df)
train_exog = build_design_matrices([dm.design_info], train_df, return_type='dataframe')[0]
test_exog = build_design_matrices([dm.design_info], test_df, return_type='dataframe')[0]

ar_model = AutoReg(endog=train_df.log_passengers, exog=train_exog, lags=5, trend='ct')
ar_fit = ar_model.fit()

plt.plot(df.t, df.Passengers, label='Observed')
plt.plot(train_df.t, 
         np.exp(ar_fit.predict(start=train_df.t.min(), end=train_df.t.max(), exog=train_exog)), linestyle='dashed', label='In-sample prediction')
plt.plot(test_df.t, 
        np.exp(ar_fit.predict(start=test_df.t.min(), end=test_df.t.max(), exog_oos=test_exog)), 
        linestyle='dashed', label='Out-of-sample prediction')
plt.show()
```

Add a monthly cycle; Logarithm; Check https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html
