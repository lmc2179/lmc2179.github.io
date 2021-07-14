$$
 \underbrace{y_t}_\textrm{Outcome at time t} \sim
\underbrace{(\sum_{i=1}^{p} \phi_i y_{t-i})}_\textrm{Lag terms} 
+ \underbrace{\alpha}_\textrm{Intercept} 
+ \underbrace{\gamma t}_\textrm{Trend}
+ \underbrace{\beta X_t}_\textrm{Extra factors}
+ \underbrace{\epsilon_t}_\textrm{White Noise} 
$$

https://otexts.com/fpp2/AR.html
https://otexts.com/fpp2/wn.html

```python
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from patsy import dmatrix, build_design_matrices
from statsmodels.graphics.tsaplots import plot_pacf

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

train_log_pred = ar_fit.predict(start=train_df.t.min(), end=train_df.t.max(), exog=train_exog)
test_log_pred = ar_fit.predict(start=test_df.t.min(), end=test_df.t.max(), exog_oos=test_exog)

plt.plot(df.t, df.Passengers, label='Observed')
plt.plot(train_df.t, 
         np.exp(train_log_pred), linestyle='dashed', label='In-sample prediction')
plt.plot(test_df.t, 
        np.exp(test_log_pred), 
        linestyle='dashed', label='Out-of-sample prediction')
plt.show()

# Model check
plot_pacf(ar_fit.resid)
plt.show()

# Prediction intervals
residual_variance = np.var(ar_fit.resid)

plt.plot(df.t, df.Passengers, label='Observed')
plt.plot(train_df.t, 
         np.exp(train_log_pred), linestyle='dashed', label='In-sample prediction')

prediction_interval_variance = np.arange(1, len(test_df)+1) * residual_variance
test_log_pred_lower = test_log_pred - 1.96*np.sqrt(prediction_interval_variance)
test_log_pred_upper = test_log_pred + 1.96*np.sqrt(prediction_interval_variance)

plt.plot(test_df.t, 
        np.exp(test_log_pred), 
        linestyle='dashed', label='Out-of-sample prediction')
plt.plot(test_df.t, 
        np.exp(test_log_pred), 
        linestyle='dashed', label='Out-of-sample prediction')
plt.fill_between(test_df.t, 
        np.exp(test_log_pred_lower), np.exp(test_log_pred_upper), 
        label='Prediction interval', alpha=.1)
plt.legend()
plt.show()
```

Add a monthly cycle; Logarithm; Check https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html
