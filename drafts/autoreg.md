# Intro

TS data is everywhere

Use cases:
- Monitoring KPIs
- Capacity planning
- Understanding treatment rollout

Tasks
- Description task
- Forecasting task

# Example: Airline passenger forecasting and the AR-X(p) model

Read CSV

```python
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from patsy import dmatrix, build_design_matrices

df = pd.read_csv('airline.csv')

df['log_passengers'] = np.log(df.Passengers)

df['year'] = df['Month'].apply(lambda x: int(x.split('-')[0]))
df['month_number'] = df['Month'].apply(lambda x: int(x.split('-')[1]))
df['t'] = df.index.values

train_cutoff = 96
validate_cutoff = 120

train_df = df[df['t'] <= train_cutoff]
select_df = df[(df['t'] > train_cutoff) & (df['t'] <= validate_cutoff)]
forecast_df = df[df['t'] > validate_cutoff]

dm = dmatrix('C(month_number)-1', df)
train_exog = build_design_matrices([dm.design_info], train_df, return_type='dataframe')[0]
select_exog = build_design_matrices([dm.design_info], select_df, return_type='dataframe')[0]
forecast_exog = build_design_matrices([dm.design_info], forecast_df, return_type='dataframe')[0]

plt.plot(train_df.t, train_df.Passengers, label='Training data')
plt.plot(select_df.t, select_df.Passengers, label='Model selection holdout')
plt.legend()
plt.title('Airline passengers by month')
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.show()

```

Show plot with validation line [IMAGE]

We can observe a few features of this data set which will show up in our model:
- On the first date observed, the value is non-zero
- There is a positive trend
- There are regular cycles of 12 months
- The next point is close to the last point

Our model will include:
- An intercept term, representing the value at t = 0
- A linear trend term
- A set of lag terms, encoding how the next observation depends on those just before it
- A set of "additional factors", which in our case will be dummy variables for the months of the year
- A [white noise term](https://otexts.com/fpp2/wn.html), the time-series analogue of IID Gaussian noise (the two are [not quite identical](https://dsp.stackexchange.com/questions/23881/what-is-the-difference-between-i-i-d-noise-and-white-noise), but the differences aren't relevant here)

Formally, the model we'll use looks like this:

$$
 log \underbrace{y_t}_\textrm{Outcome at time t} \sim
\underbrace{\alpha}_\textrm{Intercept} 
+ \underbrace{\gamma t}_\textrm{Trend}
+ \underbrace{(\sum_{i=1}^{p} \phi_i y_{t-i})}_\textrm{Lag terms} 
+ \underbrace{\beta X_t}_\textrm{Extra factors}
+ \underbrace{\epsilon_t}_\textrm{White Noise} 
$$

The model above is a type of autoregressive model (so named because the target variable is regressed on lagged versions of itself). More precisely, this gives us the [AR-X(p) model](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html), an [AR(p) model](https://otexts.com/fpp2/AR.html) with extra inputs.

As we've [previously discussed in this post](https://lmc2179.github.io/posts/multiplicative.html), it makes sense to take the log of the dependent variable here.

```python
from statsmodels.tsa.ar_model import AutoReg

ar_model = AutoReg(endog=train_df.log_passengers, exog=train_exog, lags=5, trend='ct')
ar_fit = ar_model.fit()

train_log_pred = ar_fit.predict(start=train_df.t.min(), end=train_df.t.max(), exog=train_exog)

plt.plot(train_df.t, 
         np.exp(train_log_pred), linestyle='dashed', label='In-sample prediction')
plt.plot(train_df.t, train_df.Passengers, label='Training data')
plt.legend()
plt.title('Airline passengers by month')
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.show()
```

plot the in-sample fit [IMAGE]

summary()

```python
print(ar_fit.summary())
```

```
[OUTPUT]
```

# Model checking and model selection

plot OoS fit on validation [IMAGE]

```python
select_log_pred = ar_fit.predict(start=select_df.t.min(), end=select_df.t.max(), exog_oos=select_exog)

plt.plot(train_df.t, 
         np.exp(train_log_pred), linestyle='dashed', label='In-sample prediction')
plt.plot(select_df.t, 
         np.exp(select_log_pred), linestyle='dashed', label='Validation set prediction')
plt.plot(train_df.t, train_df.Passengers, label='Training data')
plt.plot(select_df.t, select_df.Passengers, label='Model selection holdout')
plt.legend()
plt.title('Airline passengers by month')
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.show()
```

error vs choice of p plot [CODE][IMAGE]

residual plot [CODE][IMAGE]

PACF [CODE][IMAGE]

# Producing forecasts and prediction intervals

Show prediction intervals [CODE][IMAGE]

$\sqrt{k \hat{\sigma}^2}$

------------------------------



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
