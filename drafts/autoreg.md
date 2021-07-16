# Intro

For practicing data scientists, time series data is everywhere - almost anything we care to observe can be observed over time. Some use cases that have shown up frequently in my work are:
- **Monitoring metrics and KPIs**: We use KPIs to understand some aspect of the business as it changes over time. We often want to model changes in KPIs to see what affects them, or construct a forecast for them into the near future.
- **Capacity planning**: Many businesses have seasonal changes in their demand or supply. Understanding these trends helps us make sure we have enough production, bandwidth, sales staff, etc as conditions change.
- **Understanding the rollout of a new treatment or policy**: As a new policy takes effect, what results do we see? How do our measurements compare with what we expected? By comparing post-treatment observations to a forecast, or including treatment indicators in the model, we can get an understanding of this.

Each of these use cases is a combination of **description** (understanding the structure of the series as we observe it) and **forecasting** (predicting how the series will look in the future). We can perform both of these tasks using the implementation of Autoregressive models in Python found in statsmodels. 

# Example: Airline passenger forecasting and the AR-X(p) model

We'll use a time series of [monthly airline passenger counts from 1949 to 1960](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv) in this example. An airline or shipping company might use this for capacity planning.

We'll read in the data using pandas:

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
```

Then, we'll split it into three segments: training, model selection, and forecasting. We'll select the complexity of the model using the model selection set as a holdout, and then attempt to forecast into the future on the forecasting set. Note that this is time series data, so we need to split the data set into three sequential groups, rather than splitting it randomly. We'll use a model selection/forecasting set of about 24 months each, a plausible period of time for an airline to forecast demand.

```python
train_cutoff = 96
validate_cutoff = 120

train_df = df[df['t'] <= train_cutoff]
select_df = df[(df['t'] > train_cutoff) & (df['t'] <= validate_cutoff)]
forecast_df = df[df['t'] > validate_cutoff]

dm = dmatrix('C(month_number)-1', df)
train_exog = build_design_matrices([dm.design_info], train_df, return_type='dataframe')[0]
select_exog = build_design_matrices([dm.design_info], select_df, return_type='dataframe')[0]
forecast_exog = build_design_matrices([dm.design_info], forecast_df, return_type='dataframe')[0]
```

Let's visualize the training and model selection data:

```python
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

```python
from scipy.stats import sem

lag_values = np.arange(1, 40)
mse = []
error_sem = []

for p in lag_values:
    ar_model = AutoReg(endog=train_df.log_passengers, exog=train_exog, lags=p, trend='ct')
    ar_fit = ar_model.fit()
    
    select_log_pred = ar_fit.predict(start=select_df.t.min(), end=select_df.t.max(), exog_oos=select_exog)
    select_resid = select_df.Passengers - np.exp(select_log_pred)
    mse.append(np.mean(select_resid**2))
    error_sem.append(sem(select_resid**2))
    
mse = np.array(mse)
error_sem = np.array(error_sem)
    
plt.plot(lag_values, mse, marker='o')
plt.fill_between(lag_values, mse - error_sem, mse + error_sem, alpha=.1)
plt.show()
```

```python
train_and_select_df = df[df['t'] <= validate_cutoff]
train_and_select_exog = build_design_matrices([dm.design_info], train_and_select_df, return_type='dataframe')[0]

ar_model = AutoReg(endog=train_and_select_df.log_passengers, 
                   exog=train_and_select_exog, lags=17, trend='ct')
                   # Actually this should be on both the train + select sets :(
ar_fit = ar_model.fit()

plt.title('Residuals')
plt.plot(ar_fit.resid)
plt.show()
```

residual plot [IMAGE]

```python
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(ar_fit.resid)
plt.show()
```

PACF [IMAGE]

# Producing forecasts and prediction intervals

```python
train_and_select_log_pred = ar_fit.predict(start=train_and_select_df.t.min(), end=train_and_select_df.t.max(), exog_oos=train_and_select_exog)
forecast_log_pred = ar_fit.predict(start=forecast_df.t.min(), end=forecast_df.t.max(), exog_oos=forecast_exog)

plt.plot(train_and_select_df.t, 
         np.exp(train_and_select_log_pred), linestyle='dashed', label='In-sample prediction')
plt.plot(forecast_df.t, 
         np.exp(forecast_log_pred), linestyle='dashed', label='Forecast')
plt.plot(train_and_select_df.t, train_and_select_df.Passengers, label='Training data')
plt.plot(forecast_df.t, forecast_df.Passengers, label='Out-of-sample')
plt.legend()
plt.title('Airline passengers by month')
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.show()
```

```python
residual_variance = np.var(ar_fit.resid)
prediction_interval_variance = np.arange(1, len(forecast_df)+1) * residual_variance
forecast_log_pred_lower = forecast_log_pred - 1.96*np.sqrt(prediction_interval_variance)
forecast_log_pred_upper = forecast_log_pred + 1.96*np.sqrt(prediction_interval_variance)

plt.plot(train_and_select_df.t, 
         np.exp(train_and_select_log_pred), linestyle='dashed', label='In-sample prediction')
plt.plot(forecast_df.t, 
         np.exp(forecast_log_pred), linestyle='dashed', label='Forecast')
plt.fill_between(forecast_df.t, 
        np.exp(forecast_log_pred_lower), np.exp(forecast_log_pred_upper), 
        label='Prediction interval', alpha=.1)
plt.plot(train_and_select_df.t, train_and_select_df.Passengers, label='Training data')
plt.plot(forecast_df.t, forecast_df.Passengers, label='Out-of-sample')
plt.legend()
plt.title('Airline passengers by month')
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.show()
```

Show prediction intervals [IMAGE]

$\sqrt{k \hat{\sigma}^2}$
