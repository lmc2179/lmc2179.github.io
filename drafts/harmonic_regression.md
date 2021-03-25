# Time Series data is full of cyclic patterns

Airline data

# Modeling cyclic behavior with sinusoidal models: Trend + Seasonality + Noise

$y_t \sim \alpha + \beta t + \sum\limits_i \gamma_i sin(\lambda_i t) + \eta_i cos(\lambda_i t)$

# Removing the trend

Get residuals of lin-log a + bt model, or detrend by differencing

# What frequencies should I include? The Periodogram

Extract top k frequencies from periodogram; fit a model with 2k + 2 parameters

# Visualizing the components of our model

plot trend, cyclic part, residual part

# Building a forecast for airline usage

Our previous model had 12+21 (month+year) parameters, and no clear forecasting method

Check cross validation, and in-sample residuals vs time

# Summary: Cooking up a sinusoid model

- Detrend
- Extract top frequencies
- Fit sine model with 2k+2 parameters

# Appendix: Okay, what the _hell_ is a Fourier Transform

3blue1brown

chatfield 2004, ch. 7

# Appendix: The road not taken - detrending by differencing

It's mostly the same: Log, difference, then do the sine thing
