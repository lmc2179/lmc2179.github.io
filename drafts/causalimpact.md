CausalImpact - https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41854.pdf

TFP version in Python - https://github.com/google/tfp-causalimpact/blob/main/docs/quickstart.ipynb

Generate counterfactual via time series modeling, fit on pre-treatment period

y_t = (linear?) local trend + local level from seasonal state space model + f(covar)

Local level is something like AR(1) plus lagged terms for seasonal components

Rubin perspective: We are filling in the other half of the potential outcome equation

Pearl perspective: We are controlling for cyclic effects, the same things that affect parallel time series, and any other factors we include as covariates

# Intro: What and why in causal inference

Potential outcomes

DAG

# Problem setup

Break down the model - functional form, dag

# Using CausalImpact to estimate the effect

# Doing it right: Placebo, Sensitivity

# Outro
