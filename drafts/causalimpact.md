---
layout: post
title: "Causal Inference"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: causalimpact.png
---

# Causal inference is everywhere you look

Children frequently 

# How would we know if the IRA decreased inflation?

# Computing a counterfactual with `CausalImpact`

# Stress testing our counterfactual model: Placebo trials and assumption robustness

# Causal inference is everywhere you look, and now you get to do it too

# Draft ideas

CausalImpact - https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41854.pdf

TFP version in Python - https://github.com/google/tfp-causalimpact/blob/main/docs/quickstart.ipynb

Generate counterfactual via time series modeling, fit on pre-treatment period

y_t = (linear?) local trend + local level from seasonal state space model + f(covar)

Local level is something like AR(1) plus lagged terms for seasonal components

Rubin perspective: We are filling in the other half of the potential outcome equation

Pearl perspective: We are controlling for cyclic effects, the same things that affect parallel time series, and any other factors we include as covariates

# Intro: What and why in causal inference

An example we'll return to again and again: Did the inflation reduction act reduce inflation? (I am not an economist, but please see https://apnews.com/article/biden-inflation-reduction-climate-anniversary-9950f7e814ac71e89eee3f452ab17f71, who cite the price of oil and gas on the world market, interest rate, and supply chain challenges as causes)

Potential outcomes

DAG

# Problem setup

Break down the model - functional form, dag

# Using CausalImpact to estimate the effect

# Doing it right: Placebo, Sensitivity

# Outro

Some last thoughts

Are your counterfactual states well defined? Is it really clear whether "more jobs" is an unalloyed good?
