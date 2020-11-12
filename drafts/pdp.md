---
layout: post
title: "Partial dependence plots are a simple way to make black-box models simple to understand"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*A commonly cited drawback of black-box Machine Learning or nonparametric models is that they're hard to interpret. Sometimes, analysts are even willing to use a model that fits the data poorly because the model is easy to interpret. However, we can often produce clear interpretations of complex models by constructing Partial Dependence Plots. These plots are model-agnostic, easy to implement in Python, and have a natural interpretation even for non-experts. They're awesome tools, and you should use them all the time to understand the relationships you're modeling, even when the model that best fits the data is really complex.*

# Modeling complex relationships often requires complex models

# But black-box models can make it hard to understand the effect of a single feature

I've met a number of smart, skilled analysts who at this point will throw up their hands and just fit a model that they know is not very good, but has a clear interpretation. This is really unfortunate, and generally not necessary.

# An example: ???

https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity

# Option 1: Make a scatter plot and ignore the other variables

# Option 2: Build a parametric model with a clear interpretation, like a linear model

# Option 3: Build a complex model and use a partial dependence plot

# "Significance tests" and Confidence intervals for black-box models with bootstrapping

# Black-box causal models: Tricky, but possible with some assumptions

# Epilogue: So are machine learning models and PDPs the solution to every modeling problem?

No, the statistical theory around regression models is often the right solution but you should be prepared to realize when it's not the only solution
