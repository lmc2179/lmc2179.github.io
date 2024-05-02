---
layout: post
title: "Interpreting unit-level feature explanations in Python with SHAP"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: ?.png
---

# Problem - Explaining unit-level feature importance

### "Why is this thing so expensive?"

Building a machine learning model is a little bit like setting up a little robot advisor, who helps you figure out what to do next by giving you expert advice. For example, a model which predicts customer LTV based on customer attributes is like having an advisor who knows all about your customer base. For example, if you build an price prediction model, the expert is able to answer the question: "How much do you think this is going to cost?". A reasonable follow up question for such an expert would be "Why do you think that?". The same way we want our colleagues and advisors to be transparent, we want our model to be transparent. We want to be able to ask a question like: "Why did the model predict this outcome for this unit?".

There are lots of reasons we might ask this question:
* It gives us a quick check on whether the model is doing something surprising - if it's relying heavily on a feature we weren't expecting to be important, 
* We want to explain anomalously high predictions to see what's driving them. Similarly, we might use this to do root cause analysis - if the model predicts a surprisingly large value, we can get some intuition about what potential root cause is relevant.
* We can think about unit-level interventions
* We can start to form causal intuition by looking at this as an "autoamted case study".

Let's return to a familiar example. Imagine you've decided you want to go buy a house, and so you do the most normal thing for a prospective homebuyer to do an create a machine learning model that predicts sale price based on house attributes. You now have a model that gives you $\mathbb{E}[Price \mid Attributes]$, the expected price given what we know about the house. You also have a specific house you're looking at, but _wow_ is it expensive. Why, what specifically about it makes the model think it is expensive? We can answer this question by getting a _unit-level feature explanation_.

### Population-level feature importance vs unit-level feature explanations

In a supervised ML system, we are trying to model $\mathbb{E}[y \mid X] = f(X)$. If we knew the true $f(X)$, it would be very useful! well we have an approximation to it at least. if the CV scores are good it may even be a useful approximation. f(X) - E[f(X)] tells us "does the model think this outcome is unusual". then we can look at which values of X move it further or closer from E[f(X)].

It's important not to confuse two similar but not identical concepts:
* Feature importance tells us something about which features the model uses in general. RF importance, regression coefficients, permutation, etc. "What columns are important in $f(X) = \mathbb{E}[y \mid X]$?" "What columns of $X$ are associated with changes in $f(X)$?" What in general is correlated with house price?
* Prediction explanations, which tell us which features are most useful in predicting this unit's outcome. SHAP, LIME. why is this house so much more expensive than average? "Which of the observed values of X_i explains why f(X_i) is far from $\mathbb{E}[f(X)]$? What makes this unit's predicted outcome different from the average unit's outcome?"

### What would a useful explanation? The case of linear models

linear models are so popular in part because they give us that nice regression summary at the population level. and we can get unit-level importances by looking at the largest coefficiencts. could we do something like that?

Why is that useful? (1) Linear changes are easy to imagine (2) It tells us about the unique importance of features, "all else held equal" (3) It's easy to compare the magnitude _and_ the sign of different ones

We can do this in python with **shap**, using the **permutation**

### Intuition behind solution, its interpretation

SHAP paper: https://arxiv.org/pdf/1705.07874.pdf

one-line description of how to interpret the shapley value for a unit

### The shap library

# Example: Examining predicted house price more closely

### Get the data, check shape. Fit model, do CV

Lets build our model of $\mathbb{E}[Price \mid Attributes] = f(X)$

```python
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame

input_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
target_variable = 'MedHouseVal'

X = data[input_features]
y = data[target_variable]

model = HistGradientBoostingRegressor()
# train an XGBoost model (but any other model type would also work)
model.fit(X, y);

# TODO: CV, or maybe just link to seychelles
```

Great. Now let's zoom in on house ID XXX.

### Call the shap library

compare $\hat{price}$ to $\bar{price}$

https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html

Do this one: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/labor_market_discrimination.html

```python
# build a Permutation explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Permutation(model.predict, X)
shap_values = explainer(X[:100])
```

`shap_values.feature_names`

what is the shape of shap_values?

### Interpret the quantities & plots that come out

https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html#shap.plots.waterfall 

```python
shap.plots.bar(shap_values)
```

```python
shap.plots.waterfall(shap_values[0])
```



```python
# Global summary
print(pd.DataFrame(shap_values_positive.values, columns=shap_values_positive.feature_names).describe())
```

# Other options

PDP?

# Appendix: ### Annotated key formulas from the paper

A linear explanation would be nice, ie

Eqn (1)

$$
\underbrace{g(z')}_\textrm{Predicted value at z'} =
\underbrace{\phi_0}_\textrm{Baseline prediction} +  
\underbrace{\sum_{i=1}^{M} \phi_i z'_i}_\textrm{Effect of feature i at point \ $z'_i$} 
$$

We expect $\phi_i to be large when including feature i causes \hat{y} to change

Eqn (4) 

$$

\underbrace{\phi_i}_\textrm{Effect of feature i} =  
\underbrace{\sum_{S \subseteq F \backslash \{i\}}}_\textrm{Sum over subsets of F without i}
\underbrace{ \frac{\mid S \mid ! (\mid F \mid - \mid S \mid - 1)!}{\mid F \mid!} (f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S))}_\textrm{Average change due to excluding feature i}

$$

it's pretty beastly to condense this, but really valuable

https://christophm.github.io/interpretable-ml-book/shapley.html#the-shapley-value-in-detail

| Symbol  | Meaning |
| ------------- | ------------- |
| $x$  | The unit to explain  |
| $F \backslash {i}$  | F without $i$  |
|S|Does not include $i$!!|
|$S 	\subseteq F \backslack {i}$ | All subsets of $F$ without $i$ |
|$f_{S \cup {i}$ | Model with $S$ and feature $i$ included |
|$x_{S \cup {i}$ | Unit with $S$ and feature $i$ included |

> The Shapley value can be misinterpreted. The Shapley value of a feature value is not the difference of the predicted value after removing the feature from the model training. The interpretation of the Shapley value is: Given the current set of feature values, the contribution of a feature value to the difference between the actual prediction and the mean prediction is the estimated Shapley value.

I think this isn't causal because of the table 2 fallacy - https://dagitty.net/learn/graphs/table2-fallacy.html

Coalitional intuition

# Appendix: Shap for classifiers

```python
import xgboost

import shap

# get a dataset on income prediction
X, y = shap.datasets.adult()


# train an XGBoost model (but any other model type would also work)
model = xgboost.XGBClassifier()
model.fit(X, y);

# build a Permutation explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Permutation(model.predict_proba, X)
shap_values = explainer(X[:100])

# get just the explanations for the positive class
shap_values_positive = shap_values[..., 1]

shap.plots.bar(shap_values_positive)

shap.plots.waterfall(shap_values_positive[0])

shap_values_positive.feature_names

# Global summary
print(pd.DataFrame(shap_values_positive.values, columns=shap_values_positive.feature_names).describe())
```

