Interpreting unit-level feature importance in Python with SHAP

# Problem - Explaining unit-level feature importance

### "Why is this customer so valuable?"

Building a machine learning model is a little bit like setting up a little robot advisor, who helps you figure out what to do next by giving you expert advice. For example, a model which predicts customer LTV based on customer attributes is like having an advisor who knows all about your customer base. The expert is able to answer the question: "How much do you think this customer is worth?". A reasonable follow up question for such an expert would be "Why do you think that?".

We want model transparency = "Why did the model predict this outcome for this unit?"

We want to explain anomalies/do RCA = "Why was this unit's outcome different from the average?"

Lets us my favorite example of a vital product that everyone needs: little tinny sunglasses for cats.

LTV example - Key question is "what makes this customer so valuable?. X = customer attributes measured at signup, y = LTV $.

### Unit vs population-level explanations

Population explanation = "What columns are important in $f(X) = \mathbb{E}[y \mid X]$?" "What columns of $X$ are associated with changes in $f(X)$?"

Unit explanation = "Which of the observed values of X_i explains why f(X_i) is far from $\mathbb{E}[f(X)]$? What makes this unit's predicted outcome different from the average unit's outcome?"

Why is are unit-level explanations useful?
* Root cause analysis for an outlier
* Potential areas for unit-level intervention
* "Automatic case study"
* Why did this get flagged as fraud (robocop examples)

imagine we actually knew the function f:X -> y, inspecting it would tell us interesting things 

f(X) - E[f(X)] tells us "does the model think this outcome is unusual". shap tells us "what about X makes the model think this outcome will be unusual"

well we have an approximation to it at least. if the CV scores are good it may even be a useful approximation

"why does the approximation to f think that f(X) is different from the average"

### What would a useful explanation? The case of linear models

linear models are so popular in part because they give us that nice regression summary at the population level. and we can get unit-level importances by looking at the largest coefficiencts. could we do something like that?

Why is that useful? (1) Linear changes are easy to imagine (2) It tells us about the unique importance of features, "all else held equal" (3) It's easy to compare the magnitude _and_ the sign of different ones

# Solution - Shap

### Intuition behind solution, its interpretation

SHAP paper: https://arxiv.org/pdf/1705.07874.pdf

### Annotated key formulas from the paper

Eqn (1)

Eqn (4) 

it's pretty beastly to condense this, but really valuable

https://christophm.github.io/interpretable-ml-book/shapley.html#the-shapley-value-in-detail

> The Shapley value can be misinterpreted. The Shapley value of a feature value is not the difference of the predicted value after removing the feature from the model training. The interpretation of the Shapley value is: Given the current set of feature values, the contribution of a feature value to the difference between the actual prediction and the mean prediction is the estimated Shapley value.


I think this isn't causal because of the table 2 fallacy - https://dagitty.net/learn/graphs/table2-fallacy.html

### The shap library

# Example: Examining customer LTV more closely

### Get the data, check shape. Fit model, do CV

```python
import xgboost

import shap

# get a dataset on income prediction
X, y = shap.datasets.adult()
```

### Call the shap library

https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html

Do this one: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/labor_market_discrimination.html

```python
# train an XGBoost model (but any other model type would also work)
model = xgboost.XGBClassifier()
model.fit(X, y);

# build a Permutation explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Permutation(model.predict_proba, X)
shap_values = explainer(X[:100])
```

### Interpret the quantities & plots that come out

https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html#shap.plots.waterfall 

```python
# get just the explanations for the positive class
shap_values_positive = shap_values[..., 1]

shap.plots.bar(shap_values_positive)

shap.plots.waterfall(shap_values_positive[0])

shap_values_positive.feature_names
```

```python
# Global summary
print(pd.DataFrame(shap_values_positive.values, columns=shap_values_positive.feature_names).describe())
```

# Other options

PDP?


