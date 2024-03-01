# Intuition behind SHAP and Python examples

Why is shap useful?
* Root cause analysis for an outlier
* Potential areas for unit-level intervention
* "Automatic case study"

imagine we actually knew the function f:X -> y, inspecting it would tell us interesting things 

f(X) - E[f(X)] tells us "does the model think this outcome is unusual". shap tells us "what about X makes the model think this outcome will be unusual"

well we have an approximation to it at least. if the CV scores are good it may even be a useful approximation

"why does the approximation to f think that f(X) is different from the average"

I think this isn't causal because of the table 2 fallacy - https://dagitty.net/learn/graphs/table2-fallacy.html

description is still handy though

https://arxiv.org/pdf/1705.07874.pdf

https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html

https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html#shap.plots.waterfall 

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
```
