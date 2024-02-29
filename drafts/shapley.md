# Intuition behind SHAP and Python examples

https://arxiv.org/pdf/1705.07874.pdf

https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html

https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html#shap.plots.waterfall 

```python
def f(x):
    return knn.predict_proba(x)[:, 1]

med = X_train.median().values.reshape((1, X_train.shape[1]))

explainer = shap.Explainer(f, med)
shap_values = explainer(X_valid.iloc[0:1000, :])
Permutation explainer: 1001it [00:25, 38.69it/s]
shap.plots.waterfall(shap_values[0])
```
