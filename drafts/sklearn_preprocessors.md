?

# pipelines r good

# parts of a preprocessor

fit: take care of any state you need to track

transform: apply the change

A cool example: Statsmodels and sklearn

```python
class FormulaTransformer(BaseEstimator, TransformerMixin):
    # Adapted from https://juanitorduz.github.io/formula_transformer/
    def __init__(self, formula):
        self.formula = formula
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_formula = patsy.dmatrix(formula_like=self.formula, data=X)
        columns = X_formula.design_info.column_names
        return pd.DataFrame(X_formula, columns=columns)
```

link to formula transformer ^

# example: Replace a rare token with some value

```python

```
