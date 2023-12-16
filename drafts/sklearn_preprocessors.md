?

# pipelines r good

# parts of a preprocessor

fit: take care of any state you need to track

transform: apply the change

A cool example: Statsmodels and sklearn

fit: generate the patsy object

transform: patsify the design matrix

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

init: `target_column`, `min_pct`, `min_count`, `replacement_token`

fit: look at examples of `target_column` and find examples of tokens with less than `min_pct` or `min_count`

transform: look at the `target_column`, and 

```python
class RareTokenTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, min_pct=None, min_count=None, replacement_token='__RARE__'):
        self.target_column = target_column
        if (min_pct and min_count) or (not min_pct and not min_count):
            raise Exception("Please provide either min_pct or min_count, not both")
        self.replacement_token = replacement_token
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_formula = patsy.dmatrix(formula_like=self.formula, data=X)
        columns = X_formula.design_info.column_names
        return pd.DataFrame(X_formula, columns=columns)

```
