?

# pipelines r good

idea: stateful transformations are basically part of your model, so you should fit/transform them the same way

https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

# parts of a preprocessor

fit: take care of any state you need to track

transform: apply the change

A cool example: Statsmodels and sklearn

fit: generate the patsy object

transform: patsify the design matrix

```python
import patsy

class FormulaTransformer(BaseEstimator, TransformerMixin):
    # Adapted from https://juanitorduz.github.io/formula_transformer/
    def __init__(self, formula):
        self.formula = formula
    
    def fit(self, X, y=None):
        dm = patsy.dmatrix(formula, X)
        self.design_info = dm.design_info
        return self
    
    def transform(self, X):
        X_formula = patsy.dmatrix(formula_like=self.formula, data=X)
        columns = X_formula.design_info.column_names
        X_formula = patsy.build_design_matrices([self.design_info], X, return_type='dataframe')
        return X_formula
```

link to formula transformer ^

A simple 2-matrix example with one categorical variable and one numeric variable

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
        counts = df[target_columns].value_counts()
        if self.min_count:
            rare_tokens = set(counts.index[counts >= min_count])
        if self.min_pct:
            pcts = df[target_columns].value_counts() / counts.sum()
            rare_tokens = set(pcts.index[pcts >= min_pct])
        self.rare_tokens = rare_tokens
        return self
    
    def transform(self, X):
        X_formula = patsy.dmatrix(formula_like=self.formula, data=X)
        columns = X_formula.design_info.column_names
        return pd.DataFrame(X_formula, columns=columns)

```

A simple 2-matrix example with one categorical variable and two rare values
