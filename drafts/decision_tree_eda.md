idea: use a decision tree to reveal groups with similar X variables _and_ a similar average _y_ value. get the leaf ID

decision trees are useful analysis tools in their own right, not just an implementation detail of random forests

what is the regression.summary() for trees?

is this EDA? Arguably, eda is really broad. look at tukey's doc? I _think_ this is the right category, because we are looking to explore key subgroups without making any ironclad claims that this is the one true grouping 

is this clustering? well we are creating clusters, but it doesn not meet the textbook definition of clustering

ideas from tukey:
* his idea, that data = fit+residual, can be checked by looking at how much overlap there is in the resulting segments
* emphasis on visualization/numerical summaries

the idea of "key subgroups" is familiar but comes only indirectly from regression; regression is concerned with additive linear relationships

what if we make a marimekko diagram from the results??

how many subgroups? do cross-validation and look for the elbow

https://chatgpt.com/c/eba61cce-fc31-4cc3-8648-7fe95daa76b9

https://vincentarelbundock.github.io/Rdatasets/doc/AER/HousePrices.html

```python
import patsy
from sklearn.base import BaseEstimator, TransformerMixin

class FormulaTransformer(BaseEstimator, TransformerMixin):
    # Adapted from https://juanitorduz.github.io/formula_transformer/
    def __init__(self, formula):
        self.formula = formula
    
    def fit(self, X, y=None):
        dm = patsy.dmatrix(self.formula, X)
        self.design_info = dm.design_info
        return self
    
    def transform(self, X):
        X_formula = patsy.dmatrix(formula_like=self.formula, data=X)
        columns = X_formula.design_info.column_names
        X_formula = patsy.build_design_matrices([self.design_info], X, return_type='dataframe')[0]
        return X_formula


import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from matplotlib import pyplot as plt

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/AER/HousePrices.csv')

preproc = FormulaTransformer('lotsize + bedrooms + bathrooms + stories + driveway + recreation + gasheat + aircon + garage + prefer')

tree = DecisionTreeRegressor(max_leaf_nodes=5)

X = preproc.fit_transform(df)

tree.fit(X, df['price'])

df['group'] = tree.apply(X)

print(export_text(tree, feature_names=list(X.columns)))

plt.figure(figsize=(12,12))
plot_tree(tree, feature_names=list(X.columns), node_ids=True, filled=True)

from matplotlib import pyplot as plt
import seaborn as sns

for group_id, group_df in df.groupby('group'):
    plt.plot([group_id, group_id], group_df['price'].quantile([.1, .9]))
    print('Size of group:', len(group_df))
```
