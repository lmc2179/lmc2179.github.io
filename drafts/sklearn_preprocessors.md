---
layout: post
title: "Building your own sklearn transformer is easy and very useful"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: transformer.png
---

_Scikit-learn pipelines let you snap together transformations like Legos to make a Machine Learning model. The transformers included in the box with Sklearn are handy for anyone doing ML in Python, and practicing data scientists use them all the time. Even better, it's very easy to build your own transformer, and doing so unlocks a zillion opportunities to shape your data._

# Pipelines make model specification easy

Most of the time, ML models can't just suck in data from the world and spit predictions back out, whaterver overzealous marketers of the latest AI fad might tell you. Usually, you need a bit of careful sculpting of the input matrix in order to make sure it is usable by your favorite model. For example, you might do things like:

* Scale variables by [setting them from 0 to 1](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler) or [normalizing them](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer)
* Encoding non-numeric values as [one-hot vectors](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
* Generating [spline features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html#sklearn.preprocessing.SplineTransformer) for continues numeric values
* Running [some function](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) on the inputs values, like `sqrt(x)`

In Python, this process is eased quite a bit by the usage of [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), which let you chain together as many preprocessing steps as you like and then treat them like one big model. The idea here is that stateful transformations are basically part of your model, so you should fit/transform them the same way you do your model. The [FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) allows you to perform stateless transformations. In order to create a stateful transformations, you'll need to write your own Transformer class - but luckily, it's pretty easy once you have an idea of how to structure it.

# Anatomy of an Sklearn Transformer

Creating a subclass is as easy as inheriting from `BaseEstimator` and `TransformerMixin` and writing a couple of methods which might be familiar if you've been using scikit-learn already:
* `fit(X, y)`: This method takes care of any state you need to track. In the scaling example, this means computing the observed min and max of each feature, so we can scale inputs later.
* `transform(X)`: This method applies the change. In the scaling example, this means subtracting the min value and dividing by the max, both of which were stored previously.

For example, if you wanted to write a transformer that centered data by subtracting its mean (de-meaning it? that feels too mean), its `fit` and `transform` would do the following:
* `fit(X, y)`: Calculate the average of each column (ie, take the vector average of `X`).
* `transform(X)`: Subtract the stored average from the input vectors in `X`.

Lets take a look at a couple of examples that I've found useful in my work.

# An example: Replace a rare token in a column with some value

A common trick in dealing with categorical columns in ML models is to replace rare categories with a unique value that indicates "Other" or "This is a rare value". This kind of prepreocessing would be handy to have available as a transformer, so let's build one.

At init time, we'll take in parameters from the user:
* `target_column` - The column to scan
* `min_pct` - Values which appear in a smaller percentage of rows than this will be considered rare
* `min_count` - Values which appear in fewer rows than this will be considered rare. Mutually exclusive with the previous
* `replacement_token` - The token to convert rare values to.

We can sketch out the `fit` and `transform` methods:
* `fit(X, y)`: Look at examples of `target_column` and find examples of tokens with less than `min_pct` or `min_count`. Store them in the object's state.
* `transform(X)`: Look at the `target_column`, and replace all the known rare tokens with the replacement token.

Here's what that looks like in code as a transformer subclass:

```python
class RareTokenTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, min_pct=None, min_count=None, replacement_token='__RARE__'):
        self.target_column = target_column
        if (min_pct and min_count) or (not min_pct and not min_count):
            raise Exception("Please provide either min_pct or min_count, not both")
        self.min_pct = min_pct
        self.min_count = min_count
        self.replacement_token = replacement_token
    
    def fit(self, X, y=None):
        counts = X[self.target_column].value_counts()
        if self.min_count:
            rare_tokens = set(counts.index[counts <= self.min_count])
        if self.min_pct:
            pcts = X[self.target_column].value_counts() / counts.sum()
            rare_tokens = set(pcts.index[pcts <= self.min_pct])
        self.rare_tokens = rare_tokens
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.target_column] = X_copy[self.target_column].replace(self.rare_tokens, self.replacement_token)
        return X_copy
```

Let's try it on a real dataframe.

```python
X1 = pd.DataFrame({'numeric_col': [0, 1, 2, 3, 4], 'categorical_col': ['A', 'A', 'A', 'B', 'C']})
X2 = pd.DataFrame({'numeric_col': [0, 1, 2, 3, 4], 'categorical_col': ['C', 'A', 'B', 'A', 'A']})

t = RareTokenTransformer('categorical_col', min_pct=0.2)
t.fit(X1)
print(t.transform(X1).to_markdown())
print(t.transform(X2).to_markdown())
```

This gives us the expected `X1`:

|    |   numeric_col | categorical_col   |
|---:|--------------:|:------------------|
|  0 |             0 | A                 |
|  1 |             1 | A                 |
|  2 |             2 | A                 |
|  3 |             3 | __RARE__          |
|  4 |             4 | __RARE__          |

And `X2`:

|    |   numeric_col | categorical_col   |
|---:|--------------:|:------------------|
|  0 |             0 | __RARE__          |
|  1 |             1 | A                 |
|  2 |             2 | __RARE__          |
|  3 |             3 | A                 |
|  4 |             4 | A                 |


# A borrowed example: Combining patsy and sklearn

One of the few flaws of Scikit-learn is that it doesn't include out-of-the-box support for [Patsy](https://patsy.readthedocs.io/en/latest/). Patsy is a library that lets you easily specify design matrices with a single string. [Statsmodels](https://www.statsmodels.org/dev/example_formulas.html) allows you to fit models specified using Patsy strings, but Statsmodels only really covers generalized linear models. 

It would be really handy to be able to use scikit-learn models with Patsy. A [`FormulaTransformer` is implemented by Dr. Juan Camilo Orduz on his blog](https://juanitorduz.github.io/formula_transformer/) that does just that - I've borrowed his idea here and modified it to make it stateful.

This transformer will include the following `fit` and `transform` steps:
* `fit(X, y)`: Compute the `design_info` based on the specified formula and `X`. For example, Patsy needs to keep track of which columns are categorical and which are numeric.
* `transform(X)`: Run `patsy.dmatrix` using the `design_info` to generate the transformed version of `X`.

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
```

Lets take a look at how this transforms an actual dataframe. We'll use input matrices with one numeric and one categorical column. We'll square the numeric column, and one-hot encode the categorical one.

```python
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

X1 = pd.DataFrame({'numeric_col': [0, 1, 2], 'categorical_col': ['A', 'B', 'C']})
X2 = pd.DataFrame({'numeric_col': [0, 1, 2], 'categorical_col': ['C', 'A', 'B']})

t = FormulaTransformer('np.power(numeric_col, 2) + categorical_col - 1')
t.fit(X1)
print(t.transform(X1).to_markdown())
print(t.transform(X2).to_markdown())
```
This shows us what we expect, namely that `X1` is:

|    |   categorical_col[A] |   categorical_col[B] |   categorical_col[C] |   np.power(numeric_col, 2) |
|---:|---------------------:|---------------------:|---------------------:|---------------------------:|
|  0 |                    1 |                    0 |                    0 |                          0 |
|  1 |                    0 |                    1 |                    0 |                          1 |
|  2 |                    0 |                    0 |                    1 |                          4 |

And that `X2` is:


|    |   categorical_col[A] |   categorical_col[B] |   categorical_col[C] |   np.power(numeric_col, 2) |
|---:|---------------------:|---------------------:|---------------------:|---------------------------:|
|  0 |                    1 |                    0 |                    0 |                          0 |
|  1 |                    0 |                    1 |                    0 |                          1 |
|  2 |                    0 |                    0 |                    1 |                          4 |
