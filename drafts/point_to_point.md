Key idea: Use the model + PDP + bootstrap to understand how changing some input column from x_0 to x_1, by looking at how $\hat{y_0} and \hat{y_1}$ are affected

$\Delta = \hat{y_1} - \hat{y_0}$

$\delta_y = \frac{\hat{y_1}}{\hat{y_0}} - 1$

$Elasticity = \frac{\delta_y}{\delta_x}$

https://en.wikipedia.org/wiki/Arc_elasticity

https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html

"Causal dependence plot" https://arxiv.org/pdf/2303.04209

```python
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:31:46 2024

@author: louis
"""

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


from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
from statsmodels.api import formula as smf
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/AER/HousePrices.csv')
df = df.sort_values('lotsize')

X, y = df.drop('price', axis=1), df['price']

model = make_pipeline(FormulaTransformer('lotsize + bedrooms + bathrooms + stories + garage + driveway + recreation + fullbase + gasheat + aircon + prefer'), 
                       RandomForestRegressor(n_estimators=100))

model.fit(X, y)

# Draw the PDP
lotsize_range = np.linspace(np.min(X['lotsize']), np.max(X['lotsize']))

pdp_values = []
for lotsize in lotsize_range:
  X_pdp = X.copy()
  X_pdp['lotsize'] = lotsize
  pdp_values.append(np.mean(model.predict(X_pdp)))

plt.plot(lotsize_range, pdp_values)
plt.ylabel('Predicted house price')
plt.xlabel('Lotsize')
plt.title('Partial dependence plot for Lotsize vs Price for Random Forest')
plt.show()

# Draw the PDP
aircon_range = ['no', 'yes']

pdp_values = []
for aircon in aircon_range:
  X_pdp = X.copy()
  X_pdp['aircon'] = aircon
  pdp_values.append(np.mean(model.predict(X_pdp)))

plt.plot(aircon_range, pdp_values)
plt.ylabel('Predicted house price')
plt.xlabel('Aircon')
plt.title('Partial dependence plot for aircon vs Price for Random Forest')
plt.show()
```
