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
```
