https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

Find the interesting units in your dataset - the extremes

Find out what a "normal" unit looks like

Predictive application: Calibrate level of error vs similarity to training set (???) OR identify when the distribution of X is changing

Anomaly detection application

Causal inference application: How much does the treated group look like the control group?

Can we use Shap with isoforest to understand why a particular unit is abnormal? **THAT** would be real cool

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.datasets import fetch_california_housing
from matplotlib import pyplot as plt
import seaborn as sns

california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame

input_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
target_variable = 'MedHouseVal'

X = data[input_features]
y = data[target_variable]

model = HistGradientBoostingRegressor()

y_cv = cross_val_predict(model, X, y)

iso_score_cv = cross_val_predict(IsolationForest(n_estimators=100), X, y, method='score_samples')

errors = y - y_cv

sns.regplot(x=iso_score_cv, y=errors, lowess=True) # Surprising - the two are uncorrelated

sns.distplot(iso_score_cv) # Which units are in the "tail" of the scores

print(np.quantile(iso_score_cv, .2))
```
