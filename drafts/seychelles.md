Evaluating your regression model with a diagnostic

The regression sequel to the post about the confusion matrix - main diagonal vs off-diagonal

```
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame

input_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
target_variable = 'MedHouseVal'

X = data[input_features]
y = data[target_variable]
```

```
model = HistGradientBoostingRegressor()

predictions = cross_val_predict(model, X, y, cv=5)  # cv=5 for 5-fold cross-validation

from matplotlib import pyplot as plt
import seaborn as sns

sns.histplot(x=predictions, y=y)
x_y_line = np.array([min(predictions), max(predictions)])
plt.plot(x_y_line, x_y_line, label='Perfect accuracy')
p = 0.35 # Size of threshold
plt.plot(x_y_line, x_y_line*(1+p), label='Acceptable upper bound')
plt.plot(x_y_line, x_y_line*(1-p), label='Acceptable lower bound')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.legend()
plt.show()

# Within triangle calculation

within_triangle = sum((y*(1-p) < predictions) & (predictions < y*(1+p)))

print(round(100 * (within_triangle / len(y))), 2)
```
