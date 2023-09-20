Evaluating your regression model with a diagnostic

The regression sequel to the post about the confusion matrix - main diagonal vs off-diagonal

```
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame

# Select the input features and target variable
input_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
target_variable = 'MedHouseVal'

X = data[input_features]
y = data[target_variable]

# Initialize a linear regression model (you can choose a different model as needed)
model = DecisionTreeRegressor()

# Perform cross-validation predictions
predictions = cross_val_predict(model, X, y, cv=5)  # cv=5 for 5-fold cross-validation

from matplotlib import pyplot as plt
import seaborn as sns

plt.scatter(predictions, y)
x_y_line = np.array([min(predictions), max(predictions)])
plt.plot(x_y_line, x_y_line)
p = 0.35
plt.plot(x_y_line, x_y_line*(1+p))
plt.plot(x_y_line, x_y_line*(1-p))
plt.show()

within_triangle = sum((y*(1-p) < predictions) & (predictions < y*(1+p)))

print(within_triangle / len(y))
```
