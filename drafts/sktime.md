sktime

```python
import pandas as pd
import numpy as np
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.split import CutoffSplitter
from matplotlib import pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')

y = df['Passengers']

cv = CutoffSplitter(fh=np.arange(1, 12), # Predict tomorrow to twelve days out
                    cutoffs=[len(y) - 12], # Create a single train/test split, 12 from the end of the data set
                    window_length=36) # Training window

for train_index, test_index in cv.split(y):
    y_train, y_test = y[train_index], y[test_index]
    forecaster = NaiveForecaster(strategy='last')
    forecaster.fit(y_train)
    y_pred = forecaster.predict(np.arange(1, 12))
    plt.plot(y_pred)
    plt.plot(y_test)
```
