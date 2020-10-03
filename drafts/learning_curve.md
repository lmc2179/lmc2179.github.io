Key question: Would collecting more data materially benefit my model?

```
curl http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip --output winequality.zip
unzip winequality.zip
cd winequality/
```

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('winequality-red.csv', sep=';')

y = df['quality']
X = df.drop('quality', axis=1)

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

n = int(len(X)*.89)
b = 200

n_folds = 20

train_sizes, _, test_scores = learning_curve(LinearRegression(), X, y, cv=n_folds, scoring='neg_root_mean_squared_error', train_sizes=np.linspace(0.1, 1, 20)) # , train_sizes=[n - 2*b, n - b, n]
test_scores = -test_scores
#test_scores -= test_scores[0]
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_se = np.std(test_scores, axis=1) / np.sqrt(n_folds)

plt.plot(train_sizes, test_scores_mean, marker='o')
plt.fill_between(train_sizes, test_scores_mean - 3*test_scores_se, test_scores_mean + 3*test_scores_se, alpha=.1) 
plt.show()

plt.plot(train_sizes[1:], np.diff(test_scores_mean), marker='o') # SE around first difference; will manually inspecting this tell us if we've "maxed out"?
plt.axhline(0, linestyle='dotted')
plt.show()
```
