Key question: Would collecting more data materially benefit my model, or has my chosen model hit the ceiling of potential performance?

```
curl http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip --output winequality.zip
unzip winequality.zip
cd winequality/
```

```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import learning_curve
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.api import formula as smf
from sklearn.utils import resample

df = pd.read_csv('winequality-red.csv', sep=';')

y = df['quality']
X = df.drop('quality', axis=1)

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

n_folds = 10

train_sizes, _, test_scores = learning_curve(Lasso(alpha=10**(-3), normalize=True), X, y, cv=n_folds, scoring='neg_root_mean_squared_error', train_sizes=np.linspace(0.1, 1, 20))
test_scores = -test_scores
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_se = np.std(test_scores, axis=1) / np.sqrt(n_folds)
test_scores_var = test_scores_se**2

plt.plot(train_sizes, test_scores_mean, marker='o')
plt.title('Learning Curve for Lasso model')
plt.xlabel('Sample size')
plt.ylabel('CV RMSE')
plt.tight_layout()
plt.show()

mean_diff = np.diff(test_scores_mean)
diff_se = np.sqrt(test_scores_var[1:] + test_scores_var[:-1])

diff_df = pd.DataFrame({'mean_diff': mean_diff, 
                        'n': train_sizes[1:]})

spline_fit = smf.wls('mean_diff ~ bs(n, df=3)', diff_df, weights=1./diff_se**2).fit() # Differing variances of observations

y_pred_df = spline_fit.get_prediction(diff_df).summary_frame(alpha=.05)

plt.scatter(diff_df['n'], diff_df['mean_diff'], label='Observed CV error')
plt.plot(diff_df['n'], y_pred_df['mean'], label='Smoothed error')
plt.fill_between(diff_df['n'], y_pred_df['mean_ci_lower'], y_pred_df['mean_ci_upper'], alpha=.1, color='blue', label='CI')
plt.axhline(0, linestyle='dotted')
plt.xlabel('Sample size')
plt.ylabel('Improvement in RMSE')
plt.tight_layout()
plt.legend()
plt.show()
```
