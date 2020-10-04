---
layout: post
title: "Would collecting more data improve my model's predictions? The learning curve and the value of incremental samples"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*Since we usually need to pay for data (either with money to buy it or effort to collect it), it's worth knowing the value of getting more data points to fit your predictive model. We'll explore the learning curve, a model-agnostic way of understanding how black box performance changes as we add more data points to our sample. Analysis of the learning curve tells us whether it's worth it to collect a larger dataset, and it's easy to do this analysis in Python with scikit-learn.*

Key question: Would collecting more data materially benefit my model, or has my chosen model hit the ceiling of potential performance?

# Is it worth collecting more samples?

# The learning curve tells us how model performance varies with sample size

# The incremental value of a data point

# Putting it all together: Computing the value of a larger sample

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
