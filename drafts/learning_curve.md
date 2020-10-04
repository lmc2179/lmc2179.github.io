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

When we have a sample of data we want to use to fit a model, it's natural to ask ourselves whether we have "enough" samples on hand. Would collecting more data improve our model, or has it reached a performance plateau? This is a question with a lot of practical importance, because data is expensive!  We need to pay to acquire samples - either literally exchanging money with a data vendor or building systems to collect data. When our analysis is simple, like a difference in means, we can achieve this by computing the power of a hypothesis test or the precision of the effect size confidence interval. But if you have a carefully-crafted regression model or black-box machine learning model, it's a lot less clear how to gauge whether you have enough samples.

For example, let's say you've already collected [a number of datapoints about Portugese wine](http://www3.dsi.uminho.pt/pcortez/wine/). You'd like to build a predictive model that relates the measurable properties of wine to its quality, perhaps so you can sell it to a Portugese Winery to automate their quality assurance. You've just done some fancy [model selection](https://lmc2179.github.io/posts/cvci.html) and decided that a Lasso model will probably give you the best trade-off between prediction quality and model simplicity. You've already went through some effort to gather the 1,599 samples of wine data in your data set, but perhaps your model would make better predictions if you collected even more samples? After all, a more accurate model has a higher value, so this investment might make your model more valuable.

In order to answer this question, we can think about what information we wish we could have. The simplest thing that comes to my mind is the relationship between the sample size and the quality of the model - if we knew that, we could figure out if it's likely that more data would provide incremental value to your model's predictive ability.

# The learning curve tells us how model performance varies with sample size

The relationship between sample size and model quality has a name: the [learning curve](https://en.wikipedia.org/wiki/Learning_curve_(machine_learning)). 

We have some intuition about the shape of this curve. As the number of samples grows, the performance of the model usually improves rapidly and then "flattens out" until adding more data points have little effect.

Let's plot the learning curve for our portugese wine data set. As usually, we can have our old friend scikit-learn do all the hard work

# The incremental value of a data point

??

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
