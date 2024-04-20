Take off your point prediction blinders: Conformal inference in Python with MAPIE

> All exact science is dominated by the idea of approximation. When a man tells you that he knows the exact truth about anything, you are safe in inferring that he is an inexact man.

Bertrand Russell

# You may be suffering from point prediction tunnel vision

E[y | X] or P(y | X) isn't the end of the story

Usually, using one number is false precision

It's useful to make predictions about the range of plausible outcomes, since the average alone hides a lot of information about the spread around it

Some examples:
* regression example: forecasting a range of revenues for next month's sales lets you prepare for both the best and worst cases.
* classification example: predicting a range of plausible diagnoses based on patient data lets you assess a range of treatment options and risks

There are a few ways to generate prediction intervals that have already been covered here:
* Linear models, but they make strong assumptions: https://lmc2179.github.io/posts/confidence_prediction.html
* Quantile regression, though that's also a linear model: https://lmc2179.github.io/posts/quantreg_pi.html

what if we have a black box model? we often do, all the cool ones are. we have tools like [PDPs](https://lmc2179.github.io/posts/pdp.html) for analyzing black box models, even computing CIs with bootstrapping, why not for making PIs from them

# The key idea in conformal inference: the prediction interval contains all the points within "error distance" of the point prediction

PIs in arbitrary spaces based where conformity=distance

Distance framing of conformal inference

1. Pick a distance measure in the Y-space
2. Generate OOS predictions
3. Look at how close predictions "usually" are to the actual values
4. Make a prediction; points which are the "usual" distance from the prediction are included in the PI

# Black box PIs with conformal inference

user choices (after MAPIE paper s2.1)

1. pick conformity score
2. How to generate the out-of-sample predictions (jackknife, CV, etc)
3. Risk level \alpha



# MAPIE example for regression: Training and prediction

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from mapie.regression import MapieRegressor
from mapie.metrics import regression_mean_width_score, regression_coverage_score


X, y = make_regression(n_samples=50, n_features=1, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

regressor = LinearRegression()

mapie_regressor = MapieRegressor(estimator=regressor, method='plus', cv=5)

mapie_regressor = mapie_regressor.fit(X_train, y_train)
y_pred, y_pi = mapie_regressor.predict(X_test, alpha=[0.05, 0.32]) 

# Shape of y_pis is n_rows x {low, high} x alphas
y_pi_05 = y_pi[:,:,0]

plt.scatter(y_test, y_pred)
plt.vlines(y_test, y_pi_05[:,0], y_pi_05[:,1])

plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], linestyle='dotted')

print('Coverage:', regression_coverage_score(y_test, y_pi_05[:,0], y_pi_05[:,1]))
print('Average interval width:', regression_mean_width_score(y_pi_05[:,0], y_pi_05[:,1]))
# Pick the model with acceptable coverage + lowest width
```

CQR for heteroskedasticity

https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf

https://mapie.readthedocs.io/en/stable/examples_regression/4-tutorials/plot_cqr_tutorial.html

# Evaluating PI models

hit rate/coverage

https://stats.stackexchange.com/questions/465799/testing-for-clairvoyance-or-performance-of-a-model-where-the-predictions-are-i/465808#465808

which is introduced in section 6.2 of https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf


# Appendix - relevant papers

gentle introduction - https://arxiv.org/pdf/2107.07511.pdf

mapie - https://arxiv.org/abs/2207.12274
