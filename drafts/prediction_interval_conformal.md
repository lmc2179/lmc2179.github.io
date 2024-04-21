Take off your point prediction blinders: Conformal inference in Python with MAPIE

> All exact science is dominated by the idea of approximation. When a man tells you that he knows the exact truth about anything, you are safe in inferring that he is an inexact man.

Bertrand Russell

# You may be suffering from point prediction tunnel vision

Let's say you want to sell your house (also you own a house, congratulations), and you decide to build a model to predict what kind of price you might get for it. You do all the usual steps - you collect historical data, try a wide range of features, and perform model selection without a holdout set. Armed with a model that can tell you the expected value of your house, you run `model.predict(my_house_features)` and find out that your model predicts a value of...drum roll please...one _million_ dollars! Nice, you're going to be a millionaire! Well, maybe a sub-millionaire after all applicable taxes and legal fees, but still not bad.

Of course, you know enough about models to realize that your house won't sell for _exactly_ one million dollars and zero cents. It'll be a little more or a little less. How much more or less, though? If the answer is "a few dollars" more or less, great. If the answer is "a million dollars" more or less, then your forecast isn't actually saying much. Reviewing our basic machine learning, our model is really giving us the _expected house price given all the house's features_. That is, the model is a model of $\mathbb{E}[y \mid X]$, where $y$ is the house price and $X$ are the house features.

Quantifying the range of possible outcomes around the prediction is _really_ important. Despite that, lots of folks frequently take the point prediction made by a model as the gospel. I sometimes think of this as _point prediction tunnel vision_ - if we only consider the point prediction, we miss all the other possible cases we should also factor into our decision.

So what is a house seller to do? This is the familiar problem of producing a _prediction interval_. A couple of options present themselves:
* If your model is a linear regression, and your data follows the linear regression assumptions (homoskedasticity, gaussian noise, etc etc), then you can get a prediction interval by computing the size of the noise term $\hat{\sigma}$, [as we've seen before](https://lmc2179.github.io/posts/confidence_prediction.html).
* Alternatively, you could just dump your existing model and instead build a [quantile regression model](https://lmc2179.github.io/posts/quantreg_pi.html). 

Both of those are a little unsatisfying. Plenty of models are not linear models, and after all we went through all this work to build a model of $\mathbb{E}[y \mid X]$. Can't we use that somehow?

One tempting option is to try and build multiple models by bootstrapping, and looking at the distribution of predictions. However, that's not quite what we want, because the variation in the bootstrap samples isn't telling us what range of actual values this particular house might sell for. Rather, bootstrapping tells us about how uncertain we should be about the model's prediction of the expected value - it tells us how much uncertainty we should have around $\mathbb{E}[y \mid X]$. Bootstrapping gives us a _confidence interval_ of the conditional mean, not a _prediction interval_ of actual values we might observe.

Intuitively, lets think about what properties a good solution should have. 

# The key idea in conformal inference: the prediction interval contains all the points within "error distance" of the point prediction

Linear model intuition: if we knew the true model, the prediction interval would be about +- 2 \times \hat{\sigma}, which is the "usual" size of error, most errors are smaller than that

PIs in arbitrary spaces based where conformity=distance

Distance framing of conformal inference

1. Pick a distance measure in the Y-space
2. Generate OOS predictions
3. Look at how close predictions "usually" are to the actual values
4. Make a prediction; points which are the "usual" distance from the prediction are included in the PI

this actually _generalizes_ the PI

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

low_05, high_05 = y_pi_05[:,0], y_pi_05[:,1]

plt.scatter(y_test, y_pred)
plt.vlines(y_test, low_05, high_05)

plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], linestyle='dotted')

print('Coverage:', regression_coverage_score(y_test, low_05, high_05))
print('Average interval width:', regression_mean_width_score(low_05, high_05))
# Pick the model with acceptable coverage + lowest width
```

explain the input kwargs to MapieRegressor

CQR for heteroskedasticity

https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf

https://mapie.readthedocs.io/en/stable/examples_regression/4-tutorials/plot_cqr_tutorial.html

# Evaluating PI models

hit rate/coverage plot - we want coverage better than (1-\alpha) and width as low as possible. heuristic: plot coverage and width, pick model with lowest width that has coverage better than target

https://stats.stackexchange.com/questions/465799/testing-for-clairvoyance-or-performance-of-a-model-where-the-predictions-are-i/465808#465808

which is introduced in section 6.2 of https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

# What does a non-regression version look like?

Classification: Prediction set. Method = ???. Doc link = ???.

Times series: Prediction band. Method = EnbPI. Doc link = ???.

# Outro

Simulation and decision for conformal models

# Appendix - relevant papers

gentle introduction - https://arxiv.org/pdf/2107.07511.pdf

mapie - https://arxiv.org/abs/2207.12274
