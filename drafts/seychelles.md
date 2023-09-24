# Is my regression model "good enough"?

We use predictive models as our advisors, helping us make better decisions using their output. A reasonable question, then, is "is my model accurate enough to be useful"? Every practitioner who uses ML methods is used to cross-validation, that beloved Swiss Army Knife of model validation. Anyone doing their due diligence when training a predictive model will try a few out, and select the one with the minimum Mean Squared Error, or perhaps use the [one standard error rule](https://lmc2179.github.io/posts/cvci.html). This doesn't usually answer our question, though. Model selection tells us which choice is the best among the available options, but it's unclear whether even the best one is actually good enough to be useful. I myself have had the frustrating experience of performing an in-depth model selection process, only to realize at the end that all my careful optimizing has given me a model which is better than the baseline, but still unusable for any practical purpose.

So, back to our questions. What does "accurate enough to be useful" mean, exactly? 

We could try imposing a rule of thumb like "your MSE must be this small", but this seems to require context. Different tasks require different levels of precision in the real world - this is why dentists do not (except in extreme situations) use jack hammers, preferring tools with a diameter measured in millimeters. Relative measures don't seem to solve the problem either - a model with 5% relative error is preferrable to one with 10%, but we still need to know if either one is acceptable for real world use. 

Statistical measures of model or coefficient significance don't seem to help either; knowing that a given coefficient (or all of them) are statistically significantly different from zero is handy, but does not tell us that the model is ready for prime time. Even the legendary $R^2$ doesn't really have a clear a priori "threshold of good enough" (though surprisingly, I see to frequently run into people who are willing to do so). If you don't believe me, a perspective I found really helpful is Ch. 10 of Cosma Shalizi's [The Truth About Linear Regression](https://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf).

An actual viable method is to look at whether your prediction intervals are both practically precise enough for the task and also cover the data, an approach detailed [here](https://statisticsbyjim.com/regression/how-high-r-squared/). This is a perfectly sensible choice if your model provides you with an easy way to compute prediction intervals. However, if you're using something like scikit-learn you'll usually be creating just a single point estimate (ie, a single fitted model of $\mathbb{E}[y \mid X]$ which you can deploy), and it may not be easy to generate prediction intervals for your model.

Evaluating your regression model with a diagnostic

The regression sequel to the post about the confusion matrix - main diagonal vs off-diagonal

Idea: Tie size of miss to the resulting decision in a more interpretable way than MSE and friends. how much percent can your prediction be off by, and still let you make a good decision?

Define a big hit, big miss, small miss, etc

Let's do a quick example using this [dataset of California House Prices](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). Imagine that you're planning on using this to figure out what the potential price of your house might be when you sell it; you want to know how much you might get for it so you can figure out how to budget for your other purchases.

In this case it's a gradient booster but that doesn't really matter

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

model = HistGradientBoostingRegressor()
```

In this context, let's say that you're on a pretty tight budget, and you really need to get this number right in order to figure out what you cut. For your purposes, you decide that a difference of 10% compared to the actual value would be too much additional cost for you to bear.

the seychelles

```


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
```

In addition to a chart like this, it's also handy to define a numeric score (we coudl even use this for model selection)

Within-band percentage metrics

```
# Within triangle calculation

within_triangle = sum((y*(1-p) < predictions) & (predictions < y*(1+p)))

print(round(100 * (within_triangle / len(y))), 2)
```

Could compare score to each size of miss
