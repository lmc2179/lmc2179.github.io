---
layout: post
title: "Is my regression model good enough to make decisions? Evaluating actual vs predicted plots and relative error of regression models"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: Flag_of_Seychelles.svg.png
---

We use predictive models as our advisors, helping us make better decisions using their output. A reasonable question, then, is "is my model accurate enough to be useful"? Every practitioner who uses ML methods is used to cross-validation, that beloved Swiss Army Knife of model validation. Anyone doing their due diligence when training a predictive model will try a few out, and select the one with the minimum Mean Squared Error, or perhaps use the [one standard error rule](https://lmc2179.github.io/posts/cvci.html). This doesn't usually answer our question, though. Model selection tells us which choice is the best among the available options, but it's unclear whether even the best one is actually good enough to be useful. I myself have had the frustrating experience of performing an in-depth model selection process, only to realize at the end that all my careful optimizing has given me a model which is better than the baseline, but still unusable for any practical purpose.

So, back to our questions. What does "accurate enough to be useful" mean, exactly? How do we know if we're they're?

We could try imposing a rule of thumb like "your MSE must be this small", but this seems to require context. Different tasks require different levels of precision in the real world - this is why dentists do not (except in extreme situations) use jackhammers, preferring tools with a diameter measured in millimeters.

Statistical measures of model or coefficient significance don't seem to help either; knowing that a given coefficient (or all of them) are statistically significantly different from zero is handy, but does not tell us that the model is ready for prime time. Even the legendary $R^2$ doesn't really have a clear a priori "threshold of good enough" (though surprisingly, I see to frequently run into people who are willing to do so). If you don't believe me, a perspective I found really helpful is Ch. 10 of Cosma Shalizi's [The Truth About Linear Regression](https://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf).

An actual viable method is to look at whether your prediction intervals are both practically precise enough for the task and also cover the data, an approach detailed [here](https://statisticsbyjim.com/regression/how-high-r-squared/). This is a perfectly sensible choice if your model provides you with an easy way to compute prediction intervals. However, if you're using something like scikit-learn you'll usually be creating just a single point estimate (ie, a single fitted model of $\mathbb{E}[y \mid X]$ which you can deploy), and it may not be easy to generate prediction intervals for your model.

The method that I've found most effective is to work with my stakeholders and try to determine **what size of relative (percent) error would be good enough for decision making**, and then see how often the model predictions meet that requirement. Usually, I ask a series of questions like:
* Imagine the model was totally accurate and precise, ie it hit the real value 100% of the time. What would that let us do? What value would that success bring us in terms of outcomes? Presumably, there is a clear answer here, and this would let us increase output, sell more products, or something else we want.
* Now imagine that the model's accuracy was off by a little bit, say 5%. Would you still be able to achieve the desired outcome?
* If so, what if it was 10%? 20%? How large could the error be and still allow you to achieve your desired outcome?
* Take this threshold, and consider every prediction within it to be a "hit", and everything else is a "miss". In that case, we can evaluate the model's practical usefulness by seeing how often it produces a hit.

This allows us to take our error measure, which is a continuous number, and discretize it. We could add more categories by defining what it means to have a "direct hit", a "near miss", a "bad miss" etc. You could then attach a predicted outcome to each of those discrete categories, and you've learned something not just about how the model makes **predictions**, but how it lets you make **decisions**. In this sense, it's the regression-oriented sequel to our previous discussion about [analyzing the confusion matrix](https://lmc2179.github.io/posts/decisions.html) for classifiers - we go from pure regression analysis to decision analysis using a diagnostic. The "direct hits" for a regression model are like landing in the main diagonal of the confusion matrix.

In a sense, this is a check of the model's "calibration quality". While I usually hear that term referring to [probability calibration](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html), I think it's relevant here too. In the regression setting, a model is "well calibrated" when its prediction are at or near the actual value. We'll plot the regression equivalent of the calibration curve, and highlight the region that counts as a good enough fit.

Let's do a quick example using this [dataset of California House Prices along with their attributes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). Imagine that you're planning on using this to figure out what the potential price of your house might be when you sell it; you want to know how much you might get for it so you can figure out how to budget for your other purchases. We'll use a Gradient Boosting model, but that's not especially important - whatever black-box method you're using should work.

First, lets get all our favorite toys out of the closet, grabbing our data and desired model:

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

In this context, the acceptable amount of error is probably dictated by how much money you have in the bank as a backup in case you get less for the house than you expected. For your purposes, you decide that a difference of 35% compared to the actual value would be too much additional cost for you to bear.

For reasons I can't really explain, I find it very amusing that this diagram looks like the flag of the Seychelles, and would look even more so if we added finer gradations of hit vs missed targets. We'll first come up with out-of-sample predictions using the cross validation function, and then we'll plot the actual vs predicted values along with the "good enough" region we want to hit.

```
predictions = cross_val_predict(model, X, y, cv=5)  # cv=5 for 5-fold cross-validation

from matplotlib import pyplot as plt
import seaborn as sns

x_y_line = np.array([min(predictions), max(predictions)])
p = 0.35 # Size of threshold, 35%

sns.histplot(x=predictions, y=y) # Plot the predicted vs actual values
plt.plot(x_y_line, x_y_line, label='Perfect accuracy', color='orange') # Plot the "perfect calibration" line
plt.fill_between(x_y_line, x_y_line*(1+p), x_y_line*(1-p), label='Acceptable error region', color='orange', alpha=.1) # Plot the "good enough" region
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.legend()
plt.show()
```

![image](https://github.com/lmc2179/lmc2179.github.io/assets/1301965/be70f215-01ab-438e-8afe-da66f01eb958)


In addition to a chart like this, it's also handy to define a numeric score - we could even use this for model selection, if we wanted to.

If we define $p$ as the acceptable percent change, we can compute the estimated _percent of predictions within acceptable error as_:

$$\text{Estimated probability of acceptable error} \\
= \frac{\text{Count of predictions within band}}{\text{Count of all predictions}}
= \frac{\sum_i I[y_i \times (1-p) \leq \hat{y}_i \leq y_i \times (1+p)]}{n}$$

To think about this from an engineering perspective, our use case defines the "tolerance", similar to the tolerance which is set in machining parts. This quantity tells us how often the product which our model produces (ie its output) is within the tolerance for error that we can handle.

```
# Within target region calculation

within_triangle = sum((y*(1-p) < predictions) & (predictions < y*(1+p)))

print(round(100 * (within_triangle / len(y))), 2)
```

That gives us 66% - a strong start, though there's probably room for improvement. It seems unlikely that we'd be willing to deploy this model as-is, and we'd want to improve performance by adding more features, more data, or improving the model design. However, even though this model is not usable currently, it's useful to now have an acceptance threshold for model quality that you can use with your stakeholders.

If we wanted to get a finer idea of how our decisions might play out, we could break the plot into more segments, like introducing regions for "near misses" or "catastrophic misses".

You could also probably analyze this with quantile regression, ie see how bad the 5% and 95% cases are
