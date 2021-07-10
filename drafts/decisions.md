---
layout: post
title: "Machine learning models for decision making in Python: Picking thresholds for asymmetric payoffs"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: decision.png
---

*Machine learning practitioners spend a lot of time thinking about whether their model makes good predictions. But for ML to add value, its predictions need to be harnessed for decision making, not just prediction.*

# The point of machine learning is to make predictions that help us decisions

?

For example:
* Irrelevant content
* Criminal sentencing
* Fraudulent user ban
* Systems that diagnose disease 

1. Observe the newest instance we want to make a decision about.
2. Use our model to predict whether or not this instance belongs to the class we have targeted for action.
3. If it's likely that the instance

The default

What should the threshold be

# A prototypical example: Disease detection

Assume we've used our favorite library to build a model which predicts the probability that an individual has a malignant tumor based on some tests we ran. We're going to use this prediction to decide whether we want to refer the patient for a more detailed test, which is more accurate but more costly and invasive. Following tradition, we refer to the test data as $X$ and the estimated probability of a malignant tumor as $\hat{y}$. We think, based on cross-validation, that our model proves a well-calibrated estimate of $\mathbb{P}(Cancer \mid X) = \hat{y}$. For some particular patient, we run their test results ($X$) through our model, and compute their probability of a malignant tumor, $\hat{y}$. We've used our model to make a prediction, now comes the decision: *Should we refer the patient for further, more accurate (but more invasive) testing?*

There are four possible outcomes of this process:
* We **refer** the patient for further testing, but the second test reveals the tumor is **benign**. This means our initial test provided a **_false positive (FP)_**.
* We **refer** the patient for further testing, and the second test reveals the tumor is **malignant**. This means our initial test provided a **_true positive (TP)_**.
* We **decline** to pursue further testing. Unknown to us, the second test would have shown the tumor is **benign**. This means our initial test provided a **_true negative (TN)_**.
* We **decline** to pursue further testing. Unknown to us, the second test would have shown the tumor is **malignant**. This means our initial test provided a **_false negative (FN)_**.

We can group the outcomes into "bad" outcomes (false positives, false negatives), as well as "good" outcomes (true positives, true negatives). However, there's a small detail here we need to keep in mind - not all bad outcomes are equally bad. A false positive results in costly testing and psychological distress for the patient, which is certainly an outcome we'd like to avoid; however, a false negative results in an untreated cancer, posing a risk to the patient's life. There's an important **asymmetry** here, in that the cost of a FN is much larger than the cost of a FP.

Let's be really specific about the costs of each of these outcomes, by assigning a score to each. Specifically, we'll say:
* In the case of a True Negative (correctly detecting that there is no illness), nothing has really changed for the patient. Since this is the status quo case, we'll assign this outcome a score of 0.
* In the case of a True Positive (correctly detecting that there is illness), we've successfully found someone who needs treatment. While such therapies are notoriously challenging for those who endure them, this is a positive outcome for our system because we're improving the health of people. We'll assign this outcome a score of 1.
* In the case of a False Positive (referring for more testing, which will reveal no illness), we've incurred extra costs of testing and inflicted undue distress on the patient. This is a bad outcome, and we'll assign it a score of -1.
* In the case of a False Negative (failing to refer for testing, which would have revealed an illness), we've let a potentially deadly disease continue to grow. This is a bad outcome, but it's much worse than the previous one. We'll assign it a score of -100, reflecting our belief that it is about 100 times worse than a False Positive. 

We'll write each of these down in the form of a **payoff matrix**, which looks like this:

$$
P = 

\begin{bmatrix}
\text{TN value} & \text{FP value}\\ 
\text{FN value} & \text{TP value}
\end{bmatrix}

=

\begin{bmatrix}
0 & -1\\ 
-100 & 1
\end{bmatrix}
$$

The matrix here has the same format as the commonly used [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix). It is written (in this case) in unitless "utility" points which are relatively interpretable, but for some business problems we could write the matrix in dollars or another convenient unit. This particular matrix implies that a false negative is 100 times worse than a false positive, but that's based on nothing except my subjective opinion. Some amount of subjectivity (or if you prefer, "expert judgement") is usually required to set the values of this matrix, and the values are usually up for debate in any given use case.

We can now combine our estimate of malignancy probability ($\hat{y}$) with the payoff matrix to compute the expected value of both referring the patient for testing and declining future testing:

$$
\mathbb{E}[\text{Send for testing}] = \mathbb{P}(Cancer | X) \times \text{TP value} + (1 - \mathbb{P}(Cancer | X)) \times \text{FP value} \\
= \hat{y} \times 1 + (1 - \hat{y}) \times (-1)
= 2 \hat{y} - 1
$$

$$
\mathbb{E}[\text{Do not test}] = \mathbb{P}(Cancer | X) \times \text{FN value} + (1 - \mathbb{P}(Cancer | X)) \times \text{TN value} \\
= \hat{y} \times (-100) + (1 - \hat{y}) \times 0
= -100 \hat{y}
$$

What value of $\hat{y}$ is large enough that we should refer the patient for further testing? That is - what **threshold** should we use to turn the probabilistic output of our model into a decision? We want to send the patient for testing whenver $\mathbb{E}[\text{Send for testing}] \geq \mathbb{E}[\text{Do not test}]$. So we can set the two expected values equal, and find the point at whch they cross to get the threshold value, which we'll call $\hat{y}_*$:

$$2\hat{y}_* - 1 = -100 \hat{y}_*$$
$$\Rightarrow \hat{y}_* = \frac{1}{102}$$

So we should refer a patient for testing whenever $\hat{y} \geq \frac{1}{102}$. This is _very_ different than the aproach we would get if we used the default classifier threshold, which in scikit-learn is $\frac{1}{2}$.

# Picking the best threshold and evaluating out-of-sample decision-making in Python

We can do a little algebra to show that if we know the 2x2 payoff matrix, then the optimal threshold is:

$y_* = \frac{\text{TN value - FP value}}{\text{TP value + TN value - FP value - FN value}}$

Let's compute this threshold and apply it to the in-sample predictions in Python:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt

payoff = np.array([[0, -1], [-100, 1]])

X, y = load_breast_cancer(return_X_y=True)
y = 1-y # In the original dataset, 1 = Benign

model = LogisticRegression(max_iter=10000)
model.fit(X, y)

y_threshold = (payoff[0][0] - payoff[0][1]) / (payoff[0][0] + payoff[1][1] - payoff[0][1] - payoff[1][0])

send_for_testing = model.predict_proba(X)[:,1] >= y_threshold
```

Does the $y_*$ we computed lead to optimal decision making on this data set? Let's find out by computing the average out-of-sample payoff for each threshold:

```python
# Cross val - show that the theoretical threshold is the best one for this data

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(LogisticRegression(max_iter=10000), X, y, method='predict_proba')[:,1]

thresholds = np.linspace(0, .95, 1000)
avg_payoffs = []

for threshold in thresholds:
  cm = confusion_matrix(y, y_pred > threshold)
  avg_payoffs += [np.sum(cm * payoff) / np.sum(cm)]

plt.plot(thresholds, avg_payoffs)
plt.title('Effect of threshold on average payoff')
plt.axvline(y_threshold, color='orange', linestyle='dotted', label='Theoretically optimal threshold')
plt.xlabel('Threshold')
plt.ylabel('Average payoff')
plt.legend()
plt.show()
```
Our $y_*$ is very close to optimal on this data set. It is much better than the sklearn default of $\frac{1}{2}$.

Note that in the above example we calculate the out-of-sample confusion matrix `cm`, and estimate the average out-of-sample payoff as `np.sum(cm * payoff) / np.sum(cm)`. We could also use this as a metric for model selection, letting us directly select the model that makes the best decisions on average.

# Where do the numbers in the payoff matrix come from

These are decisions about your priorities, they don't come from the data set

Maybe in some cases you can determine them experimentally or observationally using a causal analysis

# When _is_ the optimal threshold $y_* = \frac{1}{2}?$

In the cancer example above, we may think it's more likely than not that the patient is healthy, yet still refer them for testing. Because the cost of a false negative is so large, the optimal behavior is to act conservatively, recommending testing in all but the most clear-cut cases.

How would things be different if our goal was simply to make our predictions as _accurate_ as possible? In this case we might imagine a payoff matrix like $\begin{bmatrix} 1 & 0\\  0 & 1 \end{bmatrix}$. For this payoff matrix, we are awarded a point for each correct prediction (TP or TN), and no points for incorrect predictions (FP or FN). IF we do the math for this payoff matrix, we see that $y_* = \frac{1}{2}$. That is, the default threshold of $\frac{1}{2}$ makes sense when we want to maximize the prediction accuracy, and there are no asymmetric payoffs. Other "accuracy-like" payoff matrices like $\begin{smallmatrix} 0 & -1\\  -1 & 0 \end{smallmatrix}$ and $\begin{smallmatrix} 1 & -1\\  -1 & 1 \end{smallmatrix}$ also have $y_* = \frac{1}{2}$.


# Wrapping it up: The short version

