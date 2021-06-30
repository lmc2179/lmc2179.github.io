---
layout: post
title: "Machine learning models for decision making in Python: Picking thresholds for asymmetric payoffs"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: decision.png
---

*Machine learning practitioners spend a lot of time thinking about whether their model makes good predictions. But for ML to add value, its predictions need to be harnessed for decision making, not just prediction.*

# The point of machine learning is to make decisions

$$
\begin{bmatrix}
t & t\\ 
t & t
\end{bmatrix}
$$

# A prototypical example: Disease detection

Assume we've used our favorite library to build a model which predicts the probability that an individual has a malignant tumor based on some tests we ran. We're going to use this prediction to decide whether we want to refer the patient for a more detailed test, which is more accurate but more costly and invasive. Following tradition, we refer to the test data as $X$ and the estimated probability of a malignant tumor as $\hat{y}$. We think, based on cross-validation, that our model proves a well-calibrated estimate of $\mathbb{P}(Cancer \mid X) = \hat{y}$. For some particular patient, we run their test results ($X$) through our model, and compute their probability of a malignant tumor, $\hat{y}$. We've used our model to make a prediction, now comes the decision: *Should we refer the patient for further, more accurate (but more invasive) testing?*

There are four possible outcomes of this process:
* We **refer** the patient for further testing, but the second test reveals the tumor is **benign**. This means our initial test provided a **_false positive (FP)_**.
* We **refer** the patient for further testing, and the second test reveals the tumor is **malignant**. This means our initial test provided a **_true positive (TP)_**.
* We **decline** to pursue further testing. Unknown to us, the second test would have shown the tumor is **benign**. This means our initial test provided a **_true negative (TN)_**.
* We **decline** to pursue further testing. Unknown to us, the second test would have shown the tumor is **malignant**. This means our initial test provided a **_false negative (FN)_**.

We can group the outcomes into "bad" outcomes (false positives, false negatives), as well as "good" outcomes (true positives, true negatives). However, there's a small detail here we need to keep in mind - not all bad outcomes are equally bad. A false positive results in costly testing and psychological distress for the patient, which is certainly an outcome we'd like to avoid; however, a false negative results in an untreated cancer, posing a risk to the patient's life. There's an important **asymmetry** here, in that the cost of a FN is much larger than the cost of a FP.

Let's be really specific about the costs of each of these outcomes. We'll write them down in the form of a **payoff matrix**, which looks like this:

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

The matrix here has the same format as the commonly used [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix). It is written (in this case) in unitless "utility" points which are relatively interpretable, but for some business problems we could write the matrix in dollars or another convenient unit. 

We can now combine our estimate of malignancy ($\hat{y}$) with the payoff matrix to compute the expected value of both referring the patient for testing and declining future testing:

$$
\mathbb{E}[Send for testing] = \mathbb{P}(Cancer | X) \times \text{TP value} + (1 - \mathbb{P}(Cancer | X)) \times \text{FP value} 
= \hat{y} \times 1 + (1 - \hat{y}) \times (-1)
$$
