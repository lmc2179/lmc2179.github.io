---
layout: post
title: "Machine learning models for decision making in Python: Picking thresholds for asymmetric payoffs"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: decision.png
---

*Machine learning practitioners spend a lot of time thinking about whether thir model makes good predictions. But for ML to add value, its predictions need to be harnessed for decision making, not just prediction.*

# The point of machine learning is to make decisions

$$
\begin{bmatrix}
t & t\\ 
t & t
\end{bmatrix}
$$

# A prototypical example: Disease detection

Assume we've used our favorite library to build a model which predicts the probability that an individual has a malignant tumor based on some tests we ran. We're going to use this prediction to decide whether we want to refer the patient for a more detailed test, which is more accurate but more costly and invasive. Following tradition, we refer to the test data as $X$ and the estimated probability of a malignant tumor as $\hat{y}$. We think, based on cross-validation, that our model proves a well-calibrated estimate of $\mathbb{P}(Cancer \mid X) = \hat{y}$. For some particular patient, we run their test results ($X$) through our model, and compute their probability of a malignant tumor, $\hat{y}$. *Should we refer the patient for further, more accurate (but more invasive) testing?*

There are four possible outcomes of this process:
* We **refer** the patient for further testing, but the second test reveals the tumor is **benign**. This means our initial test provided a **_false positive (FP)_**.
* We **refer** the patient for further testing, and the second test reveals the tumor is **malignant**. This means our initial test provided a **_true positive (TP)_**.
* We **decline** to pursue further testing. Unknown to us, the second test would have shown the tumor is **benign**. This means our initial test provided a **_true negative (TN)_**.
* We **decline** to pursue further testing. Unknown to us, the second test would have shown the tumor is **malignant**. This means our initial test provided a **_false negative (FN)_**.
