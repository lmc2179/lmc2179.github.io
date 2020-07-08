---
layout: post
title: "Picking the model with the lowest cross validation error is not enough"
author: "Louis Cialdella"
categories: posts
tags: [test]
image: cutting.jpg
---

*TL;DR - We often pick the model with the lowest CV error, but this leaves out valuable information. Specifically, it ignores the uncertainty around the estimated out-of-sample error. It's useful to calculate the standard errors of a CV score, and incorporate this uncertainty into our model selection process. Doing so avoids false precision in model selection and allows us to better balance out-of-sample error with other factors like model complexity.*

We build trust in our models by demonstrating that they make good predictions on out-of-sample data. This process, called cross validation, is at the heart of most model evaluation procedures. It allows us to easily compare the performance of any black-box models, though they may have differing structures or assumptions.
