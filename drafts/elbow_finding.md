---
layout: post
title: "Heuristics for knee finding in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: bad_day.png
---

# The knee finding problem is everywhere

The theme: How much to spend before diminishing returns

In ML, selecting the number of clusters in an unsupervised model is a classic example

It has also come up in my career in making business decisions, for example figuring out the amount to spend on an ad campaign before diminishing returns kick in

Finding out where the "tail" starts

lets say you have done some analysis which produced this curve

# Defining the knee: Curvature 

$\kappa = \frac{\mid f''(x) \mid}{(1 + \mid f'(x) \mid^2)^{3/2}}$

"Osculating circle" intuition

```python
import numpy as np
from scipy.stats import norm, t

x = np.linspace(-3, 3, 30)
y = norm(0, 1).cdf(x)

from matplotlib import pyplot as plt
import seaborn as sns

plt.scatter(x, y)
plt.show()
```

# Discrete curvature calculation

```python
first_derivative = np.gradient(y, x)
second_derivative = np.gradient(first_derivative, x)

curvature = np.abs(second_derivative) / (1 + np.abs(first_derivative)**2)**(3./2)

plt.scatter(x, curvature)
plt.show()
```

# Dealing with noise: Smoothing/Continuous curvature calculation with splines

```python
from scipy.interpolate import make_smoothing_spline

y_with_noise = y + norm(0, .05).rvs(len(y))

plt.scatter(x, y_with_noise)
plt.show()

spline = make_smoothing_spline(x, y_with_noise)

plt.scatter(x, y_with_noise)
plt.plot(x, spline(x))
plt.show()

curvature_fxn = lambda x: np.abs(spline.derivative(2)(x)) / (1 + np.abs(spline.derivative(1)(x))**2)**(3./2)

x_smooth = np.linspace(min(x), max(x), 100)

plt.plot(x_smooth, curvature_fxn(x_smooth))
plt.show()
```

# References

_Finding a “Kneedle” in a Haystack: Detecting Knee Points in System Behavior_ https://raghavan.usc.edu/papers/kneedle-simplex11.pdf

# Appendix: Osculating circle intuition

The curvature at a point is the inverse of the radius of the osculating circle

The osculating circle is tangent to the curve, and the radius points in the direction of the sign of the curvature (? I think)

Let's consider the point on the curve $(x, f(x))$. The vector which is tangent to the curve at this point is $(1, f'(x))$, so the normal vector is $(-f'(x), 1)$.

```python
import numpy as np
from scipy.stats import norm, t

x = np.linspace(-3, 3, 30)
f = lambda x: x**3
y = f(x)

from scipy.interpolate import make_smoothing_spline
from matplotlib import pyplot as plt

spline = make_smoothing_spline(x, y)

curvature_fxn = lambda x: np.abs(spline.derivative(2)(x)) / (1 + np.abs(spline.derivative(1)(x))**2)**(3./2)

radius_fxn = lambda x: 1./curvature_fxn(x)

plt.plot(x, y)

test_point = np.pi/2

plt.scatter([test_point], [f(test_point)])

direction = np.sign(spline.derivative(2)(test_point))

radius = radius_fxn(test_point)

plt.plot([test_point, test_point - (1./radius)*direction*spline.derivative(1)(test_point)], 
         [f(test_point), f(test_point) + (1./radius)*direction*1])
```