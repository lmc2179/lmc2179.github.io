---
layout: post
title: "Heuristics for knee finding in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: bad_day.png
---

# The "where does the curve level out" problem is all over the place

Lots of relationships have diminishing returns, where putting more raw material in the top gets less efficient as time goes on. <sup>[1](#foot1)</sup> For many data science practitioners, a familiar example is thinking about how many dimensions to retain when doing PCA or other reduction, but there are plenty of others.<sup>[2](#foot2)</sup> A cartoon version of this kind of relationship looks something like:

```python
from matplotlib import pyplot as plt
import numpy as np

plt.xkcd(scale=0.5)

x = np.linspace(0, 1, 10)
y = (1 / (1 + np.exp(-4*x))-0.5)

plt.title('Diminishing returns')
plt.plot(x, y, marker='o')
plt.xlabel('Cost')
plt.ylabel('Output')
plt.show()
```

This is https://en.wikipedia.org/wiki/Knee_of_a_curve

Our intuition is that we want to stop at the point of diminishing returns. In a diagram like the one above, it's the "knee" or "elbow", where the curve is the most bent before it straightens out again.

# Curvature

Cartoon: The one from before. But shade in the "flat"/"curved" regions.

we want to find the "most curved" point

curvature is given by

$\kappa = \frac{\mid f''(x) \mid}{(1 + \mid f'(x) \mid^2)^{3/2}}$

It's a function of the first and second derivatives at that point <sup>[3](#foot3)</sup>

Lets compare the shape of f and the curvature of f

Cartoon: previous one, plus curvature at x overlaid

The elbow/knee point is the point of max curvature

The intuition is ... <sup>[4](#foot4)</sup>

# Finite differences

Key idea: Finite difference allows estimate of $f'(x)$ locally. we are basically using the definition 

$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}$

use np.gradient

BUT if your observations of $f$ include any noise, for example if they are generated from cross-validation runs, they may need smoothing

# Using Splines: Smoothing

Smooth noise observations with splines, which allow easy calculation of the derivatives of the approximating spline

scipy version; CV fitting for spline hyperparams

we could just do finite diff now. derivative calculated from spline

# Other ideas

# Footnotes

<a name="foot1">1</a>: On a previous episode of _Casual Inference_, we talked about how we might model these as [log-log relationships](https://lmc2179.github.io/posts/isoelastic.html). We won't make strict parametric assumptions in this post, but you may also find that formulation useful.

<a name="foot2">2</a>: Some others:
* Dimensions vs goodness of fit in dimensionality reduction
* \# of clusters in k-means
* Marketing spend vs traffic

<a name="foot3">3</a>: Wait, isn't the second derivative the curvature??

<a name="foot4">4</a>: Osculating circle plot

# Draft

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

We have the point of interest $v = (x, f(x))$

Then we can get the tangent vector at this point, $t = (1, f'(x)) \times \frac{1}{\sqrt{1 + f'(x)^2}}$

We can then compute the normal vector

So if we know the curvature $\kappa$, then the osculating circle is centered at $c = v + n \kappa^{-1}$ and has radius $\kappa^{-1}$

(I think I still need to include the sign of the second derivative? set $s = sign(f''(x))$ and then it becomes $c = v + n \kappa^{-1} s$)

```python
import numpy as np
from scipy.stats import norm, t

x = np.linspace(-3, 3, 100)
f = lambda x: np.sin(x)**3
y = f(x)

from scipy.interpolate import make_smoothing_spline
from matplotlib import pyplot as plt

spline = make_smoothing_spline(x, y)

curvature_fxn = lambda x: np.abs(spline.derivative(2)(x)) / (1 + np.abs(spline.derivative(1)(x))**2)**(3./2)

radius_fxn = lambda x: 1./curvature_fxn(x)

plt.plot(x, y)

def plot_osculating_circle_of_test_point(test_point):
    v = np.array([test_point, spline(test_point)])

    plt.scatter(*zip(v))

    t = np.array([1, spline.derivative(1)(test_point)])
    t /= np.linalg.norm(t)

    n = np.array([-t[1], t[0]])

    radius = radius_fxn(test_point)

    s = np.sign(spline.derivative(2)(test_point))

    plt.plot(*zip(v, v + t*radius))
    plt.plot(*zip(v, v + n*radius*s))

    plt.ylim(-2, 2)
    plt.xlim(-3, 3)

    circle1 = plt.Circle(v + n*radius*s, radius, color='r', fill=False)
    plt.gca().add_patch(circle1)

plot_osculating_circle_of_test_point(np.pi/2)
plot_osculating_circle_of_test_point(-0.5)
```

# Appendix: Wait, isn't the second derivative the curvature?? Curvature vs Concavity

This was my initial reaction; you may be better informed than I am

the second derivative is "concavity", which is similar to but not the same as the curvature