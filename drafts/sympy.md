---
layout: post
title: "Symbolic Calculus in Python: Simple Samples of Sympy"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: sympy.png
---

_My job seems to involve just enough calculus that I can't afford to forget it, but little enough that I always feel rusty when I need to do it. In those cases, I'm thankful to be able to check my work with [Sympy](https://www.sympy.org/en/index.html), a symbolic mathematics library in Python. Here are two examples of recent places I've used Sympy to do calculus. We'll start by computing the expected value of a distribution by doing a symbolic definite integral. Then, we'll find the maximum of a model by finding its partial derivatives symbolically, and setting it to zero._

# Symbolic Integration: Finding the moments of a probability distribution

A simple model for a continuous, non-negative random variable is a [half-normal distribution](https://en.wikipedia.org/wiki/Half-normal_distribution). This is implemented in scipy as [halfnorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfnorm.html), but that implementation differs from the one on Wikipedia (the PDFs are different). The `scipy` version is implemented in terms of a `scale` parameter which we'll call $s$. If we're going to use this distribution, there are a few questions we'd like to answer about it:
- What are the moments of this distribution? How do the mean and variance of the distribution depend on $s$?
- How might we estimate $s$ from some data?

Scipy lets us do all of these numerically (using functions like `mean()`, `var()`, and `fit(data)`).

$f(x) = \frac{1}{s} \sqrt{\frac{2}{\pi}} exp(\frac{-x^2}{2})$

[mean](https://en.wikipedia.org/wiki/Expected_value#Absolutely_continuous_case)

How would we estimate s hat from the data? [Method of Moments](https://en.wikipedia.org/wiki/Method_of_moments_(statistics))

Compute mu and var in terms of s by integrating, then rearrange

$\mu = \int_{0}^{\infty} x f(x) dx$

```python
import sympy as sm
from scipy.stats import halfnorm
import numpy as np
```

```python
x = sm.Symbol('x')
s = sm.Symbol('s')
```

```python
f = (sm.sqrt(2/sm.pi) * sm.exp(-x**2/2))/s
```

```python
mean = sm.integrate(x*f, (x, 0, sm.oo))

var = sm.integrate(((x-mean)**2)*f, (x, 0, sm.oo))
```

```python
print(mean.subs(s, 1).subs(sm.pi, np.pi).evalf(), halfnorm(scale=1).mean())
print(var.subs(s, 1).subs(sm.pi, np.pi).evalf(), halfnorm(scale=1).var())
```

```python
mu = sm.Symbol('mu')

s_in_terms_of_mu = sm.solve(mean-mu,s)[0]

s_when_mu_is_8 = s_in_terms_of_mu.subs(mu, 8).evalf()
```

# Symbolic Differentiation: Finding the maximum of a response surface model

what value of the inputs maximizes the output?

set partials df/dx and df/dy to zero, solve for x and y

```python
import sympy as sm
from sympy import init_printing

init_printing(use_unicode=False)

x, y, a, b_x_1, b_y_1, b_x_y, b_x_2, b_y_2 = sm.symbols('x y a b_x_1 b_y_1 b_x_y b_x_2 b_y_2')

f = a + b_x_1*x + b_y_1*y + b_x_y*x*y + b_x_2*x**2 + b_y_2*y**2 

result = sm.solve([sm.Eq(f.diff(var), 0) for var in [x, y]], [x, y])

result
```

Plot level curves for a specific quadratic, calculate point at max
