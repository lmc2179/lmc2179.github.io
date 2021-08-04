---
layout: post
title: "Symbolic Calculus in Python: Simple Samples of Sympy"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: sympy.png
---



_My job seems to involve just enough calculus that I can't afford to forget it, but little enough that I always feel rusty when I need to do it. In those cases, I'm thankful to be able to check my work with [Sympy](https://www.sympy.org/en/index.html), a symbolic mathematics library in Python. Here are two examples of recent places I've used Sympy to do calculus._

# Symbolic Integration: Finding the moments of a probability distribution

Scipy halfnorm has one param - what are its moments? How does the mean and SD depend on s?

$f(x) = \sqrt{\frac{2}{\pi}} exp(\frac{-x^2}{2})$

How would we estimate s hat from the data? MoM

Compute mu in terms of s by integrating, then rearrange

$\mu = \int_{0}^{\infty} x f(x) dx$

```python
import sympy as sm
from scipy.stats import halfnorm

x = sm.Symbol('x')
s = sm.Symbol('s')

f = (sm.sqrt(2/pi) * sm.exp(-x**2/2))/s

mean = sm.integrate(x*f, (x, 0, sm.oo))

var = sm.integrate(((x-mean)**2)*f, (x, 0, sm.oo))

print(mean.subs(s, 1).subs(pi, np.pi).evalf(), halfnorm(scale=1).mean())
print(var.subs(s, 1).subs(pi, np.pi).evalf(), halfnorm(scale=1).var())

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
