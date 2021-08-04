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

A simple model for a continuous, non-negative random variable is a [half-normal distribution](https://en.wikipedia.org/wiki/Half-normal_distribution). This is implemented in scipy as [halfnorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfnorm.html). The `scipy` version is implemented in terms of a `scale` parameter which we'll call $s$. If we're going to use this distribution, there are a few questions we'd like to answer about it:
- What are the moments of this distribution? How do the mean and variance of the distribution depend on $s$?
- How might we estimate $s$ from some data? If we knew the relationship between the first moment and $s$, we could use the [Method of Moments](https://en.wikipedia.org/wiki/Method_of_moments_(statistics)) for this univariate distribution.

Scipy lets us do all of these numerically (using functions like `mean()`, `var()`, and `fit(data)`). However, answering the above gives us some intuition about how the distribution behaves more generally, and could be the starting point for further analysis like computing the standard errors of $s$.

The scipy docs tell us that the PDF is:

$f(x) = \frac{1}{s} \sqrt{\frac{2}{\pi}} exp(\frac{-x^2}{2})$

Computing the [mean](https://en.wikipedia.org/wiki/Expected_value#Absolutely_continuous_case) of the distribution requires solving a definite integral:

$\mu = \int_{0}^{\infty} x f(x) dx$

Similarly, finding the [variance](https://en.wikipedia.org/wiki/Variance#Definition) requires doing some integration:

$\sigma^2 = \int_{0}^{\infty} (x - \mu)^2 f(x) dx$

We'll perform these integrals symbolically to learn how $s$ relates to the mean and variance. We'll then rearrange $s$ in terms of $\mu$ to get an estimating equation for $s$.

We'll import everything we need:

```python
import sympy as sm
from scipy.stats import halfnorm
import numpy as np
```

Variables which we can manipulate algebraically in Sympy are called "symbols". We can instantiate one at a time using `Symbol`, or a few at a time using `symbols:

```python
x = sm.Symbol('x', positive=True)
s = sm.Symbol('s', positive=True)

# x, s = sm.symbols('x s') # This works too
```

We'll specify the PDF of `scipy.halfnorm` as a fimctopm pf $x$ and $s$:

```python
f = (sm.sqrt(2/sm.pi) * sm.exp(-(x/s)**2/2))/s
```

It's now a simple task to symbolically compute the definite integrals defining the first and second moments. The first argument to `integrate` is the function to integrate, and the second is a tuple `(x, start, end)` defining the variable and range of integration. For an indefinite integral, the second argument is just the target variable.

```python
mean = sm.integrate(x*f, (x, 0, sm.oo))

var = sm.integrate(((x-mean)**2)*f, (x, 0, sm.oo))
```

Easy! You could use the [LOTUS](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician) to calculate the EV of any function of a random variable this way, if you wanted to.

Printing $sm.latex(mean)$ and $sm.latex(var)$, we see that:

$\mu = \frac{\sqrt{2} s}{\sqrt{\pi}}$

$\sigma^2 = - \frac{2 s^{2}}{\pi} + s^{2}$

?

```python
random_s = np.random.uniform(0, 10)

print(mean.subs(s, random_s).subs(sm.pi, np.pi).evalf(), halfnorm(scale=random_s).mean())
print(var.subs(s, random_s).subs(sm.pi, np.pi).evalf(), halfnorm(scale=random_s).var())
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

print(print(sm.latex(result[x])))
print(print(sm.latex(result[y])))
```

Plot level curves for a specific quadratic, calculate point at max
