
A step-by-step guide to propagating error using the Delta Method

# Error Propagation is Everywhere

Little sapling data scientists, coming up in the Garden of Analysis, learn early on that their best friend is the **sample mean**:

$$\hat{\mu} = \frac{1}{n} \sum_{i=1}^{i=n} y_i$$

They also learn that statements about the sample mean are most useful if you calculate the **standard error of the sample mean**

$$\hat{SE}(\hat{\mu})= \frac{\sigma}{\sqrt{n}}$$

This calculation is at the heart of many practical applications, especially, clinical trials and A/B tests.

There are lots of situations which go beyond this, and include **functions of the mean**. For experimentation especially, we often learn about the special case of the **difference in means**. But we can easily find others:

* Lift, or ratios
* Multiply by a constant (forex, yield)
* Retention <-> Churn
* We estimated an increase of \Delta in the input; how does it translate to a delta on the output if we know the production function
* Output of yearly revenue is a combination of many inputs

These are hard because we know the SE of inputs, but not the outputs. If you transformation has a nice form (and a 1st derivative), the delta method lets you propagate the error from the ones you know, to the ones you don't.

# How it works - Univariate

# Alternatives to the delta method

Jackknife/bootstrap

Transform and OLS

Delta method relies on more assumptions but is much faster than the bootstrap

---------------------------
---------------------------
---------------------------
---------------------------
---------------------------
---------------------------

# Draft

I need the standard error of something and it's not a "standard" statistic

so you need the variance of the sampling distribution of some statistic



Plan

✅ Understand the theory - what is the relevant theorem? where does it originate?

From wikipedia - https://en.wikipedia.org/wiki/Variance#Arbitrary_functions

If X is a random  variable, then:

$Var[f(X)] \approx (f'(\mathbb{E}[X]))^2 Var[X]$

(use this to derive a confidence interval and a hypothesis test by applying it to the sampling distribution)

multivariate version - https://en.wikipedia.org/wiki/Delta_method#Multivariate_delta_method

Double check but I think this _should_ work for X is multivariate, which becomes some scalar. like the ratio

$Var[f(X)] \approx\nabla f(X)^T Cov(X) \nabla f (X)$

$(1 \times k) (k \times k) (k \times 1)$

since it's a pretty elementary application of the taylor series, it was known sometime in the early 20th century

* Find a good real-life univariate example. maybe estimates of raw product and then production curve of finished product
* Do the math
* Sympy usage
* Numerical differentiation usage??
* What about the multivariate version? Maybe a simple example there too
* Write section 1: The common problem that the delta method solves, incl an example
* Write section 2: How to use it, in theory and with sympy/numerical analysis. Do simulation and confirm with bootstrap
* Write section 3: What about the multivariate case?
*  Other ways

Output
* When to use it
* How to use it
* A minimum of theory needed to understand it
* A clear worked example

Sources
* https://bookdown.org/ts_robinson1994/10EconometricTheorems/dm.html
* https://www.alexstephenson.me/post/2022-04-02-standard-errors-and-the-delta-method/
* https://en.wikipedia.org/wiki/Delta_method
* https://web.archive.org/web/20150525191234/http://data.math.au.dk/courses/advsimmethod/Fall05/notes/1209.pdf 
* https://egrcc.github.io/docs/math/all-of-statistics.pdf - 9.9, 9.10