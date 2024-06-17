I need the standard error of something and it's not a "standard" statistic

so you need the variance of the sampling distribution of some statistic



Plan

✅ Understand the theory - what is the relevant theorem? where does it originate?

From wikipedia - https://en.wikipedia.org/wiki/Variance#Arbitrary_functions

If X is a random  variable, then:

$Var[f(X)] \approx (f'(\mathbb{E}[X]))^2 Var[X]$

(use this to derive a confidence interval and a hypothesis test by applying it to the sampling distribution)

multivariate version - https://en.wikipedia.org/wiki/Delta_method#Multivariate_delta_method

$Var[f(X)] \approx\nabla f(X)^T Cov(X) \nabla f (X)$

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