I need the standard error of something and it's not a "standard" statistic

so you need the variance of the sampling distribution of some statistic

from https://bookdown.org/ts_robinson1994/10EconometricTheorems/dm.html

$\hat{\mu} = \frac{1}{n} \sum_i y_i$

by the CLT assuming n is "big"

$\hat{\mu} \sim N(\mu, SE(\mu)^2)$

then

$g(\hat{\mu}) \sim \approx N(g(\mu), g'(\mu)^2 \times SE(\mu)^2)$

check this against the wasserman version. write out what the confidence interval is (like wasserman does)

Plan
* Understand the theory - what is the relevant theorem? where does it originate?
* What is the proof? Something about a Taylor Series
* Find a good real-life univariate example. How 
* Do the math
* Sympy usage
* Numerical differentiation usage??
* What about the multivariate version? Maybe a simple example there too
* Write section 1: The common problem that the delta method solves, incl an example
* Write section 2: How to use it, in theory and with sympy/numerical analysis
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