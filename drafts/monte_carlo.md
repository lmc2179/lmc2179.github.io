I've measured some statististic, which I believe gives me useful information about something I want to understand

A quick refresher - sampling process, sample, statistic, sampling distribution

# One-sample analysis

## Put error bars around it

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html#scipy.stats.bootstrap

why the bootstrap works - is it magic? rubin link on modeling assumptions. intuitive description of bca and why it is a good idea

this is about sampling error. so  think carefully about what your sampling process is

honorable mention to the jackknife

## Calculate a P-value under some null hypothesis

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.monte_carlo_test.html

# Two-sample analyis: Is it significant?

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html#scipy.stats.permutation_test

Example: Did we move the left tail of the distribution. Statistic is the difference in the 5th percentile (a light version of integration from 0% to 5%)

WARNING: No power analysis
