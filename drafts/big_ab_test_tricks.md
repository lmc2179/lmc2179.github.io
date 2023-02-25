When sample sizes are _really_ big (millions of units), it's useful to analyze A/B tests from summaries on the data instead of the raw data. At that size, even $O(n)$ operations start to hurt. luckily we can ...

# Sample size big what you do now??

Examples: Japanese cat island marketing campaign

large sample, and also cat island has no wiki so you only have the summaries of a/b test results in North Cat Island and South Cat Island

There are two useful things to know about performing A/B test from summary statistics:
* Once you've constructed the appropriate data set; you can basically just do standard errors and treatment effects using some large-sample rules
* If you have two groups and you know their summaries, you can get the combined summaries of the groups without accessing the original unit-level data. This kind of "what happens if we cut by [attribute]" question is so common, that's it's useful to be able to speed it up. don't forget FDR and all that


# Computing treatment effects from summaries

the difference

$\Delta = m^T - m^C$

$SE(\Delta) = \sqrt{\frac{SE^T}{\sqrt{n^T}} + \frac{SE^C}{\sqrt{n^C}}}$

```python
def treatment_effect_with_se(m_t, v_t, n_t, m_c, v_c, n_c):
    effect = m_t - m_c
    se_t = np.sqrt(v_t) / np.sqrt(n_t)
    se_c = np.sqrt(v_c) / np.sqrt(n_c)
    se_effect = np.sqrt(se_t**2 + se_c**2)
    return effect, se_effect
```

the lift

$\delta = \frac{m^T}{m^C}$

$SE(\delta) = \frac{m^T}{m^C} * (\frac{SE^T^2}{m^T^2} + \frac{SE^C^2}{M^C^2})$

```python
def lift_with_se(m_t, v_t, n_t, m_c, v_c, n_c):
    lift = m_t / m_c - 1
    se_t = np.sqrt(v_t) / np.sqrt(n_t)
    se_c = np.sqrt(v_c) / np.sqrt(n_c)
    lift_se = np.sqrt((m_t**2 / m_c**2) * ((se_t**2 / m_t**2) + (se_c**2 / m_c**2)))
    return lift, lift_se
```

```
north_te, north_te_se = treatment_effect_with_se(north_treated_mean, 
                                                 north_treated_var, 
                                                 north_treated_n,
                                                 north_control_mean, 
                                                 north_control_var, 
                                                 north_control_n)

print('North treatment effect was: ', north_te, ' +- ', 1.96*north_te_se)
```


# Combining summary statistics wthout recalculation



combining means and combining variances

$m^{combined} = \frac{m^1 \times n^1 + m^2 \times n^2}{n^1 + n^2}$
$\sigma^{combined}^2 = \frac{\sigma^1^2 \times n^1 + \sigma^2^2 \times n^2}{n^1 + n^2}$

```python
def combine(m1, v1, n1, m2, v2, n2):
    n_new = n1 + n2
    m_new = (m1*n1 + m2*n2) / (n1 + n2)
    var_new = (v1*n1 + v2*n2) / (n1 + n2)
    return m_new, var_new, n_new
```

```
pooled_treated_mean, pooled_treated_var, pooled_treated_n \
    = combine(north_treated_mean, north_treated_var, north_treated_n,
                         south_treated_mean, south_treated_var, south_treated_n)
pooled_control_mean, pooled_control_var, pooled_control_n \
    = combine(north_control_mean, north_control_var, north_control_n,
                         south_control_mean, south_control_var, south_control_n)

pooled_lift, pooled_lift_se = lift_with_se(pooled_treated_mean, 
                                           pooled_treated_var, 
                                           pooled_treated_n,
                                           pooled_control_mean, 
                                           pooled_control_var, 
                                           pooled_control_n)

print('Pooled lift was: ', pooled_lift, ' +- ', 1.96*pooled_lift_se)
```


Note about bessel - https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups

Extra note about bayesian shit - https://izbicki.me/blog/gausian-distributions-are-monoids.html

# Outro: Serious likelihood function shit, isn't that neat

# Appendix: How the data was generated

```python
import numpy as np

n = 10000000
v = 1

north_base = 1.1
south_base = 1
north_lift = 1.2
south_lift = 1.25

north_treated = np.abs(np.random.normal(north_base*north_lift, v, size=n))
north_control = np.abs(np.random.normal(north_base, v, size=n))
south_treated = np.abs(np.random.normal(south_base*south_lift, v, size=n))
south_control = np.abs(np.random.normal(south_base, v, size=n))

north_treated_mean, north_treated_var, north_treated_n \
    = np.mean(north_treated), np.var(north_treated), len(north_treated)
north_control_mean, north_control_var, north_control_n \
    = np.mean(north_control), np.var(north_control), len(north_control)
south_treated_mean, south_treated_var, south_treated_n \
    = np.mean(south_treated), np.var(south_treated), len(south_treated)
south_control_mean, south_control_var, south_control_n \
    = np.mean(south_control), np.var(south_control), len(south_control)    
```
