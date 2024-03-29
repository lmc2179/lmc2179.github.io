# What is observational analysis

Potential outcomes; the potential-outcome ideal

Matching as a common-sense solution: fill in the other side of the potential outcome with a unit that looks like it but got the opposite treatment status

A bonus: this gives you an opportunity to get your hands dirty with the data and see how the coverage/validity looks up close



Stuart's list of steps


My list of steps:
0. Do causal diagramming to figure out the set of controls
1. Split the dataframe into treated/untreated outcome vectors/covariate matrices.
2. Compute distance matrix between treated and untreated units (we'll use the hamming distance)
3. apply a caliper: drop pairs which are too far apart to be plausible counterfactuals
4. run validity checks: what subgroup will my inferences apply over?
5. run univariate balance checks: on average, how similar are paired units in each column?
6. Compute the ATT

# Example:

Introduce the problem; the treatment, outcome, and confounders

Look at X and y for treat and control



```python
import pandas as pd

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ames.csv')

covariate_columns = ['area', 'Lot.Area', 'Yr.Sold', 'Mo.Sold', 'Year.Built', 
                     'Bldg.Type', 'House.Style', 'Overall.Qual', 'Overall.Cond', 
                     'Foundation', 'Kitchen.Qual', 'Heating', '']
outcome_column = 'price'
treatment_column = 'Fireplaces'

df = df[df['Fireplaces'].isin([0, 1])]

#Next line will apply:  # df[k] = np.digitize(df[k], v)
covar_bin_edges = {'area': [], 
                   'Lot.Area': []}

def split_treated_and_control(df, treatment_column, output_column, covariate_columns):
    treatment_indicator = df[treatment_column]

    y_c = df[~treatment_indicator][outcome_column]
    y_t = df[treatment_indicator][outcome_column]

    df = df[covariate_columns]

    X_c = df[~treatment_indicator].astype('float')

    X_t = df[treatment_indicator].astype('float')
    return
```

# The idea of matching

Stuart step 1: Closeness

http://biostat.jhsph.edu/~estuart/Stuart10.StatSci.pdf

Notions of similarity, distance

# Matching in Python

Stuart step 2: Do matching

https://gist.github.com/lmc2179/7ae1dcc04ba95cccd8c118f25bd94e4f

```python
from scipy.spatial.distance import cdist, squareform
from scipy.optimize import linear_sum_assignment
import numpy as np
    
D = cdist(X_c.values, X_t.values, 'hamming') # Double check hamming works here

c_pair, t_pair = linear_sum_assignment(D)
pair_distances = pd.Series([D[c][t] for c, t in zip(c_pair, t_pair)])
```

# Checking the match quality

Stuart step 3

Distance distribution and spot checks

```python
# Apply caliper
sns.kdeplot(pair_distances)
plt.show()

distance_acceptable = pair_distances <= 1

c_pair, t_pair = c_pair[distance_acceptable], t_pair[distance_acceptable]
```

spot check

```python
i = np.random.randint(0, len(c_pair))
print(X_c.iloc[c_pair[i]])
print(X_t.iloc[t_pair[i]])
```

Differences and SMDs; SMD of matched vs all data; do it for all columns
```python
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6351359/

age_diffs = (X_t.iloc[t_pair]['age'].values - X_c.iloc[c_pair]['age'].values)

print(np.sum((age_diffs <= 10) & (age_diffs >= -10))) # How often are the age differences "large"?

age_smd = (X_t.iloc[t_pair]['age'].values.mean() - X_c.iloc[c_pair]['age'].values.mean()) / np.sqrt((X_t.iloc[t_pair]['age'].values.var() + X_c.iloc[c_pair]['age'].values.var()) / 2)

sns.kdeplot(age_diffs)
plt.show()
```

marginal dist plots for external validity

```python
sns.kdeplot(X_t.iloc[t_pair]['age'].values, label='Treatment matches')
sns.kdeplot(X_c.iloc[c_pair]['age'].values, label='Control matches')
sns.kdeplot(df['age'].values, label='Full data set')

plt.show()

```

# Estimating the treatment effect

Stuart step 4

```python
from scipy.stats import sem, norm

y_t_pair = (y_t.iloc[t_pair].reset_index(drop=True))
y_c_pair = (y_c.iloc[c_pair].reset_index(drop=True))

match_diff = y_t_pair - y_c_pair
sns.kdeplot(match_diff)
plt.show()

diff = np.mean(match_diff)
se = sem(match_diff)

z = norm(0, 1).ppf(1.-(.01/2))

low = diff - z * se
high = diff + z * se

# Statistically significant different means; Not clear if practically significant (look at lower end of bound and high uncertainty)

# Intuition of heterogeneity: Size of effect vs age
sns.regplot(X_c.iloc[c_pair]['age'].values, match_diff)
```

# Causal considerations

What causal assumptions were embedded in the previous analysis

dagitty, DAGs, backdoor

# Other matching methods

CEM, PSM; mention regression as matching

https://arxiv.org/pdf/1707.06315.pdf

```
df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Treatment.csv')

from statsmodels.api import formula as smf

smf.ols(outcome_column + '~' + '+'.join(covariate_columns) + '+treat', df).fit().summary()
```

# Matching has many uses

* Preprocessing to understand covariate common support
* ATT estimation (what we did here)
* HTE model training
* Generating paired designs
* Paired analysis after full randomization
* Decomposing ATE by covariate (basically, another HTE idea)

# Appendix: Matching and variance reduction

Compare propagated SE with matched SE
