# What is observational analysis

Potential outcomes; the experimental ideal

Matching as a common-sense solution

# Example:

Introduce the problem; the treatment, outcome, and confounders

Look at X and y for treat and control

https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Treatment.html

```python
import pandas as pd

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Treatment.csv')

#http://business.baylor.edu/scott_cunningham/teaching/lalonde-1986.pdf

covariate_columns = ['age', 'educ', 'ethn', 'married', 're74', 're75', 'u74', 'u75']
outcome_column = 're78'

treatment_indicator = df['treat']

y_c = df[treatment_indicator][outcome_column]
y_t = df[~treatment_indicator][outcome_column]

df = pd.get_dummies(df[covariate_columns])

X_c = df[treatment_indicator].astype('float')

X_t = df[~treatment_indicator].astype('float')
```

this is the ATC, weirdly enough

# The idea of matching

Stuart step 1: Closeness

http://biostat.jhsph.edu/~estuart/Stuart10.StatSci.pdf

Notions of similarity, distance

# Matching in Python

Stuart step 2: Do matching

https://gist.github.com/lmc2179/7ae1dcc04ba95cccd8c118f25bd94e4f

```python
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment

def mahalanobis_matrix(X_c, X_t):
    V = np.cov(np.concatenate((X_c, X_t)), rowvar=0)
    VI = np.linalg.inv(V)
    D = np.zeros((len(X_c),len(X_t)))
    for i in range(len(X_c)):
        for j in range(len(X_t)):
            D[i][j] = mahalanobis(X_c[i], X_t[j], VI)
    return D
    
D = mahalanobis_matrix(X_c.values, X_t.values)

D[np.isnan(D)] = np.inf # Hack to get rid of nans, where are they coming from

c_pair, t_pair = linear_sum_assignment(D)
pair_distances = [D[c][t] for c, t in zip(c_pair, t_pair)]
```

Compare with Gower (https://gist.github.com/lmc2179/d4bd1091821db7048bbca5f77b785a4c), Exact similarity (or almost-exact matching)

# Checking the match quality

Stuart step 3

Distance distribution and spot checks

Differences and SMDs
```python
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6351359/

age_diffs = (X_t.iloc[t_pair]['age'].values - X_c.iloc[c_pair]['age'].values)

print(np.sum((age_diffs <= 10) & (age_diffs >= -10)))

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

?

# Causal considerations

What causal assumptions were embedded in the previous analysis

dagitty, DAGs, backdoor

# Other matching methods

CEM, PSM; mention regression as matching

https://arxiv.org/pdf/1707.06315.pdf
