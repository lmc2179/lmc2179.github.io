# What is observational analysis

# Example:

Introduce the problem; the treatment, outcome, and confounders

Look at X and y for treat and control

https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Treatment.html

```python

import pandas as pd

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Treatment.csv')

covariate_columns = ['age', 'educ', 'ethn', 'married', 're74', 're75', 'u74', 'u75']
outcome_column = 're78'

X_c = df[df['treat']][covariate_columns]
y_c = df[df['treat']][outcome_column]

X_t = df[~df['treat']][covariate_columns]
y_t = df[~df['treat']][outcome_column]

```

# The idea of matching

http://biostat.jhsph.edu/~estuart/Stuart10.StatSci.pdf

Notions of similarity, distance

# Matching in Python

https://gist.github.com/lmc2179/7ae1dcc04ba95cccd8c118f25bd94e4f

```python

def mahalanobis_matrix(X_c, X_t):
    V = np.cov(X, rowvar=0)
    VI = np.linalg.inv(V)
    D = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(i, len(X)):
            D[i][j] = D[j][i] = mahalanobis(X[i], X[j], VI)
    return D

```

Need the VI matrix for all the data, then create the rectangular matrix

https://gist.github.com/lmc2179/d4bd1091821db7048bbca5f77b785a4c

Gower? ^^

# Checking the match quality

Distance distribution and spot checks

SMDs

marginal dist plots for external validity

# Causal considerations

dagitty, DAGs, backdoor

# Other matching methods

CEM, PSM; mention regression as matching
