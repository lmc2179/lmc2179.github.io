https://replit.com/@srs_moonlight/causal-discovery

https://lingam.readthedocs.io/en/stable/reference/ica_lingam.html

```python
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
import pandas as pd

import numpy as np

n = 1000

print('Generating data')
data = pd.DataFrame({'x': np.random.normal(0, 1, n)})
data['b'] = np.random.binomial(1, 0.5, size=n)
data['not_b'] = 1-data['b']
data['y'] = 1 + data['x'] + np.random.normal(0, .1, n) + data['b']*np.random.normal(0, .1, n)
data['z'] = 1 + data['y'] + np.random.normal(0, .1, n)
data['a'] = np.random.normal(0, 1, n)

# https://causal-learn.readthedocs.io/en/latest/search_methods_index/Score-based%20causal%20discovery%20methods/ExactSearch.html
d, _ = bic_exact_search(data.values) 

print(d)

for i in range(len(data.columns)):
    for j in range(len(data.columns)):
        if d[i][j] == 1:
            print(data.columns[i],'->', data.columns[j])
```

# What we do and why

## What is a dag and why have one

## How data tells us about the causal graph

Simple example of a DAG; intuition about how it tells us about the DAG structure

link shalizi here

## An example: housing prices

# How to do it

## Vars

## Constraints

## Inference

# Then what do you do with the graph

# Outline

https://github.com/py-why/causal-learn/blob/main/causallearn/search/ConstraintBased/PC.py

https://github.com/py-why/causal-learn/blob/main/tests/TestBackgroundKnowledge.py

https://github.com/py-why/causal-learn/blob/main/causallearn/utils/PCUtils/BackgroundKnowledge.py

to read: https://github.com/PacktPublishing/Causal-Inference-and-Discovery-in-Python?tab=readme-ov-file

https://arxiv.org/abs/2307.16405

https://github.com/py-why/dowhy/blob/main/docs/source/example_notebooks/dowhy_causal_discovery_example.ipynb

Big idea: (1) Learn graph from causal-learn, output as nx (2) Input to dowhy to get confounders, do estimates 
