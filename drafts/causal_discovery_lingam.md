
```python
import lingam
from lingam.utils import make_dot
import pandas as pd
import numpy as np

n = 1000

print('Generating data')
data = pd.DataFrame({'x': np.random.normal(0, 1, n)})
data['y'] = 1 + data['x'] + np.random.normal(0, .1, n)
data['z'] = 1 + data['y'] + np.random.normal(0, .1, n)
data['a'] = np.random.normal(0, 1, n)

print('Fitting LinGAM model')
model = lingam.DirectLiNGAM()
model.fit(data)

print(model.causal_order_)

print(model.adjacency_matrix_)

print(model.get_error_independence_p_values(data))

make_dot(model.adjacency_matrix_)
```
