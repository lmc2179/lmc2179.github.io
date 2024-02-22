novelty effects in A/B testing are common

https://arxiv.org/pdf/2102.12893.pdf

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

n = 20

walk = np.cumsum(np.random.normal(0, 1, n)) + 10

y_c = walk + np.random.normal(0, 1, n)
y_t = y_c * 1.05 + np.random.normal(0, 1, n)

plt.plot(y_c)
plt.plot(y_t)
```
