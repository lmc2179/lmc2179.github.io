ESL section

Examples

- Anomaly detection
- Basket recommendation
- ~K~DE on integer discrete data

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

x = np.random.normal(0, 1, 5000)
x_sim = np.random.uniform(np.min(x), np.max(x), 5000)

X = np.concatenate((x, x_sim)).reshape(-1, 1)
y = [1]*5000 + [0]*5000

m = MLPClassifier(hidden_layer_sizes=(10,))
m.fit(X, y)

x_plot = np.linspace(np.min(x), np.max(x), 1000)
y_plot = m.predict_proba(x_plot.reshape(-1, 1))[:,1]

plt.plot(x_plot, y_plot)
sns.distplot(x, bins=np.linspace(np.min(x), np.max(x), 10), kde=False, norm_hist=True)
plt.show()
```
