ESL section

Examples

- Anomaly detection
- Basket recommendation
- ~K~DE on integer discrete data

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

x = np.random.normal(0, 1, 5000)
x_sim = np.random.uniform(np.min(x), np.max(x), 5000)

X = np.concatenate((x, x_sim)).reshape(-1, 1)
y = [1]*5000 + [0]*5000

m = DecisionTreeClassifier(max_depth=3)
m.fit(X, y)

x_plot = np.linspace(np.min(x), np.max(x), 1000)
y_plot = m.predict_proba(x_plot.reshape(-1, 1))[:,1]

plt.plot(x_plot, y_plot)
sns.distplot(x, bins=np.linspace(np.min(x), np.max(x), 10), kde=False, norm_hist=True)
plt.show()

for _ in range(1000):
  m = DecisionTreeClassifier(max_depth=3)
  X_r, y_r = resample(X, y)
  m.fit(X_r, y_r)

  x_plot = np.linspace(np.min(x), np.max(x), 1000)
  y_plot = m.predict_proba(x_plot.reshape(-1, 1))[:,1]

  plt.plot(x_plot, y_plot, color='blue', alpha=.01)
sns.distplot(x, bins=np.linspace(np.min(x), np.max(x), 10), kde=False, norm_hist=True)
plt.show()
```
