http://e.sci.osaka-cu.ac.jp/yoshino/download/rw/

[https://github.com/jbrownlee/Datasets/blob/master/monthly-robberies.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-robberies.csv)

```python

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-robberies.csv')

t = list(df.index)
y = df['Robberies']
diff_y = y - y.shift(12)

def gen_iid_noise(size, sigma):
  return np.random.normal(0, sigma, size=size)

def gen_random_walk(size, sigma):
  return np.cumsum(np.random.normal(0, sigma, size=size))
  
for i in range(100):
  plt.plot(t[12:], gen_random_walk(len(t)-12, np.std(diff_y)), color='orange') # Analytical version: Expanding bands of something involving sigma and the square root of t
  
plt.plot(t, diff_y, color='blue')
plt.show()

for i in range(100):
  plt.plot(t[12:], gen_iid_noise(len(t)-12, np.std(diff_y)), color='orange') # Analytical version: constant bands of 2*sigma or so
  
plt.plot(t, diff_y, color='blue')
plt.show()

# Model comparison? Select model based on leave-one-out sequential CV/avg error

```
