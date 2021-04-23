Demonstrating an estimator by simulation 
Let $\hat{\theta}$ be an estimator for $\theta$.  
In each simulation: 
- Generate data $X$. 
- Compute $\hat{\theta}$. 
- Compute $\hat{SE}(\hat{\theta})$ 

Then: 
- Calculate the error ($\hat{\theta}$ -  $\theta$) 
- Check avg error = 0 (Unbiasedness)
- Estimate MSE 
- Calculate the observed variance $\hat{s}^2_{\hat{\theta}}$ 
- And the standard error...uh, error ($\hat{SE}(\hat{\theta})^2$ - $\hat{s}^2_{\hat{\theta}}$)
- The Standard Error error should be about zero 

For multivariate metrics, the total squared error is $\sum_i (\hat{\theta} -  \theta)^2$, the quadratic (L2?) loss
Calculate the per-parameter coverage rates, plus the "Family-Wise" coverage rate

```
from scipy.stats import norm
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

def gen_data(n, means, sds):
  samples = np.concatenate([norm(m, sd).rvs(n) for m, sd in zip(means, sds)])
  grps = np.concatenate([[i]*n for i, _ in enumerate(means)])
  return pd.DataFrame({'y': samples, 'grp': grps})

def partial_pool_mean(df, y_col, g_col):
  gb = df.groupby(g_col)
  grp_means = gb.mean()[y_col]
  grand_mean = np.mean(grp_means)
  grp_vars = gb.var()[y_col]
  grand_var = np.var(grp_means)
  n = gb.apply(lambda d: len(d))
  num = (n/grp_vars)*grp_means + (1./grand_var)*grand_mean
  den = (n/grp_vars) + (1./grand_var)
  return num/den

k = 10
TRUE_MEAN = np.arange(k)
TRUE_SD = np.array([100]*k)
  
data =  gen_data(10, TRUE_MEAN, TRUE_SD)
print('Partial Pool mean')
print(alpha(data, 'y', 'grp'))
print('Unpooled mean')
print(data.groupby('grp').mean())
print('Grand mean')
print(data.groupby('grp').mean().mean())

n_sim = 1000

pp_mean = []
up_mean = []

for _ in tqdm(range(n_sim)):
  data =  gen_data(10, TRUE_MEAN, TRUE_SD)
  pp_mean.append(alpha(data, 'y', 'grp'))
  up_mean.append(data.groupby('grp').mean()['y'])
  
pp_mean = np.array(pp_mean)
up_mean = np.array(up_mean)

for i, m in enumerate(TRUE_MEAN):
  sns.distplot(pp_mean[:,i])
  sns.distplot(up_mean[:,i])
  plt.axvline(m)
plt.show()

pp_error = pp_mean - TRUE_MEAN
up_error = up_mean - TRUE_MEAN

sns.distplot(np.mean(pp_error**2, axis=1))
sns.distplot(np.mean(up_error**2, axis=1))

plt.show()
```
