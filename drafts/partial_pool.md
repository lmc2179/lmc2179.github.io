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
  
data =  gen_data(10, [0,1,2,3,4], [1]*5)
print('Partial Pool mean')
print(alpha(data, 'y', 'grp'))
print('Unpooled mean')
print(data.groupby('grp').mean())
print('Grand mean')
print(data.groupby('grp').mean().mean())
```
