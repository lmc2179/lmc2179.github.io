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
