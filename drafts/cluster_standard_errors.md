most of the time, we assume that our data is an IID sample from the population. this is a big deal for regression tools, and lots of causal inference is regression-y

Why should we believe your sampling process is IID? Often it isn't. Happens all the time that sampling wasn't done on individual units, but on clusters. for example, if you want to poll students in a school district, it might be easier to first pick schools, and then poll all the students within the selected school. cluster a/b tests for example. or maybe you do random sampling of units from the database, but those units are from clusters that are correlated intra-cluster.

Some examples I've seen:
* Treatment assignment to groups rather than individuals
* Students within schools
* Emails within users

difference in difference! https://www.theeffectbook.net/ch-DifferenceinDifference.html#how-is-it-performed-2

https://theeffectbook.net/ch-StatisticalAdjustment.html#your-standard-errors-are-probably-wrong

"The use of clustered standard errors can account for any sort of correlation between errors within each grouping." The Effect

https://en.wikipedia.org/wiki/Intraclass_correlation

adding a term removes bias; clustering gets you correct variance

when in doubt, think about the sampling process

```python
import numpy as np
import pandas as pd

cluster_sample_sizes = [100, 1, 1, 1, 1, 1, 10, 
                        10, 10, 10, 10, 10, 100, 
                        1000, 1, 1, 2, 5, 8, 10]
cluster_means = np.random.normal(0, 100, len(cluster_sample_sizes))
cluster_samples = [np.random.normal(m, 1, size=s) for m, s in zip(cluster_means, cluster_sample_sizes)]
pooled_samples = np.concatenate(cluster_samples)
cluster_labels = np.concatenate([[i]*s for i, s in enumerate(cluster_sample_sizes)])

df = pd.DataFrame({'cluster': cluster_labels, 'y': pooled_samples})

from statsmodels.api import formula as smf

model = smf.ols('y ~ 1', df).fit()

clustered_results = model.get_robustcov_results(cov_type='cluster', 
                                                groups=df['cluster'])

print(model.summary())

print(clustered_results.summary())

from sklearn.utils import resample

def cluster_bootstrap(data, cluster_col, statistic, n_iterations=1000):
    np.random.seed(42)
    clusters = data[cluster_col].unique()
    stats = []

    for _ in range(n_iterations):
        # Sample clusters with replacement
        sampled_clusters = resample(clusters, replace=True)
        # Create a bootstrap sample with the sampled clusters
        sampled_data = pd.concat([data[data[cluster_col] == cluster] for cluster in sampled_clusters])
        
        # Apply the model function to the bootstrap sample
        stat = statistic(sampled_data)
        stats.append(stat)

    return np.std(stats)

print(cluster_bootstrap(df, 'cluster', np.mean))
```