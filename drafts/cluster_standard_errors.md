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