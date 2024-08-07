
```python
import pandas as pd

df = pd.read_csv(r'C:\Users\louis\Downloads\louis_beer_data.csv')
df = df[~df['rating_score'].isna()] # Some accidental no-rates

counts = df['beer_type'].value_counts()
one_offs = set(counts[counts == 1].index)

df['beer_type_coarse'] = df['beer_type'].apply(lambda x: x if (x not in one_offs) else 'RARE')

# Generate mean estimates using OLS
from statsmodels.api import formula as smf

ols_model = smf.ols('rating_score ~ beer_type_coarse - 1', df)
ols_model_fitted = ols_model.fit()

mean_estimates = ols_model_fitted.params
low_95_estimates = ols_model_fitted.conf_int(alpha=.05)[0]
high_95_estimates = ols_model_fitted.conf_int(alpha=.05)[1]
se = ols_model_fitted.bse

ols_param_summary = pd.DataFrame({'Point': mean_estimates, 'Low': low_95_estimates, 'High': high_95_estimates, 'SE': se}).reset_index().rename({'index': 'Style'}, axis=1)

# Bayesian version
import bambi as bmb

partial_pooling_priors = {
    # "Intercept": bmb.Prior("Normal", mu=0, sigma=10),
    "1|beer_type_coarse": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "sigma": bmb.Prior("Exponential", lam=1),
}

partial_pooling_model = bmb.Model(
    formula="rating_score ~ 0 + (1|beer_type_coarse)", 
    data=df, 
    priors=partial_pooling_priors, 
    noncentered=False
)

partial_pooling_results = partial_pooling_model.fit(cores=1, chains=1) # Windows issue with multiprocessing

import arviz as az

az.plot_trace(partial_pooling_results) # Make sure it worked

bayes_model_param_summary = az.summary(partial_pooling_results, hdi_prob=0.95)
```


```python
import pandas as pd

df = pd.read_csv(r'C:\Users\louis\Downloads\louis_beer_data.csv')

BEER_NAME = 'beer_name'
BREWERY_NAME = 'brewery_name'
BEER_TYPE = 'beer_type'
BEER_ABV = 'beer_abv'
BEER_IBU = 'beer_ibu'
COMMENT = 'comment'
VENUE_NAME = 'venue_name'
VENUE_CITY = 'venue_city'
VENUE_STATE = 'venue_state'
VENUE_COUNTRY = 'venue_country'
VENUE_LAT = 'venue_lat'
VENUE_LNG = 'venue_lng'
RATING_SCORE = 'rating_score'
CREATED_AT = 'created_at'
CHECKIN_URL = 'checkin_url'
BEER_URL = 'beer_url'
BREWERY_URL = 'brewery_url'
BREWERY_COUNTRY = 'brewery_country'
BREWERY_CITY = 'brewery_city'
BREWERY_STATE = 'brewery_state'
FLAVOR_PROFILES = 'flavor_profiles'
PURCHASE_VENUE = 'purchase_venue'
SERVING_TYPE = 'serving_type'
CHECKIN_ID = 'checkin_id'
BID = 'bid'
BREWERY_ID = 'brewery_id'
PHOTO_URL = 'photo_url'
GLOBAL_RATING_SCORE = 'global_rating_score'
GLOBAL_WEIGHTED_RATING_SCORE = 'global_weighted_rating_score'
TAGGED_FRIENDS = 'tagged_friends'
TOTAL_TOASTS = 'total_toasts'
TOTAL_COMMENTS = 'total_comments'

df = df[~df[RATING_SCORE].isna()]

from matplotlib import pyplot as plt
import seaborn as sns

sns.regplot(x=df[BEER_IBU], y=df[RATING_SCORE], lowess=True)

import bambi as bmb

partial_pooling_priors = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=10),
    "1|beer_type": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "sigma": bmb.Prior("Exponential", lam=1),
}

partial_pooling_model = bmb.Model(
    formula="rating_score ~ 1 + (1|beer_type)", 
    data=df, 
    priors=partial_pooling_priors, 
    noncentered=False
)

partial_pooling_results = partial_pooling_model.fit(cores=1, chains=2, draws=1000) # Windows issue with multiprocessing

import arviz as az
az.plot_trace(partial_pooling_results)
print(az.summary(partial_pooling_results))
print(az.summary(partial_pooling_results).sort_values('mean'))
print(partial_pooling_results.posterior['Intercept'].shape)
print(partial_pooling_results.posterior['1|beer_type'].shape) # Chain x sample x Level

from statsmodels.api import formula as smf

print(smf.ols('rating_score ~ beer_type', df).fit().summary())

# Compare 1|beer_type[Brown Ale - English]  -0.012  0.294 in bambi
# beer_type[T.Brown Ale - English]  -1.51e-14  0.940 in smf ols

posterior_predictive = az.summary(partial_pooling_results.posterior_predictive)

plt.scatter(posterior_predictive['mean'], df[RATING_SCORE])

```
