

```bash
#https://archive.ics.uci.edu/ml/datasets/Credit+Approval
curl https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data > credit.csv
```

```python
import pandas as pd
from statsmodels.api import formula as smf

df = pd.read_csv('credit.csv')
df.columns = ['A{0}'.format(i) for i in range(1, 16)] + ['approved']
df['approved'] = df['approved'].apply(lambda x: 1 if x == '+' else 0) 

model = smf.logit('approved ~ A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11 + A12 + A13 + A14 + A15', df)
results = model.fit() # Ugh singular
```
