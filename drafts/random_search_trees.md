"Don't use grid search, use random search"

drake meme, maybe with sklearn on drake's face

https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

do CV on some data with a decision tree

show that more predictive, less complex (# tree nodes) models result from random search over grid search, and that it happens faster

```python
import patsy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

class FormulaTransformer(BaseEstimator, TransformerMixin):
    # Adapted from https://juanitorduz.github.io/formula_transformer/
    def __init__(self, formula):
        self.formula = formula
    
    def fit(self, X, y=None):
        dm = patsy.dmatrix(self.formula, X)
        self.design_info = dm.design_info
        return self
    
    def transform(self, X):
        X_formula = patsy.dmatrix(formula_like=self.formula, data=X)
        columns = X_formula.design_info.column_names
        X_formula = patsy.build_design_matrices([self.design_info], X, return_type='dataframe')[0]
        return X_formula

# Assuming your data frame is named df, your feature names are in a list called feature_names,
# and your output variable name is in a variable called output_var_name
df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/openintro/resume.csv')

output_var_name = 'received_callback'

# Step 1: Prepare the data
X = df.drop(output_var_name, axis=1).fillna(0)  # Features
y = df[output_var_name]  # Output variable

# Step 2: Define the model
logistic_regression = make_pipeline(FormulaTransformer('job_city+job_industry+job_type+job_fed_contractor+job_equal_opp_employer+job_req_communication+race+gender+years_college+college_degree'), 
                                    LogisticRegression())

# Step 3: Set up the grid search
param_grid = {
    'logisticregression__C': [0.1, 1, 10, 100],  # Inverse of regularization strength
    'logisticregression__penalty': ['l1', 'l2'],  # Regularization penalty
    'logisticregression__solver': ['liblinear']  # Solver to use in the optimization problem
}

# Step 4: Cross-validation with grid search
grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=2)

# Step 5: Fit the model
grid_search.fit(X, y)

# Best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Output the results
print(f"Best parameters found: {best_params}")
print(f"Best estimator found: {best_model}")

# Predicting using the best model
y_pred_proba = best_model.predict_proba(X)[:, 1]

# Evaluate the model
roc_auc = roc_auc_score(y, y_pred_proba)
print(f"ROC-AUC of the best model: {roc_auc}")

# Look at the values we tried and the average result
searched_params = pd.DataFrame(grid_search.cv_results_['params'])
avg_roc_auc = grid_search.cv_results_['mean_test_score']
se_roc_auc = grid_search.cv_results_['std_test_score'] / np.sqrt(grid_search.cv)
```