import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression

df = pd.read_csv('train.csv', index_col=0)

print(type(df))

categoricals = []

## Pre-processing dataset

# 1. Process missing data

for col, col_type in df.dtypes.items():
    if col_type == 'O': # Check if column is of object type
        categoricals.append(col)
    else:
        # Column is numeric, fill with 0 if null
        df[col] = df[col].apply(lambda x: 0 if pd.isna(x) else x)

# 2. One-hot encode categoricals.

# dummy_na=True - create new category for NaN entries (e.g.: age_nan)
df_one = pd.get_dummies(df, columns=categoricals, dummy_na=True)


# 3. Modeling using LogisticRegresssion

target = 'Survived'

X = df_one.drop(columns=target)
y = df_one[target]


lr = LogisticRegression(
    C=1,
    class_weight=None,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    max_iter=100,
    multi_class='ovr',
    n_jobs=-1,
    penalty='l2',
    random_state=42,
    solver='liblinear',
    tol=0.0001,
    verbose=0,
    warm_start=False
)

lr.fit(X, y)

# Save columns - allows API to predict on inputs with less column than the dataset.
model_columns = list(X.columns)
joblib.dump(model_columns, 'lr_model_columns.pkl')
# Save our model. Both the persisted objects will be loaded during the API app startup.
joblib.dump(lr, 'linear_regression_model.pkl')

