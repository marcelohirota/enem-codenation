# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df_train_main= pd.read_csv('train.csv')
df_test_main= pd.read_csv('test.csv')

# Preparing datasets
var_x = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']
var_y = ['NU_NOTA_MT']

df_train_x = df_train_main[var_x]
df_train_y = df_train_main[var_y]
df_test = df_test_main[var_x]

# Replacing NaN values for mean values of each column
X=df_train_x.fillna(df_train_x.mean())
y=df_train_y.fillna(df_train_y.mean())
df_test=df_test.fillna(df_test.mean())


# Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting the Test set results
y_pred = regressor.predict(df_test)

# Submission file
submission = pd.DataFrame({
        "Id": data_test["Id"],
        'SalePrice': y_pred
        })

submission.to_csv('answer.csv', index = False)

