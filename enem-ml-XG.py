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


# GridSearch for XGBRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X,
         y)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

#Training the model
xgb1.fit(X, y)

# Predicting the Test set results
y_pred = xgb1.predict(df_test)

# Submission file
submission = pd.DataFrame({
        "NU_INSCRICAO": df_test_main["NU_INSCRICAO"],
        'NU_NOTA_MT': y_pred
        })

submission.to_csv('answer.csv', index = False)

