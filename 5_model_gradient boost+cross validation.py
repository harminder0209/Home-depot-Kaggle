from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn import grid_search



def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.1
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)



df_train = pd.read_csv("C:\\Users\\Minal\\Desktop\\Harry\\dataset\\df_train_after_tf-idf.csv", encoding="ISO-8859-1")
df_test = pd.read_csv("C:\\Users\\Minal\\Desktop\\Harry\\dataset\\df_test_after_tf-idf.csv", encoding="ISO-8859-1")

df_train = df_train.drop(['brand'],axis=1)
df_test=df_test.drop(['brand'],axis=1)


id_test = df_test['id']
y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values



rfr = GradientBoostingRegressor(n_jobs = -1)

clf = Pipeline([('rfr', rfr)])
param_grid = {'n_estimators' : [375,400],
              'max_depth': [5,10],  
            }

model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 0, scoring=RMSE)


print("fitting starts")

model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

print("predicting starts")
y_pred = model.predict(X_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('C:\\Users\\Minal\\Desktop\\Harry\\output\\submission77_gradientboost+gridsearch.csv',index=False)