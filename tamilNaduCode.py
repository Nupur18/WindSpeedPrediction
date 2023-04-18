import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import xgboost as xgb
import lightgbm as lgb


tamilNadu2017 = pd.read_csv("Madurai, Tamil Nadu Data/05a00c8ece8a418ba8371627e9f31c32/3227257_9.93_78.14_2017.csv")
tamilNadu2018 = pd.read_csv("Madurai, Tamil Nadu Data/05a00c8ece8a418ba8371627e9f31c32/3227257_9.93_78.14_2018.csv")
tamilNadu2019 = pd.read_csv("Madurai, Tamil Nadu Data/05a00c8ece8a418ba8371627e9f31c32/3227257_9.93_78.14_2019.csv")

frames = [tamilNadu2017, tamilNadu2018, tamilNadu2019]
tamilNaduData = pd.concat(frames)

y = tamilNaduData["Wind Speed"]
x = tamilNaduData.drop(['Wind Speed'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=123)

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)



#Random Forest
print("-----------------Random Forest-----------------")
clf = RandomForestRegressor()
start = time.time()
clf.fit(X_train, y_train)
stop = time.time()

print(f"Training time: {stop - start}s")

joblib.dump(clf, 'tamilNaduFiles/randomForest.pkl')

clf_pred = clf.predict(X_test)

print("R2 score of test:", r2_score(y_test, clf_pred))

np.save('tamilNaduFiles/randomForestPred.npy', clf_pred)


print("-----------------Randomized Random Forest-----------------")
rf = RandomForestRegressor()

rf_params = {
    'n_estimators': [10, 20, 30],
    'max_features': ['sqrt', 0.5, 'auto'],
    'max_depth': [15,20,30,50],
    'min_samples_split': [2, 5, 10],
    "bootstrap":[True,False],
}

rf_random = RandomizedSearchCV(rf, rf_params, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

start = time.time()
rf_random.fit(X_train, y_train)
stop = time.time()

print(f"Training time: {stop - start}s")

joblib.dump(rf_random, 'tamilNaduFiles/randomForestRandomized.pkl')

print("Best parameters:", rf_random.best_params_)

pred = rf_random.predict(X_test)
print("R2 score of test:", r2_score(y_test, pred))

np.save('tamilNaduFiles/randomForestRandomizedPred.npy', pred)

#----------------------------------------------------------------------------------------------------------------------#

# Extra Tree
print("-----------------Extra Tree-----------------")

start = time.time()
reg = ExtraTreesRegressor(n_estimators=100).fit(X_train, y_train)
stop = time.time()

print(f"Training time: {stop - start}s")

joblib.dump(reg, 'tamilNaduFiles/extraTree.pkl')

pred = reg.predict(X_test)
print("R2 score of test:", r2_score(y_test, pred))

np.save('tamilNaduFiles/extraTreePred.npy', pred)


print("-----------------Randomized Extra Tree-----------------")

model = ExtraTreesRegressor()

param_grid = {
    'n_estimators': [100,200,300,400,500],
    'criterion': ['poisson', 'absolute_error', 'friedman_mse', 'squared_error'],
    'max_depth': [2,8,16,32,50],
    'min_samples_split': [2,4,6],
    'min_samples_leaf': [1,2],
    'max_features': [None,'sqrt','log2'],    
    'bootstrap': [True, False],
    'warm_start': [True, False],
}

xt_random = RandomizedSearchCV(model, param_grid, cv = 3, verbose=2, random_state=42, n_jobs = -1)

start = time.time()
xt_random.fit(X_train, y_train)
stop = time.time()

print(f"Training time: {stop - start}s")

joblib.dump(xt_random, 'tamilNaduFiles/extraTreeRandomized.pkl')

print("Best parameters:", xt_random.best_params_)

pred = xt_random.predict(X_test)
print("R2 score of test:", r2_score(y_test, pred))

np.save('tamilNaduFiles/extraTreeRandomizedPred.npy', pred)

#----------------------------------------------------------------------------------------------------------------------#

# XGBoost

print("-----------------XGBoost-----------------")

xgb_r = xgb.XGBRegressor (objective ='reg:linear' ,n_estimators = 10, seed = 123)

start = time.time()
xgb_r.fit(X_train, y_train)
stop = time.time()

print(f"Training time: {stop - start}s")

joblib.dump(xgb_r, 'tamilNaduFiles/xgboost.pkl')

pred = xgb_r.predict(X_test)
print("R2 score of test:", r2_score(y_test, pred))

np.save('tamilNaduFiles/xgboostPred.npy', pred)


print("-----------------Randomized XGBoost-----------------")

params = { 'max_depth': [3, 5, 6, 10, 15, 20],
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [100, 500, 1000]}

xgbr = xgb.XGBRegressor(seed = 123)

clf = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params,
                         scoring='neg_mean_squared_error',
                         n_iter=25,
                         verbose=2,
                         n_jobs=-1)

start = time.time()
clf.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")

joblib.dump(clf, 'tamilNaduFiles/xgboostRandomized.pkl')

print("Best parameters:", clf.best_params_)

pred = clf.predict(X_test)
print("R2 score of test:", r2_score(y_test, pred))

np.save('tamilNaduFiles/xgboostRandomizedPred.npy', pred)

#----------------------------------------------------------------------------------------------------------------------#


# LightGBM

print("-----------------LightGBM-----------------")

model = lgb.LGBMRegressor(objective='regression', random_state=101)

start = time.time()
model.fit(X_train, y_train)
stop = time.time()

print(f"Training time: {stop - start}s")

joblib.dump(model, 'tamilNaduFiles/lightGBM.pkl')

predicted_y = model.predict(X_test)
print("R2 score of test:", r2_score(y_test, predicted_y))

np.save('tamilNaduFiles/lightGBMPred.npy', predicted_y)


print("-----------------Randomized LightGBM-----------------")

param_grid = {
    'learning_rate': [0.1, 0.01],
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'subsample': [0.7, 0.8, 0.9],
    'min_child_samples': [1, 5, 10]
}

estimator = lgb.LGBMRegressor(objective='regression', random_state=101)

model = RandomizedSearchCV(estimator=estimator, 
                     param_distributions=param_grid,
                     cv=3, 
                     n_jobs=-1, 
                     scoring='neg_root_mean_squared_error',
                     n_iter=25,
                     verbose=2)

start = time.time()
model.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")

joblib.dump(model, 'tamilNaduFiles/lightGBMRandomized.pkl')

print("Best parameters:", model.best_params_)

predicted_y = model.predict(X_test)
print("R2 score of test:", r2_score(y_test, predicted_y))

np.save('tamilNaduFiles/lightGBMRandomizedPred.npy', predicted_y)
