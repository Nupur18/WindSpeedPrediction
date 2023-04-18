import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
import time
import requests
import os
import traceback
import joblib

def send_to_telegram(message):

    apiToken = '5765471758:AAFPzn2Z2gbbe0sp6yurqxwbSmYrrGanla4'
    chatID = '1213767748'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    print(apiURL)
    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)
        
def compile():
    send_to_telegram(f"Started with {os.getpid()}")
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

    

    rf = RandomForestRegressor(n_estimators = 30, min_samples_split = 2, max_features = 0.5, max_depth = 20, bootstrap = False)
    
    
    et = ExtraTreesRegressor(warm_start = True, n_estimators = 400, min_samples_split = 2, min_samples_leaf = 1, max_features = None, max_depth = 50, criterion = 'absolute_error', bootstrap = False)


    lgbm = lgb.LGBMRegressor(subsample = 0.8, n_estimators = 150, min_child_samples = 1, max_depth = 6, learning_rate = 0.1, colsample_bytree = 0.7, objective="regression", random_state=101)
   
    xgboost = xgb.XGBRegressor(seed = 123, subsample = 0.7999999999999999, n_estimators = 1000, max_depth = 20, learning_rate = 0.01, colsample_bytree = 0.6, colsample_bylevel = 0.6)

    base_models = [
        ('rf', rf),
        ('et',et),
        ('lbgm',lgbm),
        ('xgboost',xgboost),
        ]

    stacked = StackingRegressor(
        estimators = base_models,
        final_estimator = et,
        cv = 5,
        verbose=10,
        n_jobs=-1
    )
    try:
        send_to_telegram("Started Fitting")

        start_time = time.time()
        stacked.fit(X_train, y_train)    
        end_time = time.time()

        send_to_telegram("Fitting Done")

        stacked_prediction = stacked.predict(X_test)
        np.save("tamilNaduEnsemble.npy", np.array(stacked_prediction))

        stacked_r2 = stacked.score(X_test, y_test)
        
        print("-------Stacked Ensemble-------")
        print("R2 score of train: {}".format(stacked.score(X_train, y_train)))
        print("R2 Score of test: {}".format(stacked_r2))
        print("R2 score of test: {}".format(r2_score(y_test, stacked_prediction)))
        print("Computation Time: {}".format(end_time - start_time))
        joblib.dump(stacked, 'tamilNaduEnsemble.pkl')

        
        send_to_telegram("-------Stacked Ensemble-------")
        send_to_telegram("R2 score of train: {}".format(stacked.score(X_train, y_train)))
        send_to_telegram("R2 Score of test: {}".format(stacked_r2))
        send_to_telegram("R2 score of test: {}".format(r2_score(y_test, stacked_prediction)))
        send_to_telegram("Computation Time: {}".format(end_time - start_time))
        send_to_telegram("Model and Predictions saved")

    except Exception as e:
        send_to_telegram(str(e))
        traceback.print_exc()
        
if __name__ == '__main__':
    compile()