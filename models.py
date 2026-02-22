import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import RidgeCV, LinearRegression
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

class Linear:
    def __init__(self, features: list):
        self.scaler = joblib.load('bin/scaler1.pkl')
        self.scaler_features = joblib.load('bin/features1.pkl') # might break otherwise, but not an optimal solution for sure
        self.model = None
        self.features = features
    
    def process(self, df: pd.DataFrame):
        df = pd.get_dummies(df)

        df[self.scaler_features] = self.scaler.transform(df[self.scaler_features])

        return df

    def fit_ridge(self, df: pd.DataFrame):
        self.model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)

        X_train = df[self.features]
        y = df['target']

        self.model.fit(X_train, y)

    def fit_lr(self, df: pd.DataFrame):
        self.model = LinearRegression()

        X_train = df[self.features]
        y = df['target']

        self.model.fit(X_train, y)

    def predict(self, df):
        use_cols = list(dict.fromkeys(self.features + ['market_int']))
        
        y_pred = self.model.predict(df[use_cols])
        return y_pred

    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)

class XGB:
    def __init__(self, features: list):
        self.model = None
        self.features = list(dict.fromkeys(features))

    def fit(self, df_train, df_test):
        X_train = df_train[self.features]
        X_test = df_test[self.features]
        
        y_train = df_train['target']
        y_test = df_test['target']

        self.model = xgb.XGBRegressor(
            device="cuda",
            objective='reg:squarederror',
            n_estimators=5000,
            learning_rate=0.007112277902623446,
            max_depth=8,
            subsample=0.5796806880630087,
            colsample_bytree=0.5565687076298409,
            colsample_bylevel=0.5811173897467319,
            min_child_weight=3, 
            reg_alpha=0.23393408287285775,
            reg_lambda=9.617727386132815,
            gamma=0.43983477413525485,
            early_stopping_rounds=100,
            random_state=42
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100
        )

    def predict(self, df):
        X_predict = df[self.features]
        
        y_pred = self.model.predict(X_predict)
        
        return y_pred

    def stats(self):
        xgb.plot_importance(self.model, max_num_features=20, importance_type='gain')
        plt.title("Feature Importance (Gain)")
        plt.show()

    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)

class lGBM:
    def __init__(self, features: list):
        self.model = None
        self.features = list(dict.fromkeys(features))

    def fit(self, df_train, df_test):
        X_train = df_train[self.features]
        X_test = df_test[self.features]
        
        y_train = df_train['target']
        y_test = df_test['target']

        self.model = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            random_state=42
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(100)]
        )

    def predict(self, df):
        X_predict = df[self.features]
        
        y_pred = self.model.predict(X_predict)
        
        return y_pred

    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)