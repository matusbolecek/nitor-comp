import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import RidgeCV, LinearRegression
import xgboost as xgb
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
        self.features = [f for f in features if f not in ['id', 'target', 'month', 'net_load_sq', 'net_load_cubed']]
        self.means = {}
        self.stds = {}

    def fit(self, df_train, df_test):
        X_train = df_train[self.features]
        X_test = df_test[self.features]

        df_train = df_train.copy()
        df_test = df_test.copy()
        
        df_train['target_scaled'] = 0.0
        df_test['target_scaled'] = 0.0
        
        unique_markets = df_train['market_int'].unique()
        
        for m in unique_markets:
            mask = df_train['market_int'] == m
            mu = df_train.loc[mask, 'target'].mean()
            sigma = df_train.loc[mask, 'target'].std()
            
            self.means[m] = mu
            self.stds[m] = sigma
            
            df_train.loc[mask, 'target_scaled'] = (df_train.loc[mask, 'target'] - mu) / sigma
            
            mask_test = df_test['market_int'] == m
            if mask_test.any():
                df_test.loc[mask_test, 'target_scaled'] = (df_test.loc[mask_test, 'target'] - mu) / sigma

        y_train = df_train['target_scaled']
        y_test = df_test['target_scaled']

        self.model = xgb.XGBRegressor(
            device="cuda",
            n_estimators=5000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=10,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=2.0,
            objective='reg:squarederror',
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
        
        y_pred_scaled = self.model.predict(X_predict)
        
        # inverse transform (per market)
        df_temp = df.copy()
        df_temp['pred_scaled'] = y_pred_scaled
        df_temp['final_pred'] = 0.0
        
        for m in self.means:
            mask = df_temp['market_int'] == m
            if mask.any():
                mu = self.means[m]
                sigma = self.stds[m]
                df_temp.loc[mask, 'final_pred'] = (df_temp.loc[mask, 'pred_scaled'] * sigma) + mu
                
        return df_temp['final_pred'].values

    def stats(self):
        xgb.plot_importance(self.model, max_num_features=20, importance_type='gain')
        plt.title("Feature Importance (Gain)")
        plt.show()

    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)
