import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from sklearn.linear_model import RidgeCV, LinearRegression
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

class Linear:
    def __init__(self, features: list):
        self.scaler = joblib.load('bin/scaler1.pkl')
        self.scaler_features = joblib.load('bin/features1.pkl')
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

    def fit(self, df_train, n_splits=5):
        X_train = df_train[self.features]
        y_train = df_train['target']

        tuned_params = {
            'max_depth': 9, 
            'min_child_weight': 10, 
            'learning_rate': 0.011041791784948136, 
            'subsample': 0.769216997688628, 
            'colsample_bytree': 0.3002874876389264, 
            'colsample_bylevel': 0.68925894223455, 
            'reg_alpha': 0.17428426292668459, 
            'reg_lambda': 77.5689626698248, 
            'gamma': 0.9947338308302472
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_rounds_per_fold = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            m = xgb.XGBRegressor(
                device="cuda",
                objective='reg:squarederror',
                tree_method='hist',
                enable_categorical=True,
                n_estimators=5000,
                early_stopping_rounds=100,
                random_state=42,
                **tuned_params
            )
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            best_rounds_per_fold.append(m.best_iteration)

        optimal_rounds = int(np.mean(best_rounds_per_fold))

        self.model = xgb.XGBRegressor(
            device="cuda",
            objective='reg:squarederror',
            tree_method='hist',
            enable_categorical=True,
            n_estimators=optimal_rounds,
            random_state=42,
            **tuned_params
        )
        self.model.fit(X_train, y_train, verbose=100)

    def predict(self, df):
        return self.model.predict(df[self.features])

    def stats(self, n=20):
        xgb.plot_importance(self.model, max_num_features=n, importance_type='gain')
        plt.title("Feature Importance (Gain)")
        plt.show()

    def dump_features(self):
        print(self.features)

    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)

class lGBM:
    def __init__(self, features: list):
        self.model = None
        self.features = list(dict.fromkeys(features))

    def fit(self, df_train, n_splits=5):
        X_train = df_train[self.features]
        y_train = df_train['target']

        tuned_params = {
            'num_leaves': 111,
            'max_depth': 11,
            'learning_rate': 0.017450649341248122,
            'subsample': 0.7317287482536371,
            'subsample_freq': 1,
            'colsample_bytree': 0.11288101765162045,
            'min_split_gain': 1.0482834566823582,
            'reg_alpha': 2.075968543559966,
            'reg_lambda': 20.957202176792954,
            'min_child_samples': 22
            }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_rounds_per_fold = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            m = lgb.LGBMRegressor(
                n_estimators=5000,
                random_state=42,
                **tuned_params
            )
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)]
            )
            best_rounds_per_fold.append(m.best_iteration_)

        optimal_rounds = int(np.mean(best_rounds_per_fold))

        self.model = lgb.LGBMRegressor(
            n_estimators=optimal_rounds,
            random_state=42,
            **tuned_params
        )
        self.model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=100)])

    def predict(self, df):
        return self.model.predict(df[self.features])

    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)