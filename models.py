import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import RidgeCV, LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt

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

    def predict(self, df: pd.DataFrame):
        X_predict = df[self.features]
        y_pred = self.model.predict(X_predict)

        return y_pred

    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)

class XGB:
    def __init__(self, features: list):
        self.model = None
        self.features = features

    def fit(self, df_train, df_test):
        X_train = df_train[self.features + ['market_int']]
        y_train = df_train['target']
        X_test = df_test[self.features + ['market_int']]
        y_test = df_test['target']

        self.model = xgb.XGBRegressor(
            device="cuda",
            objective='reg:squarederror',
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=9,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100
        )

    def predict(self, df):
        y_pred = self.model.predict(df[self.features + ['market_int']])
        return y_pred

    def stats(self):
        xgb.plot_importance(self.model, max_num_features=15, importance_type='weight')
        plt.title("top 15 features by freq")
        plt.show()

    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)
