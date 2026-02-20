import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import RidgeCV, LinearRegression

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