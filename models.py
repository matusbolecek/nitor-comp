import pandas as pd

class Linear:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def process(self):
        self.df = pd.get_dummies(self.df)
        self.df = self.df.drop('id', axis=1)

        return self.df

    def fit(self):
        pass