from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def rmse(actual_array, model_array):
    return np.sqrt(mean_squared_error(actual_array, model_array))

class Submission():
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def process(self):
        self.df = self.df[['id', 'target']]
        self.df = self.df.sort_values(by=["id"])

        return self.df

    def validate(self):
        assert list(self.df.columns) == ['id', 'target'], "Wrong columns!"
        assert len(self.df) == 13098, f"Wrong row count: {len(self.df)}"
        assert self.df['id'].min() == 133627, "IDs must start at 133627"
        assert self.df['id'].max() == 146778, "IDs must end at 146778"
        assert self.df['target'].notna().all(), "No NaN values allowed!"
        assert np.isfinite(self.df['target']).all(), "No infinite values allowed!"

        print("✅ Validation passed!")

    def dump(self):
        self.df.to_csv("my_submission.csv", index=False)