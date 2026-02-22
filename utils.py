from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def rmse(actual_array, model_array):
    return np.sqrt(mean_squared_error(actual_array, model_array))

class Submission:
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

class Ensemble:
    def __init__(self, weights, arr_1, arr_2, arr_3 = None):
        self.weights = weights
        self.arr_1 = arr_1
        self.arr_2 = arr_2
        self.arr_3 = arr_3

    def _two_weights(self):
        w1, w2 = self.weights

        arr_out = self.arr_1 * w1 + self.arr_2 * w2
        return arr_out

    def _three_weights(self):
        w1, w2, w3 = self.weights

        arr_out = self.arr_1 * w1 + self.arr_2 * w2 + self.arr_3 * w3
        return arr_out


    def build(self):
        if self.arr_3 is not None:
            if len(self.weights) == 3:
                return self._three_weights()
            else:
                raise ValueError("Expected 3 weights")
        
        else:
            if len(self.weights) == 2:
                return self._two_weights()
            else:
                raise ValueError("Expected 2 weights")