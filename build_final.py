import warnings
import logging

from dataengineers import Dataset
from models import XGB, lGBM
from utils import Ensemble, Submission
warnings.filterwarnings("ignore")

logging.getLogger("lightgbm").setLevel(logging.ERROR)

EXCLUDE = ["id", "target", "delivery_start"]

if __name__ == "__main__":
    train = Dataset("train").build_main()
    features = [c for c in train.columns if c not in EXCLUDE]
    print(f"        {len(train)} rows | {len(features)} features")

    df_out = Dataset("test").build_main()

    lg = lGBM(features)
    lg.fit(train)
    lg_preds = lg.predict(df_out)

    xg = XGB(features)
    xg.fit(train)
    xg_preds = xg.predict(df_out)

    y_out = Ensemble([0.5, 0.5], xg_preds, lg_preds).build()
    df_out["target"] = y_out

    sub = Submission(df_out)
    sub.process()
    sub.validate()
    sub.dump()
    print("Saved to my_submission.csv")
