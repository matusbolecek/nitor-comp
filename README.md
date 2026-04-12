# Nitor Energy Quant Trading Case 

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1172B8?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

Our team's submission for the **Nitor Energy Quantitative Trading Case Competition**. The objective was to act as quantitative traders and build a machine learning pipeline to forecast short-term wholesale electricity prices using real-world weather and market data. The result was a public RMSE of 22.87 (rakety.csv) which resulted in #10 on the public leaderboard.

## Key Highlights

*   **Feature Engineering:** We engineered physical indicators. This includes cubed wind speeds (as wind power scales non-linearly), a custom solar-cloud suppression index, and baseline deviation metrics for temperature stress.
*   **Time-Series Validation:** Implemented strict 5-fold `TimeSeriesSplit` cross-validation to simulate real-world trading and prevent future data leakage
*   **Gradient Boosted Ensembling:** Built an optimized, 50/50 weighted ensemble of **XGBoost** (GPU-accelerated) and **LightGBM** regressors to smooth variance. 

## Repository Structure

*   `dataengineers.py`: Data ingestion, linear interpolation for missing values, and all domain-feature engineering (lags, rolling means, weather transformations)
*   `models.py`: OOP-based wrappers for training, tuning, and cross-validating the XGBoost and LightGBM models
*   `utils.py`: Evaluation metrics, assertion-based submission validation, and ensembling logic
*   `final.ipynb`: The main execution notebook that ties the pipeline together and generates the final competition submission
> The repository also includes the submission history and other utilities / notebooks used in training 

## Getting a prediction 

1.  **Dependencies:** Ensure you have Python 3.10+ and install the required packages:
    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
    ```
2.  **Data:** Place `train.csv` and `test_for_participants.csv` in the `data/` root directory.
3.  **Run:** Execute `final.ipynb` to run the feature engineering, train the ensemble, and generate `my_submission.csv`.

## Reflection

This project was built under competition time pressure, and looking back there are two areas we would approach differently given more time.

**Hyperparameter tuning**
The Optuna tuning loop in `tuner.ipynb` uses the held-out validation split as the eval set inside the objective function. This means hyperparameters were selected by directly minimising error on the same data used to assess them — a subtle but real form of data leakage. The correct approach would be a three-way split (train / validation / test), or running the Optuna objective against a nested `TimeSeriesSplit` so the test set stays completely untouched until final evaluation. The tuned parameters likely still generalise reasonably well given the dataset size, but the correct approach would have likely resulted in a lower RMSE.

**Feature engineering**
`dataengineers.py` grew iteratively during the competition and accumulated features that should have been removed. There was a heavy feature engineering rewrite attempt, however we didn't have time to retrain from a cleaned feature set before the submission deadline in a way that would result in a lower RMSE. 
