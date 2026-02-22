import pandas as pd
from typing import Literal
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Dataset:
    def __init__(self, df_type: Literal['train', 'test']):
        self.type = df_type
        self.path = 'data/train.csv' if self.type == 'train' else 'data/test_for_participants.csv'
        self.df = pd.read_csv(self.path)

    def _process_dates(self):
        self.df['delivery_start'] = pd.to_datetime(self.df['delivery_start'], utc=True)
        self.df = self.df.sort_values(['market', 'delivery_start']).reset_index(drop=True)
        return self.df

    def _clean_fill(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        cols_to_interp = [c for c in numeric_cols if c not in ['id', 'target']]
        
        self.df[cols_to_interp] = self.df.groupby('market')[cols_to_interp].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )

        self.df[cols_to_interp] = self.df[cols_to_interp].fillna(0)
        
        return self.df

    def _create_features(self):
        # Time features
        self.df['hour'] = self.df['delivery_start'].dt.hour
        self.df['dayofweek'] = self.df['delivery_start'].dt.dayofweek
        self.df['month'] = self.df['delivery_start'].dt.month
        self.df['day_of_year'] = self.df['delivery_start'].dt.dayofyear
        self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)

        cyclical_features = [('hour', 24), ('dayofweek', 7), ('month', 12), ('day_of_year', 365)]
        for col, max_val in cyclical_features:
            self.df[f'{col}_sin'] = np.sin(2 * np.pi * self.df[col] / max_val)
            self.df[f'{col}_cos'] = np.cos(2 * np.pi * self.df[col] / max_val)
            self.df = self.df.drop(col, axis='columns')

        # Renewables / residual
        self.df['renewables'] = self.df['solar_forecast'] + self.df['wind_forecast']
        self.df['renewable_ratio'] = self.df['renewables'] / (self.df['load_forecast'] + 1)

        self.df['residual'] = self.df['load_forecast'] - self.df['wind_forecast'] - self.df['solar_forecast']
        
        # Cloud cover / solar interaction
        self.df['cloud_suppression'] = (self.df['cloud_cover_low'] * 0.9 + self.df['cloud_cover_mid'] * 0.6 + self.df['cloud_cover_high'] * 0.2)
        self.df['solar_cloud_adjusted'] = self.df['solar_forecast'] * (1 - self.df['cloud_suppression'] / 100)

        # Stress
        if 'air_temperature_2m' in self.df.columns:
            self.df['cold_stress'] = self.df['air_temperature_2m'].apply(lambda x: max(0, 280 - x))
            self.df['heat_stress'] = self.df['air_temperature_2m'].apply(lambda x: max(0, x - 295))

        # Wind features
        self.df['wind_dir_sin'] = np.sin(2 * np.pi * self.df['wind_direction_80m'] / 360)
        self.df['wind_dir_cos'] = np.cos(2 * np.pi * self.df['wind_direction_80m'] / 360)
        self.df = self.df.drop('wind_direction_80m', axis=1)

        self.df['wind_speed_80m_cubed'] = self.df['wind_speed_80m'] ** 3
        self.df['wind_power_residual'] = self.df['wind_speed_80m_cubed'] - self.df['wind_forecast']
        
        # Categorical market
        self.df['market'] = self.df['market'].astype('category')

    def _create_lag_features(self):
        cols_to_lag = ['renewables', 'residual', 'load_forecast']
        lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]

        for col in cols_to_lag:
            if col not in self.df.columns:
                continue
            for lag in lag_hours:
                self.df[f'{col}_lag_{lag}h'] = (
                    self.df.groupby('market')[col].shift(lag)
                )
            
            # Rolling stats
            for window in [6, 24]:
                self.df[f'{col}_rolling_mean_{window}h'] = (
                    self.df.groupby('market')[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

            # Ramp rates
            for diff in [1, 3, 6]:
                self.df[f'{col}_ramp_{diff}h'] = (
                    self.df.groupby('market')[col].diff(diff)
                )

    def _drop_cols(self):
        cols_drop = ['delivery_end', 'market']
        self.df = self.df.drop(columns=cols_drop, errors='ignore')

    def build_main(self):
        self._process_dates()   
        self._clean_fill()
        self._create_features()
        self._create_lag_features()
        self._drop_cols()

        if 'target' in self.df.columns:
            self.df = self.df.dropna(subset=['target'])

        return self.df

    def build_train_test(self, split_date=None):     
        df = self.build_main()
            
        train_size = int(len(df) * 0.8)
        train = df.iloc[:train_size].copy()
        test = df.iloc[train_size:].copy()
        
        return train, test