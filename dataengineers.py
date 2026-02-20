import pandas as pd
from typing import Literal
import numpy as np


class Dataset:
    def __init__(self, df_type: Literal['train', 'test']):
        self.type = df_type
        self.path = 'data/train.csv' if self.type == 'train' else 'data/test.csv'
        self.df = pd.read_csv(self.path)

    def _process_dates(self):
        self.df['delivery_start'] = pd.to_datetime(self.df['delivery_start'], utc=True)
        self.df['delivery_end'] = pd.to_datetime(self.df['delivery_end'], utc=True) # technically this can be removed

        return self.df

    def _drop_id(self):
        self.df = self.df.drop(columns=['id'], errors='ignore')

        return self.df

    def _clean_fill(self):
        df_pivoted = self.df.pivot(index='delivery_start', columns='market')
        df_pivoted = df_pivoted.resample('h').asfreq()
        df_pivoted = df_pivoted.interpolate(method='linear', limit_direction='both')
        df_pivoted = df_pivoted.ffill().bfill()
        df_clean = df_pivoted.stack(future_stack=True).reset_index()
        
        self.df = df_clean
        return self.df

    def _create_features(self):
        self.df['hour'] = self.df['delivery_start'].dt.hour
        self.df['dayofweek'] = self.df['delivery_start'].dt.dayofweek
        self.df['month'] = self.df['delivery_start'].dt.month

        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)

        self.df['residual'] = self.df['load_forecast'] - (self.df['solar_forecast'] + self.df['wind_forecast'])
        self.df['residual_sq'] = self.df['residual'] ** 2

        self.df['renewable_ratio'] = (self.df['solar_forecast'] + self.df['wind_forecast']) / (self.df['load_forecast'] + 1)

        # Rolling features - 6,12,24h
        self.df = self.df.sort_values(['market', 'delivery_start'])

        for col in ['air_temperature_2m', 'wind_speed_10m', 'residual']:
            self.df[f'{col}_roll_6_mean'] = self.df.groupby('market')[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
            self.df[f'{col}_roll_24_std'] = self.df.groupby('market')[col].transform(lambda x: x.rolling(24, min_periods=1).std())


    def build_main(self):
        self._process_dates()

        if self.type == 'train':
            self._drop_id()

        self._clean_fill()

        self._create_features()

        if self.type == 'train':
            self.df = self.df.dropna()


        return self.df