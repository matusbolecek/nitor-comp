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

        self.df = self.df.sort_values(['market', 'delivery_start']).reset_index(drop=True)
        return self.df

    def _clean_fill(self):
        cols_to_exclude = ['id', 'delivery_end']
        
        exclude_existing = [c for c in cols_to_exclude if c in self.df.columns]
        
        df_subset = self.df.drop(columns=exclude_existing)
        
        df_pivoted = df_subset.pivot(index='delivery_start', columns='market')
        
        df_pivoted = df_pivoted.resample('h').asfreq()
        df_pivoted = df_pivoted.interpolate(method='linear', limit_direction='both')
        df_pivoted = df_pivoted.ffill().bfill()
        
        df_clean = df_pivoted.stack(future_stack=True).reset_index()
        
        if 'id' in self.df.columns:
            df_clean = df_clean.merge(
                self.df[['delivery_start', 'market', 'id']], 
                on=['delivery_start', 'market'], 
                how='left'
            )
            
        self.df = df_clean
        return self.df

    def _create_features(self):
        self.df = self.df.sort_values(['market', 'delivery_start'])
        self.df['hour'] = self.df['delivery_start'].dt.hour
        self.df['dayofweek'] = self.df['delivery_start'].dt.dayofweek
        self.df['month'] = self.df['delivery_start'].dt.month

        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)

        self.df['residual'] = self.df['load_forecast'] - (self.df['solar_forecast'] + self.df['wind_forecast'])
        self.df['renewable_ratio'] = (self.df['solar_forecast'] + self.df['wind_forecast']) / (self.df['load_forecast'] + 1)

        for col in ['air_temperature_2m', 'wind_speed_10m', 'residual']:
            self.df[f'{col}_roll_6_mean'] = self.df.groupby('market')[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
            self.df[f'{col}_roll_24_std'] = self.df.groupby('market')[col].transform(lambda x: x.rolling(24, min_periods=1).std())

    def _drop_cols(self):
        cols_drop = [
            'delivery_end',

            'hour', 'month',
            
            # radiation, clouds (captured by solar_forecast)
            'global_horizontal_irradiance', 'diffuse_horizontal_irradiance', 'direct_normal_irradiance',
            'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'cloud_cover_total',
            
            # temperature variations (keep air_temperature_2m)
            'apparent_temperature_2m', 'dew_point_temperature_2m', 'wet_bulb_temperature_2m', 
            
            # atmospheric stability
            'convective_available_potential_energy', 'lifted_index', 'convective_inhibition', 'freezing_level_height',
            
            # wind Details (keep speed_10m and wind_forecast)
            'wind_gust_speed_10m', 'wind_speed_80m', 'wind_direction_80m', 
            'visibility', 'surface_pressure'
        ]

        self.df = self.df.drop(columns=cols_drop, errors='ignore')

    def build_main(self):
        self._process_dates()   
        self._clean_fill()
        self._create_features()
        self._drop_cols()

        if 'target' in self.df.columns:
            self.df = self.df.dropna(subset=['target'])

        return self.df

    def build_train_test(self, train_size: float = 0.8):
        df = self.build_main()

        df = df.sort_values(by=['delivery_start', 'market']).reset_index(drop=True)

        i_train_size = int(len(df) * train_size)

        train = df.iloc[:i_train_size].copy()
        test = df.iloc[i_train_size:].copy()
        
        return train, test