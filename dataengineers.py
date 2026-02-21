import pandas as pd
from typing import Literal
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class Dataset:
    def __init__(self, df_type: Literal['train', 'test']):
        self.type = df_type
        self.path = 'data/train.csv' if self.type == 'train' else 'data/test_for_participants.csv'
        self.df = pd.read_csv(self.path)
        
        if self.type == 'train':
            self.scaler = StandardScaler()
            self.le_market = LabelEncoder()
        else:
            self.scaler = joblib.load('bin/scaler_main.pkl')
            self.le_market = joblib.load('bin/le_market.pkl')

    def _process_dates(self):
        self.df['delivery_start'] = pd.to_datetime(self.df['delivery_start'], utc=True)

        self.df = self.df.sort_values(['market', 'delivery_start']).reset_index(drop=True)
        return self.df

    def _clean_fill(self):
        markets = self.df['market'].unique()
        time_range = pd.date_range(
            start=self.df['delivery_start'].min(),
            end=self.df['delivery_start'].max(),
            freq='h'
        )
        
        grid = pd.MultiIndex.from_product([markets, time_range], names=['market', 'delivery_start'])
        df_grid = pd.DataFrame(index=grid).reset_index()
        
        self.df = pd.merge(df_grid, self.df, on=['market', 'delivery_start'], how='left')
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        cols_to_interp = [c for c in numeric_cols if c not in ['id', 'target']]
        
        self.df[cols_to_interp] = self.df.groupby('market')[cols_to_interp].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        
        self.df[cols_to_interp] = self.df[cols_to_interp].fillna(0) # should be clean, but we hard fil just in case
        
        return self.df

    def _create_features(self):
        self.df['hour'] = self.df['delivery_start'].dt.hour
        self.df['dayofweek'] = self.df['delivery_start'].dt.dayofweek
        self.df['month'] = self.df['delivery_start'].dt.month

        for col, max_val in [('hour', 24), ('dayofweek', 7), ('month', 12)]:
            self.df[f'{col}_sin'] = np.sin(2 * np.pi * self.df[col] / max_val)
            self.df[f'{col}_cos'] = np.cos(2 * np.pi * self.df[col] / max_val)

        self.df['residual'] = self.df['load_forecast'] - (self.df['solar_forecast'] + self.df['wind_forecast'])
        
        self.df['renewable_ratio'] = (self.df['solar_forecast'] + self.df['wind_forecast']) / (self.df['load_forecast'] + 10)
        
        weather_cols = ['air_temperature_2m', 'wind_speed_10m', 'residual']
        for col in weather_cols:
            self.df[f'{col}_roll_6_mean'] = self.df.groupby('market')[col].transform(lambda x: x.rolling(6, min_periods=1).mean()).fillna(0)

    def _encode_and_scale(self):
        if self.type == 'train':
            self.df['market_int'] = self.le_market.fit_transform(self.df['market'])
            joblib.dump(self.le_market, 'bin/le_market.pkl')
        else:
            self.df['market_int'] = self.le_market.transform(self.df['market'])

        exclude_cols = ['id', 'delivery_start', 'market', 'market_int', 'target', 'delivery_end']
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        self.feature_names = feature_cols

        if self.type == 'train':
            self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
            joblib.dump(self.scaler, 'bin/scaler_main.pkl')
            joblib.dump(feature_cols, 'bin/feature_names.pkl')
        else:
            train_cols = joblib.load('bin/feature_names.pkl')
            for c in train_cols:
                if c not in self.df.columns:
                    self.df[c] = 0

            self.df = self.df[train_cols + [c for c in self.df.columns if c not in train_cols]]
            self.df[train_cols] = self.scaler.transform(self.df[train_cols])

    def _drop_cols(self):
            cols_drop = ['delivery_end'] # might be expanded later idk
            
            self.df = self.df.drop(columns=cols_drop, errors='ignore')

            return self.df

    def build_main(self):
        self._process_dates()   
        self._clean_fill()
        self._create_features()
        self._drop_cols()
        self._encode_and_scale()

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