import pandas as pd
from typing import Literal
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

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
        markets = self.df['market'].unique()
        time_range = pd.date_range(
            start=self.df['delivery_start'].min(),
            end=self.df['delivery_start'].max(),
            freq='h'
        )
        grid = pd.MultiIndex.from_product([markets, time_range], names=['market', 'delivery_start'])
        self.df = pd.DataFrame(index=grid).reset_index().merge(
            self.df, on=['market', 'delivery_start'], how='left'
        )

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        cols_to_interp = [c for c in numeric_cols if c not in ['id', 'target']]
        
        self.df[cols_to_interp] = self.df.groupby('market')[cols_to_interp].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        
        self.df[cols_to_interp] = self.df.groupby('market')[cols_to_interp].transform(
            lambda x: x.ffill().bfill()
        )
        
        self.df[cols_to_interp] = self.df[cols_to_interp].fillna(0)
        
        return self.df

    def _create_features(self):
        self.df['timestamp_int'] = self.df['delivery_start'].astype('int64') // 10**9

        self.df['hour'] = self.df['delivery_start'].dt.hour
        self.df['month'] = self.df['delivery_start'].dt.month
        self.df['dayofweek'] = self.df['delivery_start'].dt.dayofweek

        for col, max_val in [('hour', 24), ('dayofweek', 7), ('month', 12)]:
            self.df[f'{col}_sin'] = np.sin(2 * np.pi * self.df[col] / max_val)
            self.df[f'{col}_cos'] = np.cos(2 * np.pi * self.df[col] / max_val)

        self.df['net_load'] = self.df['load_forecast'] - (self.df['solar_forecast'] + self.df['wind_forecast'])
        self.df['net_load_sq'] = self.df['net_load'] ** 2
        
        self.df = pd.get_dummies(self.df, columns=['market'], prefix='m', dtype=int)

    def _drop_cols(self):
        cols_drop = ['delivery_end', 'market', 'delivery_start']
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
        df = df.sort_values(by=['timestamp_int']).reset_index(drop=True)
        
        i_train_size = int(len(df) * train_size)
        train = df.iloc[:i_train_size].copy()
        test = df.iloc[i_train_size:].copy()
        return train, test