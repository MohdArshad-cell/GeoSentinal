import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class IndexCalculator:
    def __init__(self, window=36): # 36-month rolling window as per PDF
        self.window = window

    def rolling_normalize(self, series):
        return (series - series.rolling(self.window).min()) / (series.rolling(self.window).max() - series.rolling(self.window).min())

    def process_index(self, df):
        # Normalization
        df['MCT_norm'] = self.rolling_normalize(df['MCT_score']).fillna(0.5)
        df['INT_norm'] = self.rolling_normalize(df['INT_score']).fillna(0.5)

        # Dynamic PCA Weighting
        features = ['MCT_norm', 'INT_norm']
        x = df[features].values
        
        # We need at least 2 rows to run PCA
        if len(df) > 1:
            pca = PCA(n_components=1)
            pca.fit(StandardScaler().fit_transform(x))
            weights = pca.components_[0]**2 / np.sum(pca.components_[0]**2)
        else:
            weights = [0.5, 0.5]

        df['weight_kinetic'] = weights[0]
        df['weight_narrative'] = weights[1]
        
        # GPTI Calculation
        df['GPTI'] = (df['MCT_norm'] * weights[0]) + (df['INT_norm'] * weights[1])
        
        return df