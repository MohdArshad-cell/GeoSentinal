import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class IndexCalculator:
    def __init__(self, window=36, epsilon=1e-5): # TECHNICAL FIX: Added epsilon
        self.window = window
        self.epsilon = epsilon # Variance Floor

    def rolling_normalize(self, series):
        # TECHNICAL FIX: Variance Floor logic prevents minor skirmishes from spiking the graph
        roll_min = series.rolling(self.window).min()
        roll_max = series.rolling(self.window).max()
        return (series - roll_min) / (roll_max - roll_min + self.epsilon)

    def process_index(self, df):
        # TECHNICAL FIX: Hybrid Frequency Fusion
        # Fills weekly military gaps to match daily narrative updates
        if 'MCT_score' in df.columns:
            df['MCT_score'] = df['MCT_score'].ffill()
        if 'INT_score' in df.columns:
            df['INT_score'] = df['INT_score'].ffill()

        # Normalization
        df['MCT_norm'] = self.rolling_normalize(df['MCT_score']).fillna(0.5)
        df['INT_norm'] = self.rolling_normalize(df['INT_score']).fillna(0.5)

        # Dynamic PCA Weighting
        features = ['MCT_norm', 'INT_norm']
        x = df[features].values
        
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
        
        # FEATURE INTEGRATION: Calculate the trend derivative for the Early Warning System
        df['GPTI_Trend'] = df['GPTI'].diff().fillna(0)
        
        return df