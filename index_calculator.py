import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class IndexCalculator:
    def __init__(self, window=36, epsilon=1e-5):
        self.window = window
        self.epsilon = epsilon 

    def rolling_normalize(self, series):
        """Min-Max Normalization with a variance floor."""
        roll_min = series.rolling(window=self.window, min_periods=1).min()
        roll_max = series.rolling(window=self.window, min_periods=1).max()
        return (series - roll_min) / (roll_max - roll_min + self.epsilon)

    def process_index(self, df):
        # 1. Gaps fill karo
        df['MCT_score'] = df['MCT_score'].ffill().fillna(10)
        df['INT_score'] = df['INT_score'].ffill().fillna(15)

        # 2. Normalization
        df['MCT_norm'] = self.rolling_normalize(df['MCT_score'])
        df['INT_norm'] = self.rolling_normalize(df['INT_score'])

        # 3. [ADVANCED FIX] Adaptive Dynamic Weighting logic
        # Har row ke liye weights calculate karna slow hoga, toh hum logic ko optimize karenge
        features = ['MCT_norm', 'INT_norm']
        x = df[features].values
        
        weights_k = []
        weights_n = []

        # Demo optimization: Pehle 30 din 0.5/0.5 rakho, phir adaptive ban jao
        for i in range(len(df)):
            if i < self.window:
                weights_k.append(0.5)
                weights_n.append(0.5)
            else:
                # Pichle 'window' dinon ka data uthao
                window_data = x[i-self.window : i+1]
                
                # Check variance: Agar data constant hai toh PCA fail hoga
                if np.var(window_data[:, 0]) < self.epsilon and np.var(window_data[:, 1]) < self.epsilon:
                    weights_k.append(0.5)
                    weights_n.append(0.5)
                else:
                    try:
                        pca = PCA(n_components=1)
                        # Scaling for local window
                        local_scaled = StandardScaler().fit_transform(window_data)
                        pca.fit(local_scaled)
                        
                        # Weight calculation (Variance explained ratio)
                        local_weights = pca.components_[0]**2 / np.sum(pca.components_[0]**2)
                        weights_k.append(local_weights[0])
                        weights_n.append(local_weights[1])
                    except:
                        weights_k.append(0.5)
                        weights_n.append(0.5)

        df['weight_kinetic'] = weights_k
        df['weight_narrative'] = weights_n
        
        # 4. Final GPTI Calculation
        df['GPTI'] = (df['MCT_norm'] * df['weight_kinetic']) + (df['INT_norm'] * df['weight_narrative'])
        
        # 5. [UI FIX] Smoothed Trend for Early Warning System
        # Simple diff ki jagah EMA use kar rahe hain taaki alerts stable rahein
        df['GPTI_Trend'] = df['GPTI'].diff().ewm(span=7).mean().fillna(0)
        
        return df