import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests

class IndexCalculator:
    def process_index(self, df):
        """
        Takes raw data and converts it into the final GPTI Index.
        """
        # 1. CALCULATE RAW NARRATIVE SCORE
        df['raw_narrative_score'] = df['narrative_volume'] * (1 - df['sentiment_score'])
        
        # 2. NORMALIZATION
        scaler = MinMaxScaler()
        df['norm_kinetic'] = scaler.fit_transform(df[['kinetic_score']])
        df['norm_narrative'] = scaler.fit_transform(df[['raw_narrative_score']])
        
        # 3. DYNAMIC WEIGHTING (PCA)
        w_kinetic_list = []
        w_narrative_list = []
        window_size = 30
        
        for i in range(len(df)):
            if i < window_size:
                w_kinetic_list.append(0.5)
                w_narrative_list.append(0.5)
            else:
                window_data = df.iloc[i-window_size:i][['norm_kinetic', 'norm_narrative']]
                try:
                    pca = PCA(n_components=1)
                    pca.fit(window_data)
                    components = abs(pca.components_[0])
                    w_norm = components / sum(components)
                    w_kinetic_list.append(w_norm[0])
                    w_narrative_list.append(w_norm[1])
                except:
                    # Fallback if variance is zero
                    w_kinetic_list.append(0.5)
                    w_narrative_list.append(0.5)
        
        df['weight_kinetic'] = w_kinetic_list
        df['weight_narrative'] = w_narrative_list
        
        # 4. FINAL CALCULATION
        df['GPTI'] = (df['weight_kinetic'] * df['norm_kinetic']) + \
                     (df['weight_narrative'] * df['norm_narrative'])
        df['GPTI'] = df['GPTI'] * 100
        
        return df

    def test_causality(self, df, maxlag=5):
        """
        Performs Granger Causality Test with Safety Checks.
        """
        # --- SAFETY CHECK 1: Data Length ---
        # We need at least (maxlag + 5) rows to run the test accurately.
        if len(df) < (maxlag + 5):
            return 1.0 # Return 1.0 (Not Significant) if data is too short

        # Prepare Data
        test_data = df[['norm_kinetic', 'norm_narrative']].copy()

        # --- SAFETY CHECK 2: Clean NaNs/Infs ---
        test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_data.dropna(inplace=True)

        # Re-check length after dropping NaNs
        if len(test_data) < (maxlag + 5):
            return 1.0

        # --- SAFETY CHECK 3: Constant Data ---
        # If a column is all zeros (no variance), the test will crash.
        # We check if the standard deviation is 0.
        if test_data['norm_kinetic'].std() == 0 or test_data['norm_narrative'].std() == 0:
            return 1.0

        try:
            # Run test
            results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
            # Extract P-Value for Lag 3
            p_value = results[3][0]['ssr_ftest'][1]
            return p_value
        except Exception:
            # If any other math error happens (e.g. singular matrix), fail safely
            return 1.0