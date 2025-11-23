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
        # High Volume + Negative Sentiment = High Tension
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
        Performs Granger Causality Test.
        Null Hypothesis: Narrative does NOT cause Kinetic.
        If p-value < 0.05, we REJECT null (Proof that Narrative -> Kinetic).
        """
        # Data must be in columns [Target, Predictor]
        # We want to see if Narrative (Predictor) causes Kinetic (Target)
        test_data = df[['norm_kinetic', 'norm_narrative']]
        
        # Run test
        # (verbose=False stops it from printing a messy table to the terminal)
        results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
        
        # We extract the P-Value for Lag 3 (assuming a 3-day delay effect)
        # This is a dictionary lookup into the statsmodels result
        p_value = results[3][0]['ssr_ftest'][1]
        
        return p_value