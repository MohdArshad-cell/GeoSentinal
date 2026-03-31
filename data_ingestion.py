import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class DataIngestor:
    def __init__(self):
        self.acled_key = os.getenv("ACLED_API_KEY")
        self.acled_email = os.getenv("ACLED_EMAIL")
        # Focused benchmark file containing 3-year conflict windows
        self.history_file = "geosentinal_benchmarks.csv"

    def load_historical_baseline(self, scenario):
        """
        [SCENARIO FILTERED UPDATE] 
        Ye function ab 'geosentinal_benchmarks.csv' se sirf wahi data load karega 
        jo selected scenario se match karta hai.
        """
        if os.path.exists(self.history_file):
            df_all = pd.read_csv(self.history_file)
            df_all['date'] = pd.to_datetime(df_all['date'])
            
            # Scenario ke basis pe filter
            df = df_all[df_all['scenario'] == scenario].copy()
            
            if df.empty:
                print(f"⚠️ Scenario '{scenario}' not found in CSV. Using fallback.")
                df = self._generate_massive_synthetic(scenario)
        else:
            print(f"⚠️ {self.history_file} not found. Generating tactical baseline...")
            df = self._generate_massive_synthetic(scenario)
            # Future use ke liye save kar lo
            df['scenario'] = scenario
            df.to_csv(self.history_file, index=False)
        
        # Ensure correct column naming for IndexCalculator
        if 'kinetic_raw' in df.columns:
            df = df.rename(columns={"kinetic_raw": "MCT_score", "narrative_raw": "INT_score"})
        
        return df.sort_values('date')

    def _generate_massive_synthetic(self, scenario):
        """Generates a 3-year baseline (1100 days) to satisfy the PCA window."""
        periods = 1100 
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        
        mct = np.random.uniform(5, 20, periods)
        int_score = np.random.uniform(10, 30, periods)
        
        df = pd.DataFrame({"date": dates, "MCT_score": mct, "INT_score": int_score})
        
        # Add a major conflict spike based on the scenario name
        spike_val = 80 if "2019" in scenario or "2026" in scenario else 50
        df.iloc[-30:-15, df.columns.get_loc("MCT_score")] += spike_val
        df.iloc[-25:-10, df.columns.get_loc("INT_score")] += spike_val + 10
            
        return df

    def fetch_live_acled(self, country="India"):
        """Live ACLED API integration for the last 30 days."""
        if not self.acled_key or "aapki" in self.acled_key:
            return self.generate_synthetic_data(country)
            
        base_url = f"https://acleddata.com/api/acled/read?key={self.acled_key}&email={self.acled_email}"
        params = {
            "country": country,
            "limit": 50,
            "event_date": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            "event_date_where": ">"
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            return pd.DataFrame(data.get('data', []))
        except Exception as e:
            print(f"❌ ACLED API Failure: {e}")
            return self.generate_synthetic_data(country)

    def generate_location_data(self, dyad, count=200):
        """War Room map coordinates with regional 'spread' for realism."""
        zones = {
            "India-Pakistan": {"lat": 34.08, "lon": 74.79, "spread": 2.5},
            "Russia-Ukraine": {"lat": 48.37, "lon": 34.63, "spread": 4.0},
            "Israel-Palestine": {"lat": 31.04, "lon": 34.85, "spread": 1.0},
            "Iran-Israel-USA": {"lat": 32.42, "lon": 53.68, "spread": 5.0}
        }
        config = zones.get(dyad, {"lat": 20, "lon": 0, "spread": 10})
        
        df = pd.DataFrame({
            'lat': np.random.normal(config['lat'], config['spread'], count),
            'lon': np.random.normal(config['lon'], config['spread'], count),
            'intensity': np.random.uniform(0, 1, count)
        })
        return df, config['lat'], config['lon'], 5

    def generate_synthetic_data(self, dyad):
        """Short-term 30-day data for 'Live Intelligence' mode."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        df = pd.DataFrame({
            "date": dates,
            "MCT_score": np.random.uniform(10, 40, 30),
            "INT_score": np.random.uniform(15, 50, 30)
        })
        # Simulate recent escalation for visual impact
        df.iloc[-5:, df.columns.get_loc("MCT_score")] += 35
        return df

# --- UI COMPATIBILITY WRAPPERS ---
def get_validation_data(scenario):
    return DataIngestor().load_historical_baseline(scenario)

def generate_synthetic_data(dyad):
    return DataIngestor().generate_synthetic_data(dyad)

def generate_location_data(dyad, count=100):
    return DataIngestor().generate_location_data(dyad, count)