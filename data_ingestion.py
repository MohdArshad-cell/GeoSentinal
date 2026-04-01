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
        self.history_file = "geosentinal_benchmarks.csv"

    def load_historical_baseline(self, scenario):
        """Loads and filters CSV data. Renames columns for the Math Engine."""
        if os.path.exists(self.history_file):
            df_all = pd.read_csv(self.history_file)
            df_all['date'] = pd.to_datetime(df_all['date'])
            df = df_all[df_all['scenario'] == scenario].copy()
            
            if df.empty:
                print(f"⚠️ Scenario '{scenario}' not found. Falling back to synthetic.")
                df = self._generate_massive_synthetic(scenario)
        else:
            df = self._generate_massive_synthetic(scenario)
        
        # Ensure compatibility with IndexCalculator
        rename_map = {"kinetic_raw": "MCT_score", "narrative_raw": "INT_score"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        return df.sort_values('date')

    def generate_location_data(self, scenario, count=200):
        """
        [PRO UPDATE] Ab ye CSV se real event coordinates uthayega.
        Random dots ki jagah actual hotspots dikhayega.
        """
        # Scenario name matching fix
        zones = {
            "India-Pakistan 2019": {"lat": 34.08, "lon": 74.79, "spread": 2.0},
            "Russia-Ukraine 2022": {"lat": 48.37, "lon": 34.63, "spread": 4.0},
            "Israel-Palestine 2023": {"lat": 31.04, "lon": 34.85, "spread": 1.0},
            "Iran-Israel-US 2026": {"lat": 32.42, "lon": 53.68, "spread": 5.0},
            "Sudan Conflict 2023": {"lat": 15.50, "lon": 32.55, "spread": 4.0}
        }
        
        config = zones.get(scenario, {"lat": 20, "lon": 0, "spread": 10})
        
        # Check if we have historical data for this scenario to pull real coordinates
        try:
            df_hist = self.load_historical_baseline(scenario)
            # Sirf un rows ko uthao jahan actual event hua hai (high score)
            event_coords = df_hist[df_hist['MCT_score'] > 40][['lat', 'lon']].dropna()
            
            if not event_coords.empty:
                # Actual events ke coordinates use karo + thoda noise for visual density
                lats = np.append(event_coords['lat'].values, np.random.normal(config['lat'], config['spread'], count))
                lons = np.append(event_coords['lon'].values, np.random.normal(config['lon'], config['spread'], count))
            else:
                lats = np.random.normal(config['lat'], config['spread'], count)
                lons = np.random.normal(config['lon'], config['spread'], count)
        except:
            lats = np.random.normal(config['lat'], config['spread'], count)
            lons = np.random.normal(config['lon'], config['spread'], count)

        df = pd.DataFrame({
            'lat': lats[:count], 
            'lon': lons[:count],
            'intensity': np.random.uniform(0, 1, len(lats[:count]))
        })
        return df, config['lat'], config['lon'], 5

    def _generate_massive_synthetic(self, scenario):
        """[FIXED] Generates a 3-year baseline with the REQUIRED 'event' column."""
        periods = 1100 
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        
        mct = np.random.uniform(5, 20, periods)
        int_score = np.random.uniform(10, 30, periods)
        
        df = pd.DataFrame({
            "date": dates, 
            "MCT_score": mct, 
            "INT_score": int_score,
            "event": "Routine Monitoring", # <--- YE ADD KARO
            "lat": 20.0, 
            "lon": 77.0,
            "scenario": scenario # Scenario column bhi add kar do safety ke liye
        })
        
        # Add a major conflict spike based on the scenario name
        spike_val = 80 if "2019" in scenario or "2026" in scenario else 50
        df.iloc[-30:-15, df.columns.get_loc("MCT_score")] += spike_val
        df.iloc[-25:-10, df.columns.get_loc("INT_score")] += spike_val + 10
            
        return df

    def generate_synthetic_data(self, dyad):
        """Short-term data for 'Live' mode."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        df = pd.DataFrame({
            "date": dates,
            "MCT_score": np.random.uniform(10, 40, 30),
            "INT_score": np.random.uniform(15, 50, 30)
        })
        df.iloc[-5:, df.columns.get_loc("MCT_score")] += 35
        return df

# --- UI WRAPPERS ---
def get_validation_data(scenario):
    return DataIngestor().load_historical_baseline(scenario)

def generate_location_data(scenario, count=100):
    return DataIngestor().generate_location_data(scenario, count)

def generate_synthetic_data(dyad):
    return DataIngestor().generate_synthetic_data(dyad)