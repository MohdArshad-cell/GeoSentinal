import pandas as pd
import numpy as np
import requests
import feedparser
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class DataIngestor:
    def __init__(self):
        self.acled_key = os.getenv("ACLED_API_KEY")
        self.acled_email = os.getenv("ACLED_EMAIL")
        
    def get_validation_data(self, scenario):
        """
        Provides hand-curated historical data for validation as per project details.
        This allows the tool to 'prove' it can detect known events.
        """
        if scenario == "India-Pakistan 2019":
            # Data representing the Pulwama to Balakot escalation sequence
            data = {
                "date": pd.to_datetime(["2019-02-10", "2019-02-14", "2019-02-18", "2019-02-24", "2019-02-26", "2019-02-27"]),
                "event": ["Baseline", "Pulwama Attack", "Troop Build-up", "Narrative Spike", "Balakot Strike", "Dogfight/Capture"],
                "kinetic_raw": [10, 85, 40, 30, 95, 100],  # MCT Pillar
                "narrative_raw": [15, 60, 80, 95, 100, 100], # INT Pillar
            }
        elif scenario == "Iran-Israel 2026 (Epic Fury)":
            # Simulation of the active 2026 conflict described in project files
            data = {
                "date": pd.date_range(start="2026-03-01", periods=14, freq='D'),
                "kinetic_raw": np.random.uniform(80, 100, 14),
                "narrative_raw": np.random.uniform(90, 100, 14)
            }
        
        df = pd.DataFrame(data)
        # Rename for the IndexCalculator
        df = df.rename(columns={"kinetic_raw": "MCT_score", "narrative_raw": "INT_score"})
        return df

    def fetch_live_acled(self, country="India"):
        """
        Fetches live conflict events from ACLED API (Free for research).
        """
        if not self.acled_key:
            return pd.DataFrame() # Fallback to synthetic if no key
            
        base_url = f"https://acleddata.com/api/acled/read?key={self.acled_key}&email={self.acled_email}"
        params = {
            "country": country,
            "limit": 100,
            "event_date": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            "event_date_where": ">"
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            return pd.DataFrame(data['data'])
        except Exception as e:
            print(f"ACLED Fetch Error: {e}")
            return pd.DataFrame()

    def generate_location_data(self, dyad, count=100):
        """
        Generates coordinates for the War Room Map based on conflict zone.
        """
        zones = {
            "India-Pakistan": [34.08, 74.79], # Kashmir
            "Russia-Ukraine": [48.37, 34.63],
            "Israel-Palestine": [31.04, 34.85],
            "Iran-Israel-USA": [32.42, 53.68]  # Central Iran
        }
        center = zones.get(dyad, [20, 0])
        
        df = pd.DataFrame({
            'lat': np.random.normal(center[0], 2, count),
            'lon': np.random.normal(center[1], 2, count)
        })
        return df, center[0], center[1], 5

    def generate_synthetic_data(self, dyad):
        """
        Fallback generator to ensure the dashboard always has data to show.
        """
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        df = pd.DataFrame({
            "date": dates,
            "MCT_score": np.random.uniform(10, 50, 30),
            "INT_score": np.random.uniform(20, 60, 30)
        })
        # Add a recent spike for visual impact
        df.iloc[-3:, df.columns.get_loc("MCT_score")] += 40
        return df

# Helper instance for app.py
def get_validation_data(scenario):
    return DataIngestor().get_validation_data(scenario)

def generate_synthetic_data(dyad):
    return DataIngestor().generate_synthetic_data(dyad)

def generate_location_data(dyad, count=100):
    return DataIngestor().generate_location_data(dyad, count)