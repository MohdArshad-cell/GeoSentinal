import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_strategic_window(scenario_name, peak_date, events_dict, base_lat, base_lon):
    """
    Creates a detailed 1100-day tactical window with multi-domain scoring.
    """
    # 3-year rolling baseline for PCA stability
    start_date = peak_date - timedelta(days=550)
    dates = pd.date_range(start=start_date, periods=1100, freq='D')
    
    # Baseline Metrics
    mct = np.random.uniform(5, 12, 1100)
    intel = np.random.uniform(8, 18, 1100)
    
    df = pd.DataFrame({
        "date": dates,
        "MCT_score": mct,
        "INT_score": intel,
        "scenario": scenario_name,
        "event": "Baseline Monitoring",
        "lat": base_lat + np.random.normal(0, 0.5, 1100),
        "lon": base_lon + np.random.normal(0, 0.5, 1100)
    })
    
    for date_str, info in events_dict.items():
        event_date = pd.to_datetime(date_str)
        if event_date in df['date'].values:
            mask = (df['date'] == event_date)
            df.loc[mask, ['MCT_score', 'INT_score', 'event']] = [info['mct'], info['int'], info['label']]
            if 'lat' in info: df.loc[mask, 'lat'] = info['lat']
            if 'lon' in info: df.loc[mask, 'lon'] = info['lon']
            
            # Narrative Lead-Lag logic: Narrative spikes BEFORE kinetic
            pre_mask = (df['date'] >= event_date - timedelta(days=3)) & (df['date'] < event_date)
            df.loc[pre_mask, 'INT_score'] += np.random.uniform(25, 45)
            
            # Trend Acceleration for Early Warning
            if info['mct'] > 80:
                accel_mask = (df['date'] >= event_date - timedelta(days=1)) & (df['date'] <= event_date)
                df.loc[accel_mask, 'MCT_score'] *= 1.3 
            
    return df

def generate_all():
    # --- 1. INDIA-PAKISTAN 2019 (11 Events) ---
    ip_events = {
        "2019-02-14": {"mct": 90, "int": 95, "label": "Pulwama Terror Attack", "lat": 33.87, "lon": 74.89},
        "2019-02-15": {"mct": 15, "int": 85, "label": "India Withdraws MFN Status", "lat": 28.61, "lon": 77.20},
        "2019-02-18": {"mct": 65, "int": 75, "label": "Major Encounter: Pulwama Sector", "lat": 33.88, "lon": 74.90},
        "2019-02-19": {"mct": 10, "int": 80, "label": "ICJ Kulbhushan Jadhav Hearing Begins", "lat": 52.08, "lon": 4.31},
        "2019-02-21": {"mct": 5, "int": 90, "label": "Water Diversion Threat (Indus Treaty)", "lat": 31.14, "lon": 75.34},
        "2019-02-22": {"mct": 40, "int": 70, "label": "Pak Military Pre-emptive Alert", "lat": 33.68, "lon": 73.04},
        "2019-02-26": {"mct": 95, "int": 100, "label": "Balakot Airstrikes (Mirage-2000)", "lat": 34.46, "lon": 73.35},
        "2019-02-27": {"mct": 100, "int": 100, "label": "Op Swift Retort / Aerial Dogfight", "lat": 33.38, "lon": 74.30},
        "2019-02-28": {"mct": 5, "int": 95, "label": "Pakistan Announces Pilot Release", "lat": 33.72, "lon": 73.06},
        "2019-03-01": {"mct": 5, "int": 98, "label": "Wing Commander Abhinandan Returns", "lat": 31.60, "lon": 74.45},
        "2019-03-05": {"mct": 45, "int": 65, "label": "Indian Submarine Detection Claim", "lat": 24.86, "lon": 67.00}
    }
    ind_pak = create_strategic_window("India-Pakistan 2019", datetime(2019, 2, 14), ip_events, 34.08, 74.79)

    # --- 2. RUSSIA-UKRAINE 2022-2024 (11 Events) ---
    ru_events = {
        "2022-02-21": {"mct": 30, "int": 85, "label": "DPR/LPR Sovereignty Recognition", "lat": 48.01, "lon": 37.80},
        "2022-02-24": {"mct": 100, "int": 100, "label": "Full-Scale Invasion / Missile Barrage", "lat": 50.45, "lon": 30.52},
        "2022-03-02": {"mct": 95, "int": 80, "label": "Siege of Kherson City", "lat": 46.63, "lon": 32.61},
        "2022-04-02": {"mct": 40, "int": 100, "label": "Bucha Massacre Evidence Revealed", "lat": 50.55, "lon": 30.21},
        "2022-04-14": {"mct": 90, "int": 95, "label": "Cruiser Moskva Sunk (Neptune Missile)", "lat": 45.17, "lon": 31.83},
        "2022-05-20": {"mct": 95, "int": 90, "label": "Fall of Mariupol / Azovstal Surrender", "lat": 47.09, "lon": 37.54},
        "2022-09-21": {"mct": 60, "int": 95, "label": "Russia Announces Partial Mobilization", "lat": 55.75, "lon": 37.61},
        "2022-09-26": {"mct": 20, "int": 98, "label": "Nord Stream Pipeline Sabotage", "lat": 54.87, "lon": 15.41},
        "2022-10-08": {"mct": 85, "int": 95, "label": "Kerch Bridge Explosion", "lat": 45.21, "lon": 36.51},
        "2023-06-06": {"mct": 90, "int": 98, "label": "Kakhovka Dam Destruction", "lat": 46.78, "lon": 33.37},
        "2024-02-17": {"mct": 85, "int": 80, "label": "Capture of Avdiivka", "lat": 48.13, "lon": 37.74}
    }
    rus_ukr = create_strategic_window("Russia-Ukraine 2022", datetime(2022, 2, 24), ru_events, 48.37, 34.63)

    # --- 3. ISRAEL-PALESTINE 2023-2024 (10 Events) ---
    ip_pal_events = {
        "2023-10-07": {"mct": 100, "int": 100, "label": "Hamas Incursion / Iron Swords Initiated", "lat": 31.50, "lon": 34.46},
        "2023-10-13": {"mct": 20, "int": 90, "label": "Gaza City Evacuation Order", "lat": 31.50, "lon": 34.46},
        "2023-10-17": {"mct": 40, "int": 100, "label": "Al-Ahli Hospital Explosion", "lat": 31.50, "lon": 34.44},
        "2023-10-27": {"mct": 95, "int": 95, "label": "Gaza Strip Ground Invasion Begins", "lat": 31.50, "lon": 34.46},
        "2023-11-24": {"mct": 10, "int": 95, "label": "Temporary Humanitarian Truce", "lat": 31.28, "lon": 34.25},
        "2024-04-01": {"mct": 30, "int": 95, "label": "World Central Kitchen Convoy Strike", "lat": 31.34, "lon": 34.30},
        "2024-07-31": {"mct": 50, "int": 100, "label": "Haniyeh Assassination (Tehran)", "lat": 35.68, "lon": 51.38},
        "2024-09-17": {"mct": 85, "int": 100, "label": "Hezbollah Pager Attack / Cyber-Kinetic", "lat": 33.88, "lon": 35.50},
        "2024-09-27": {"mct": 95, "int": 100, "label": "Nasrallah Elimination (Beirut Strike)", "lat": 33.84, "lon": 35.51},
        "2024-10-17": {"mct": 90, "int": 100, "label": "Yahya Sinwar Killed in Rafah", "lat": 31.28, "lon": 34.25}
    }
    isr_pal = create_strategic_window("Israel-Palestine 2023", datetime(2023, 10, 7), ip_pal_events, 31.04, 34.85)

    # --- 4. IRAN-ISRAEL-US 2020-2026 (11 Events) ---
    iiu_events = {
        "2020-01-03": {"mct": 95, "int": 100, "label": "Gen. Soleimani Assassination (Baghdad)", "lat": 33.26, "lon": 44.23},
        "2020-01-08": {"mct": 90, "int": 85, "label": "Iran Missile Strike: Al-Asad Base", "lat": 33.91, "lon": 42.43},
        "2020-11-27": {"mct": 40, "int": 90, "label": "Scientist Fakhrizadeh Assassinated", "lat": 35.68, "lon": 51.38},
        "2021-04-11": {"mct": 25, "int": 85, "label": "Natanz Nuclear Facility Sabotage", "lat": 33.72, "lon": 51.72},
        "2024-01-28": {"mct": 85, "int": 90, "label": "Tower 22 Attack (Jordan Border)", "lat": 33.31, "lon": 38.70},
        "2024-04-01": {"mct": 90, "int": 95, "label": "Damascus Consulate Strike", "lat": 33.51, "lon": 36.29},
        "2024-04-13": {"mct": 98, "int": 100, "label": "Op True Promise: Drone/Missile Swarm", "lat": 32.08, "lon": 34.78},
        "2024-04-19": {"mct": 80, "int": 85, "label": "Isfahan Counter-Strike", "lat": 32.65, "lon": 51.66},
        "2024-10-01": {"mct": 95, "int": 100, "label": "Iran Ballistic Missile Barrage", "lat": 31.04, "lon": 34.85},
        "2024-10-26": {"mct": 100, "int": 95, "label": "Israel Strikes Iranian Air Defenses", "lat": 35.68, "lon": 51.38},
        "2026-04-01": {"mct": 98, "int": 100, "label": "EPIC FURY: Infrastructure Peak", "lat": 35.68, "lon": 51.38}
    }
    iran_isr_us = create_strategic_window("Iran-Israel-US 2026", datetime(2020, 1, 3), iiu_events, 32.42, 53.68)

    # --- 5. SUDAN CONFLICT 2023-2024 (10 Events) ---
    sudan_events = {
        "2023-04-15": {"mct": 95, "int": 90, "label": "SAF vs RSF Civil War Erupts", "lat": 15.50, "lon": 32.55},
        "2023-04-22": {"mct": 20, "int": 85, "label": "Mass Foreign Diplomat Evacuation", "lat": 15.58, "lon": 32.53},
        "2023-05-11": {"mct": 10, "int": 80, "label": "Jeddah Declaration Signed", "lat": 21.50, "lon": 39.16},
        "2023-06-08": {"mct": 90, "int": 95, "label": "El Geneina Mass Killings Reported", "lat": 13.44, "lon": 22.44},
        "2023-10-26": {"mct": 85, "int": 80, "label": "RSF Seizes Nyala Strategic Base", "lat": 12.05, "lon": 24.88},
        "2023-12-18": {"mct": 80, "int": 85, "label": "RSF Takes Wad Madani", "lat": 14.40, "lon": 33.51},
        "2024-01-02": {"mct": 5, "int": 85, "label": "Addis Ababa Peace Declaration", "lat": 9.01, "lon": 38.75},
        "2024-03-12": {"mct": 85, "int": 80, "label": "Army Retakes Omdurman Radio HQ", "lat": 15.63, "lon": 32.47},
        "2024-05-10": {"mct": 90, "int": 90, "label": "Siege of El Fasher Intensifies", "lat": 13.61, "lon": 25.35},
        "2024-08-14": {"mct": 15, "int": 85, "label": "Geneva Peace Talks Commenced", "lat": 46.20, "lon": 6.14}
    }
    sudan_war = create_strategic_window("Sudan Conflict 2023", datetime(2023, 4, 15), sudan_events, 15.50, 32.55)

    # FINAL CONCATENATION (Merged ALL 5 Scenarios)
    final_csv = pd.concat([ind_pak, rus_ukr, isr_pal, sudan_war, iran_isr_us])
    final_csv.to_csv("geosentinal_benchmarks.csv", index=False)
    print("🌍 GLOBAL INTELLIGENCE: Benchmarks generated with 50+ strategic events.")

if __name__ == "__main__":
    generate_all()