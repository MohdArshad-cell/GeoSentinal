import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_scenario_window(scenario_name, peak_date, mct_peak, int_peak):
    """
    Ek specific 1100-day (3 years) window banata hai scenario ke liye.
    Isse PCA calculation stable rehti hai.
    """
    # 1100 days ka window taaki 36-month rolling logic fail na ho
    start_date = peak_date - timedelta(days=550)
    dates = pd.date_range(start=start_date, periods=1100, freq='D')
    
    # Baseline scores (Peaceful period)
    mct = np.random.uniform(5, 15, 1100)
    intel = np.random.uniform(8, 22, 1100)
    
    df = pd.DataFrame({
        "date": dates,
        "MCT_score": mct,
        "INT_score": intel,
        "scenario": scenario_name,
        "event": "Baseline Monitoring"
    })
    
    # Peak Crisis Injection
    # Peak ke 10 din pehle se tension badhna shuru hoti hai
    peak_mask = (df['date'] >= peak_date) & (df['date'] <= peak_date + timedelta(days=15))
    df.loc[peak_mask, 'MCT_score'] = mct_peak + np.random.uniform(-5, 5, peak_mask.sum())
    df.loc[peak_mask, 'INT_score'] = int_peak + np.random.uniform(-5, 5, peak_mask.sum())
    df.loc[peak_mask, 'event'] = f"Major Escalation: {scenario_name}"
    
    return df

def generate_benchmarks():
    # 1. India-Pakistan 2019 (Pulwama-Balakot focus)
    ind_pak = create_scenario_window("India-Pakistan 2019", datetime(2019, 2, 14), 95.0, 100.0)
    # Specific dates refinement
    ind_pak.loc[ind_pak['date'] == '2019-02-14', 'event'] = "Pulwama Terror Attack"
    ind_pak.loc[ind_pak['date'] == '2019-02-26', 'event'] = "Balakot Airstrikes"
    ind_pak.loc[ind_pak['date'] == '2019-02-27', 'event'] = "Operation Swift Retort / Dogfight"

    # 2. Russia-Ukraine 2022 (Invasion focus)
    rus_ukr = create_scenario_window("Russia-Ukraine 2022", datetime(2022, 2, 24), 100.0, 95.0)
    rus_ukr.loc[rus_ukr['date'] == '2022-02-24', 'event'] = "Full-Scale Invasion Initiated"

    # 3. Iran-Israel 2026 (Epic Fury - Fictional Scenario)
    # Peak date: 8th April 2026 (Your project deadline)
    iran_isr = create_scenario_window("Iran-Israel 2026", datetime(2026, 4, 1), 88.0, 92.0)
    iran_isr.loc[iran_isr['date'] == '2026-04-01', 'event'] = "Anomalous Infrastructure Failure / Cyber-Strike"

    # Sabko merge karo
    final_csv = pd.concat([ind_pak, rus_ukr, iran_isr])
    
    # Save to the new benchmark file
    filename = "geosentinal_benchmarks.csv"
    final_csv.to_csv(filename, index=False)
    print(f"✅ Success: '{filename}' generated with {len(final_csv)} tactical data points.")

if __name__ == "__main__":
    generate_benchmarks()