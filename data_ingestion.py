import pandas as pd
import numpy as np
import config

def generate_synthetic_data(dyad_name="India-Pakistan"):
    np.random.seed(42)  # Ensures the random numbers are the same every time
    
    # --- 1. DETERMINE TIME RANGE ---
    if dyad_name == "India-Pakistan":
        start, end = config.IND_PAK_START, config.IND_PAK_END
    elif dyad_name == "Russia-Ukraine":
        start, end = config.RUS_UKR_START, config.RUS_UKR_END
    else: # Israel-Palestine
        start, end = config.ISR_PAL_START, config.ISR_PAL_END

    # Create a list of every single day in that range
    date_range = pd.date_range(start=start, end=end)
    data = []

    print(f"Generating data for {dyad_name} ({start} to {end})...")

    # --- 2. LOOP THROUGH EVERY DAY ---
    for current_date in date_range:
        
        # A. SET BASELINES (Peace Time Values)
        if dyad_name == "Russia-Ukraine":
            kinetic_score = np.random.poisson(lam=5) # Higher base violence
            narrative_volume = int(np.random.normal(loc=100, scale=20))
            sentiment_score = np.random.normal(loc=0.3, scale=0.1)
        elif dyad_name == "Israel-Palestine":
            kinetic_score = np.random.poisson(lam=3)
            narrative_volume = int(np.random.normal(loc=60, scale=15))
            sentiment_score = np.random.normal(loc=0.4, scale=0.1)
        else: # India-Pakistan
            kinetic_score = np.random.poisson(lam=1.5)
            narrative_volume = int(np.random.normal(loc=40, scale=10))
            sentiment_score = np.random.normal(loc=0.45, scale=0.1)

        # B. INJECT CRISIS EVENTS (The "History" Override)
        # --- Scenario 1: India-Pakistan ---
        if dyad_name == "India-Pakistan":
            if current_date == pd.to_datetime(config.PULWAMA_ATTACK):
                kinetic_score += 10; narrative_volume += 500; sentiment_score = 0.1
            elif current_date == pd.to_datetime(config.BALAKOT_STRIKE):
                kinetic_score += 50; narrative_volume += 800; sentiment_score = 0.05
            elif current_date == pd.to_datetime(config.ABHINANDAN_CAPTURE):
                kinetic_score += 80; narrative_volume += 1000; sentiment_score = 0.05

        # --- Scenario 2: Russia-Ukraine ---
        elif dyad_name == "Russia-Ukraine":
            if pd.to_datetime(config.TROOP_BUILDUP_START) < current_date < pd.to_datetime(config.INVASION_START):
                narrative_volume += np.random.randint(200, 400); sentiment_score -= 0.1
            elif current_date >= pd.to_datetime(config.INVASION_START):
                kinetic_score += 500; narrative_volume += 5000; sentiment_score = 0.01

        # --- Scenario 3: Israel-Palestine ---
        elif dyad_name == "Israel-Palestine":
            if current_date == pd.to_datetime(config.OCT_7_ATTACK):
                kinetic_score += 600; narrative_volume += 8000; sentiment_score = 0.01
            elif current_date == pd.to_datetime(config.HOSPITAL_BLAST):
                kinetic_score += 50; narrative_volume += 6000; sentiment_score = 0.05
            elif current_date >= pd.to_datetime(config.GROUND_INVASION):
                kinetic_score += np.random.randint(100, 200); narrative_volume += 1000; sentiment_score = 0.1

        # C. SAVE DAY'S DATA
        sentiment_score = np.clip(sentiment_score, 0.0, 1.0)
        data.append({
            "date": current_date,
            "kinetic_score": kinetic_score,
            "narrative_volume": narrative_volume,
            "sentiment_score": sentiment_score
        })

    return pd.DataFrame(data)

def generate_location_data(dyad_name, num_points):
    """
    Generates synthetic lat/lon points for the 'War Room' map.
    """
    np.random.seed(42) # Consistent map
    
    # Define bounding boxes (Lat, Lon) + Spread for each conflict
    if dyad_name == "India-Pakistan":
        # Centered on Line of Control (LOC), Kashmir
        base_lat, base_lon = 34.0, 74.0 
        lat_spread, lon_spread = 0.5, 0.5
        zoom_level = 6
        
    elif dyad_name == "Russia-Ukraine":
        # Centered on Donbas Region
        base_lat, base_lon = 48.0, 38.0 
        lat_spread, lon_spread = 1.5, 2.0
        zoom_level = 5
        
    else: # Israel-Palestine
        # Centered on Gaza/Israel Border
        base_lat, base_lon = 31.4, 34.4 
        lat_spread, lon_spread = 0.1, 0.1
        zoom_level = 8

    # Generate random points (Gaussian distribution around the hotspot)
    lats = np.random.normal(base_lat, lat_spread, num_points)
    lons = np.random.normal(base_lon, lon_spread, num_points)
    
    return pd.DataFrame({'lat': lats, 'lon': lons}), base_lat, base_lon, zoom_level