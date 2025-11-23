import pandas as pd
import numpy as np
import config

def generate_synthetic_data(dyad_name="India-Pakistan"):
    np.random.seed(42)
    
    # --- SET TIME RANGE ---
    if dyad_name == "India-Pakistan":
        start, end = config.IND_PAK_START, config.IND_PAK_END
    elif dyad_name == "Russia-Ukraine":
        start, end = config.RUS_UKR_START, config.RUS_UKR_END
    else: # Israel-Palestine
        start, end = config.ISR_PAL_START, config.ISR_PAL_END

    date_range = pd.date_range(start=start, end=end)
    data = []

    print(f"Generating data for {dyad_name}...")

    for current_date in date_range:
        # --- BASELINES ---
        if dyad_name == "Russia-Ukraine":
            kinetic_score = np.random.poisson(lam=5)
            narrative_volume = int(np.random.normal(loc=100, scale=20))
            sentiment_score = np.random.normal(loc=0.3, scale=0.1)
        elif dyad_name == "Israel-Palestine":
            # Constant low-level tension (higher than Ind-Pak, lower than Rus-Ukr)
            kinetic_score = np.random.poisson(lam=3)
            narrative_volume = int(np.random.normal(loc=60, scale=15))
            sentiment_score = np.random.normal(loc=0.4, scale=0.1)
        else: # India-Pakistan
            kinetic_score = np.random.poisson(lam=1.5)
            narrative_volume = int(np.random.normal(loc=40, scale=10))
            sentiment_score = np.random.normal(loc=0.45, scale=0.1)

        # --- SCENARIO 1: INDIA-PAKISTAN (2019) ---
        if dyad_name == "India-Pakistan":
            if current_date == pd.to_datetime(config.PULWAMA_ATTACK):
                kinetic_score += 10; narrative_volume += 500; sentiment_score = 0.1
            elif current_date == pd.to_datetime(config.BALAKOT_STRIKE):
                kinetic_score += 50; narrative_volume += 800; sentiment_score = 0.05
            elif current_date == pd.to_datetime(config.ABHINANDAN_CAPTURE):
                kinetic_score += 80; narrative_volume += 1000; sentiment_score = 0.05

        # --- SCENARIO 2: RUSSIA-UKRAINE (2022) ---
        elif dyad_name == "Russia-Ukraine":
            if pd.to_datetime(config.TROOP_BUILDUP_START) < current_date < pd.to_datetime(config.INVASION_START):
                narrative_volume += np.random.randint(200, 400); sentiment_score -= 0.1
            elif current_date >= pd.to_datetime(config.INVASION_START):
                kinetic_score += 500; narrative_volume += 5000; sentiment_score = 0.01

        # --- SCENARIO 3: ISRAEL-PALESTINE (2023) ---
        elif dyad_name == "Israel-Palestine":
            # Oct 7: Instant, massive shock to BOTH pillars
            if current_date == pd.to_datetime(config.OCT_7_ATTACK):
                kinetic_score += 600         # Unprecedented single-day violence
                narrative_volume += 8000     # Global media saturation
                sentiment_score = 0.01       # Maximum hostility
            
            # The Hospital Blast (Narrative Spike)
            elif current_date == pd.to_datetime(config.HOSPITAL_BLAST):
                kinetic_score += 50          # Ongoing fighting
                narrative_volume += 6000     # Huge controversy/protests
                sentiment_score = 0.05
            
            # Ground Invasion (Sustained War)
            elif current_date >= pd.to_datetime(config.GROUND_INVASION):
                kinetic_score += np.random.randint(100, 200) # Heavy urban combat
                narrative_volume += 1000     # Sustained coverage
                sentiment_score = 0.1

        # Clip and Save
        sentiment_score = np.clip(sentiment_score, 0.0, 1.0)
        data.append({
            "date": current_date,
            "kinetic_score": kinetic_score,
            "narrative_volume": narrative_volume,
            "sentiment_score": sentiment_score
        })

    return pd.DataFrame(data)