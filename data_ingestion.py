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
        # Different conflicts have different "normal" levels of violence
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
        # If the current date matches a known crisis date, force a spike.

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
        sentiment_score = np.clip(sentiment_score, 0.0, 1.0) # Keep score between 0 and 1
        data.append({
            "date": current_date,
            "kinetic_score": kinetic_score,
            "narrative_volume": narrative_volume,
            "sentiment_score": sentiment_score
        })

    return pd.DataFrame(data)

# --- 3. MAIN BLOCK (Runs when you type 'python data_ingestion.py') ---
if __name__ == "__main__":
    # Generate and save ALL three datasets
    print("🚀 Starting Data Generation Engine...")
    
    df_ind_pak = generate_synthetic_data("India-Pakistan")
    df_ind_pak.to_csv("india_pakistan_data.csv", index=False)
    print("✅ Saved 'india_pakistan_data.csv'")

    df_rus_ukr = generate_synthetic_data("Russia-Ukraine")
    df_rus_ukr.to_csv("russia_ukraine_data.csv", index=False)
    print("✅ Saved 'russia_ukraine_data.csv'")

    df_isr_pal = generate_synthetic_data("Israel-Palestine")
    df_isr_pal.to_csv("israel_palestine_data.csv", index=False)
    print("✅ Saved 'israel_palestine_data.csv'")
    
    print("\n🎉 All datasets generated successfully!")