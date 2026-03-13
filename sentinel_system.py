import time
import json
import pandas as pd
from datetime import datetime
from news_fetcher import NewsIntel
from ai_brain import analyze_intelligence

# --- CONFIGURATION ---
CONFLICT_TARGET = "India Pakistan military tension"
LOG_FILE = "live_intelligence_log.json"
# Set to 1800 seconds (30 mins) for production. 
# For testing now, let's set it to 60 seconds so you see it work immediately.
UPDATE_INTERVAL_SECONDS = 60 

def run_sentinel_cycle():
    """Run one full cycle of intelligence gathering and analysis."""
    print(f"\n⚡ [WAKE UP] SENTINEL PROTOCOL INITIATED: {datetime.now()}")
    
    # 1. ACQUIRE TARGETS
    intel_agent = NewsIntel()
    print(f"📡 Scanning global feeds for: '{CONFLICT_TARGET}'...")
    raw_news = intel_agent.fetch_news(CONFLICT_TARGET, limit=2) # Keep limit low for testing
    
    if raw_news.empty:
        print("❌ No signals found.")
        return

    # 2. ANALYZE SIGNALS
    processed_intel = []
    print(f"🧠 Processing {len(raw_news)} signals through Llama 3...")
    
    for index, row in raw_news.iterrows():
        headline = row['title']
        source = row['source']
        date = row['date']
        
        # Call AI Brain
        analysis = analyze_intelligence(headline, source)
        
        if analysis:
            intel_packet = {
                "timestamp": date,
                "source": source,
                "headline": headline,
                "link": row['link'],
                "risk_score": analysis.get('risk_score', 0),
                "sitrep": analysis.get('summary', "No data"),
                "options": analysis.get('strategic_options', [])
            }
            processed_intel.append(intel_packet)
            time.sleep(1) # Polite delay

    # 3. ATOMIC UPDATE (Save to Database)
    if processed_intel:
        try:
            # Load existing history
            try:
                with open(LOG_FILE, 'r') as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                history = []
            
            # Combine new + old (Preventing duplicates logic omitted for simplicity)
            # In a real app, you'd check IDs. Here we just prepend.
            history = processed_intel + history
            
            # Save
            with open(LOG_FILE, 'w') as f:
                json.dump(history, f, indent=4)
                
            print(f"✅ [SUCCESS] Intelligence updated. Saved to {LOG_FILE}")
            
        except Exception as e:
            print(f"❌ Storage Failure: {e}")
    else:
        print("⚠️ No valid intelligence generated.")

# --- THE ENDLESS LOOP ---
if __name__ == "__main__":
    print(f"🛡️ GEO-SENTINEL DAEMON ONLINE.")
    print(f"🕒 System will wake up every {UPDATE_INTERVAL_SECONDS} seconds.")
    print("----------------------------------------------------")
    
    while True:
        try:
            run_sentinel_cycle()
        except Exception as e:
            print(f"💥 CRITICAL ERROR: {e}")
        
        print(f"💤 Mission Complete. Sleeping for {UPDATE_INTERVAL_SECONDS}s...")
        time.sleep(UPDATE_INTERVAL_SECONDS)