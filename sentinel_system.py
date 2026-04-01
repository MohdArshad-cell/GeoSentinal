import time
import json
import os
import pandas as pd
from datetime import datetime
from news_fetcher import NewsIntel
from ai_brain import analyze_intelligence

# --- Update this in sentinel_system.py ---
CONFLICT_THEATERS = {
    "INDIA-PAKISTAN": "India Pakistan military LoC news latest 2026",
    "RUSSIA-UKRAINE": "Russia Ukraine war offensive frontline updates 2026",
    "ISRAEL-PALESTINE": "Israel Gaza military operation Hezbollah escalation latest",
    "IRAN-ISRAEL-US": "Iran Israel US drone strike missile tension news 2026",
    "SUDAN CONFLICT": "Sudan army RSF fighting Khartoum updates today"
}

LOG_FILE = "live_intelligence_log.json"
UPDATE_INTERVAL_SECONDS = 300 
MAX_LOG_ENTRIES = 100 

def run_sentinel_cycle():
    """Run a synchronized intelligence sweep across all global theaters."""
    print(f"\n📡 [SCAN] GLOBAL SWEEP INITIATED: {datetime.now().strftime('%H:%M:%S')}")
    
    intel_agent = NewsIntel()
    all_processed_intel = []

    # 1. MULTI-THEATER ACQUISITION
    for zone, query in CONFLICT_THEATERS.items():
        print(f"🔍 Scanning Theater: {zone}...")
        raw_news = intel_agent.fetch_news(query, limit=2) # Reduced limit to stay safe
        
        if raw_news.empty:
            continue

        # 2. STRATEGIC ANALYSIS
        for _, row in raw_news.iterrows():
            headline = row['title']
            print(f"🧠 Gemini analyzing signal from {row['source']}...")
            
            analysis = analyze_intelligence(headline, row['source'], zone)
            
            if analysis and analysis.get('is_relevant'):
                intel_packet = {
                    "timestamp": row['date'],
                    "zone": zone,
                    "source": row['source'],
                    "headline": headline,
                    "link": row['link'],
                    "risk_score": analysis.get('risk_score', 0),
                    "sitrep": analysis.get('sitrep', "No SitRep"),
                    "integrity_score": analysis.get('integrity_score', 50),
                    "options": analysis.get('strategic_options', [])
                }
                all_processed_intel.append(intel_packet)
            
            # --- CRITICAL FIX: PACING ---
            # 25 seconds gap ensures we stay under 10 RPM (Requests Per Minute)
            print(f"⏳ Tactical Cool-down: 25s (Steady Flow Mode)")
            time.sleep(25) 

    # 3. SMART LOGGING
    if all_processed_intel:
        update_database(all_processed_intel)
    else:
        print("⚠️ No new critical signals detected in this cycle.")

def update_database(new_packets):
    """Atomic update with duplicate prevention and size capping."""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                try:
                    history = json.load(f)
                except:
                    history = []
        else:
            history = []

        existing_headlines = {item['headline'] for item in history}
        unique_new_packets = [p for p in new_packets if p['headline'] not in existing_headlines]

        if unique_new_packets:
            updated_history = unique_new_packets + history
            updated_history = updated_history[:MAX_LOG_ENTRIES]

            with open(LOG_FILE, 'w') as f:
                json.dump(updated_history, f, indent=4)
            print(f"✅ Database Updated: {len(unique_new_packets)} new unique signals logged.")
        else:
            print("ℹ️ Sweep complete: No new unique signals found.")

    except Exception as e:
        print(f"❌ Database Corruption Error: {e}")

if __name__ == "__main__":
    print("""
    #########################################
    #       GEO-SENTINEL DAEMON ONLINE      #
    #    MODE: MULTI-THEATER SURVEILLANCE   #
    #########################################
    """)
    while True:
        try:
            run_sentinel_cycle()
        except Exception as e:
            print(f"💥 CRITICAL SYSTEM FAILURE: {e}")
        
        print(f"💤 Cycle Complete. Next scan in {UPDATE_INTERVAL_SECONDS}s...")
        time.sleep(UPDATE_INTERVAL_SECONDS)