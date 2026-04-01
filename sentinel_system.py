import time
import json
import os
import pandas as pd
from datetime import datetime
from news_fetcher import NewsIntel
from ai_brain import analyze_intelligence

# --- ELITE CONFIGURATION ---
# Ab ye system saare theaters ko monitor karega
CONFLICT_THEATERS = {
    "India-Pakistan 2019": "India Pakistan military border tension LoC",
    "Russia-Ukraine 2022": "Russia Ukraine war offensive missile strike",
    "Israel-Palestine 2023": "Gaza Israel conflict IDF Hamas military",
    "Iran-Israel-US 2026": "Iran Israel US military escalation drone strike",
    "Sudan Conflict 2023": "Sudan army RSF fighting Khartoum"
}

LOG_FILE = "live_intelligence_log.json"
UPDATE_INTERVAL_SECONDS = 300  # 5 Minutes (Elite systems don't spam, they observe)
MAX_LOG_ENTRIES = 100 # Memory management

def run_sentinel_cycle():
    """Run a synchronized intelligence sweep across all global theaters."""
    print(f"\n📡 [SCAN] GLOBAL SWEEP INITIATED: {datetime.now().strftime('%H:%M:%S')}")
    
    intel_agent = NewsIntel()
    all_processed_intel = []

    # 1. MULTI-THEATER ACQUISITION
    for zone, query in CONFLICT_THEATERS.items():
        print(f"🔍 Scanning Theater: {zone}...")
        raw_news = intel_agent.fetch_news(query, limit=3)
        
        if raw_news.empty:
            continue

        # 2. STRATEGIC ANALYSIS (Sync with Gemini)
        for _, row in raw_news.iterrows():
            headline = row['title']
            
            # [FIX] Typos hata diye, ab ye Gemini ka use confirm karega
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
            time.sleep(2) # API Rate-limiting safety

    # 3. SMART LOGGING (With Duplicate Prevention)
    if all_processed_intel:
        update_database(all_processed_intel)
    else:
        print("⚠️ No critical signals detected in this cycle.")

def update_database(new_packets):
    """Atomic update with duplicate prevention and size capping."""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = []

        # [ELITE] Duplicate Prevention based on Headline
        existing_headlines = {item['headline'] for item in history}
        unique_new_packets = [p for p in new_packets if p['headline'] not in existing_headlines]

        if unique_new_packets:
            # Newest at the top
            updated_history = unique_new_packets + history
            # Memory Management: Keep only latest entries
            updated_history = updated_history[:MAX_LOG_ENTRIES]

            with open(LOG_FILE, 'w') as f:
                json.dump(updated_history, f, indent=4)
            print(f"✅ Database Updated: {len(unique_new_packets)} new signals logged.")
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