import os
import json
import time  # NEW: Added for retry delay
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. GEMINI MISSION CONTROL CONFIG
# ==========================================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Using 2.5 flash-lite for speed
model = genai.GenerativeModel('gemini-2.5-flash-lite', 
                              generation_config={"response_mime_type": "application/json"})

class StrategicBrain:
    def __init__(self):
        self.log_file = "live_intelligence_log.json"

    def analyze_intelligence(self, headline, source, current_zone):
        """
        ELITE UPDATE: Added Retry logic for 429 Quota errors.
        """
        prompt = f"""
        ROLE: Senior Strategic Defense Analyst (GeoSentinel System).
        CONTEXT: Analyzing live OSINT feed for the {current_zone} conflict.
        
        TASK: Evaluate the following headline for geopolitical relevance and propaganda risk.
        HEADLINE: "{headline}"
        SOURCE: {source}

        CRITERIA:
        - RELEVANT: Troop movements, airstrikes, sanctions, diplomatic threats, infrastructure sabotage.
        - NOISE: Sports, celebrity news, routine trade, cultural festivals (unless used as cover).

        RETURN JSON ONLY:
        {{
            "is_relevant": boolean,
            "risk_score": float (0.0 to 1.0),
            "sitrep": "Professional military-grade 1-sentence summary",
            "integrity_score": int (0-100, where 100 is highly credible),
            "flags": ["list", "of", "propaganda", "markers"],
            "strategic_options": ["Option A", "Option B", "Option C"]
        }}
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                analysis = json.loads(response.text)
                
                analysis['headline'] = headline
                analysis['timestamp'] = datetime.now().isoformat()
                analysis['zone'] = current_zone
                
                if analysis['is_relevant']:
                    self._log_intelligence(analysis)
                    
                return analysis

            except Exception as e:
                if "429" in str(e):
                    # ELITE: Exponential Backoff - Quota exceeded toh wait karo
                    wait_time = (attempt + 1) * 15 
                    print(f"⚠️ Quota Full (429). Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue # Try again
                else:
                    print(f"❌ Brain Failure: {e}")
                    return self._get_fallback_response(headline)

        return self._get_fallback_response(headline)

    def _log_intelligence(self, data):
        """Logs relevant data for the Streamlit Live Feed."""
        logs = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                try: logs = json.load(f)
                except: logs = []
        
        logs.insert(0, data)
        with open(self.log_file, 'w') as f:
            json.dump(logs[:50], f, indent=4)

    def _get_fallback_response(self, headline):
        return {
            "is_relevant": True,
            "risk_score": 0.5,
            "sitrep": f"AUTO-ANALYSIS: Potential escalation detected in '{headline[:30]}...'",
            "integrity_score": 50,
            "flags": ["API_OFFLINE_OR_QUOTA_FULL"],
            "strategic_options": ["Verify via Secondary Source", "Monitor Local Feeds"]
        }

# Sentinel System compatibility
def analyze_intelligence(headline, source, current_zone="Global"):
    brain = StrategicBrain()
    return brain.analyze_intelligence(headline, source, current_zone)

def analyze_with_gemini(headline, source, current_zone="Global"):
    return analyze_intelligence(headline, source, current_zone)