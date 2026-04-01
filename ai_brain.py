import os
import json
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. GEMINI MISSION CONTROL CONFIG
# ==========================================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Using 1.5 Flash for speed (Bouncer logic)
model = genai.GenerativeModel('gemini-1.5-flash', 
                              generation_config={"response_mime_type": "application/json"})

class StrategicBrain:
    def __init__(self):
        self.log_file = "live_intelligence_log.json"

    def analyze_intelligence(self, headline, source, current_zone):
        """
        LLM 'Bouncer' + Strategic Analyst: 
        Filters noise and generates actionable sitreps.
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
        
        try:
            response = model.generate_content(prompt)
            analysis = json.loads(response.text)
            
            # Metadata add karo logging ke liye
            analysis['headline'] = headline
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['zone'] = current_zone
            
            # Agar relevant hai toh log karo taaki UI pe dikhe
            if analysis['is_relevant']:
                self._log_intelligence(analysis)
                
            return analysis
        except Exception as e:
            print(f"❌ Brain Failure: {e}")
            return self._get_fallback_response(headline)

    def _log_intelligence(self, data):
        """Logs relevant data for the Streamlit Live Feed."""
        logs = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                try: logs = json.load(f)
                except: logs = []
        
        # Newest intelligence first
        logs.insert(0, data)
        # Keep only last 50 entries
        with open(self.log_file, 'w') as f:
            json.dump(logs[:50], f, indent=4)

    def _get_fallback_response(self, headline):
        """Emergency logic if API fails or quota hits."""
        return {
            "is_relevant": True,
            "risk_score": 0.5,
            "sitrep": f"AUTO-ANALYSIS: Potential escalation detected in '{headline[:30]}...'",
            "integrity_score": 50,
            "flags": ["API_OFFLINE"],
            "strategic_options": ["Maintain Defensive Posture", "Verify via Secondary Source"]
        }

# UI / Sentinel Compatibility Wrapper
def analyze_with_gemini(headline, source, current_zone="Global"):
    brain = StrategicBrain()
    return brain.analyze_intelligence(headline, source, current_zone)