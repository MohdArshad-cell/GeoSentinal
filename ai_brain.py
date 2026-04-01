import os
import json
import time
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class StrategicBrain:
    def __init__(self):
        self.log_file = "live_intelligence_log.json"
        # ELITE: Get multiple keys from env and split into a list
        raw_keys = os.getenv("GEMINI_KEYS", "")
        self.keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.current_key_index = 0
        
        if not self.keys:
            print("❌ CRITICAL: No API keys found in .env (GEMINI_KEYS)")
        else:
            self._configure_genai()

    def _configure_genai(self):
        """Configures the Gemini API with the current key."""
        current_key = self.keys[self.current_key_index]
        genai.configure(api_key=current_key)
        
        # Using 1.5-Flash for robust JSON handling and high speed
        # Ensure your model name matches your available API access
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash-lite', 
            generation_config={"response_mime_type": "application/json"}
        )
        print(f"📡 API GATEWAY ROTATED: Using Key_{self.current_key_index + 1}")

    def _rotate_key(self):
        """Switches to the next available API key when quota is hit."""
        if len(self.keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            self._configure_genai()
            return True
        return False

    def analyze_intelligence(self, headline, source, current_zone):
        """
        ULTIMATE UPGRADE: High-Fidelity Signal Analysis with Failover.
        """
        
        # --- GOD-TIER STRATEGIC PROMPT ---
        # Fixed: Embedded current date and strict military grading rubric
        prompt = f"""
        CURRENT DATE: April 01, 2026
        ROLE: Senior OSINT Intelligence Director (GeoSentinel Command).
        TASK: Perform high-fidelity signal analysis on the provided data packet.

        INPUT DATA:
        - HEADLINE: "{headline}"
        - SOURCE: {source}
        - THEATER: {current_zone}

        ### 1. STRICT RELEVANCY PROTOCOL (IS_RELEVANT)
        IMMEDIATELY REJECT (false) if:
        - News is dated before 2026.
        - Subject is sports, entertainment, celebrity gossip, or routine local trade.
        - Headline is a general historical summary or Wikipedia-style fact.
        ACCEPT (true) ONLY if:
        - Direct military movement, airstrikes, naval maneuvers, or cyber-warfare.
        - High-stakes diplomatic ultimatums or economic sanctions.
        - Infrastructure sabotage or nuclear/strategic posturing.

        ### 2. QUANTITATIVE SCORING RUBRIC
        - RISK_SCORE (0.0 to 1.0):
            - 0.0-0.3: Routine drills, low-level rhetoric, diplomatic meetings.
            - 0.4-0.6: Mobilization, sanctions, border skirmishes, drone intercepts.
            - 0.7-0.9: Active combat, airstrikes, declaration of war, naval blockades.
            - 1.0: Nuclear escalation, state collapse, or global conflict trigger.
        - INTEGRITY_SCORE (0-100):
            - 90+: Reuters, AP, AFP, Official Govt Portals.
            - 70-89: BBC, Al Jazeera, CNN, Major National Dailies.
            - <50: Social media leaks, unverified blogs, or state-run propaganda nodes.

        ### 3. OUTPUT SPECIFICATIONS (JSON ONLY)
        Return a MINIFIED JSON with these EXACT keys:
        {{
            "is_relevant": boolean,
            "risk_score": float,
            "sitrep": "A cold, technical 1-sentence briefing of the tactical situation.",
            "integrity_score": int,
            "flags": ["list of propaganda markers like: Loaded Language, Unverified, Bias"],
            "strategic_options": [
                "Intelligence: (Next monitoring target)",
                "Tactical: (Immediate military/security response)",
                "Diplomatic: (International posture required)"
            ]
        }}
        """

        max_retries = len(self.keys) * 2 
        
        for attempt in range(max_retries):
            try:
                # --- TACTICAL PACING ---
                # 25 seconds gap is elite for keeping free-tier keys alive all day
                print(f"⏳ Tactical Delay: 25s (API Safety Protocol)...")
                time.sleep(25) 

                response = self.model.generate_content(prompt)
                
                # Check for empty response
                if not response.text:
                    raise ValueError("Empty response from Gemini")

                analysis = json.loads(response.text)
                
                # Appending metadata for Dashboard rendering
                analysis['headline'] = headline
                analysis['timestamp'] = datetime.now().isoformat()
                analysis['zone'] = current_zone
                
                if analysis.get('is_relevant'):
                    self._log_intelligence(analysis)
                
                return analysis

            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg:
                    print(f"⚠️ KEY_{self.current_key_index + 1} LIMIT REACHED.")
                    if self._rotate_key():
                        print(f"🔄 Failover Successful. Retrying with Node_{self.current_key_index + 1}...")
                        continue 
                    else:
                        print("🚫 ALL API NODES EXHAUSTED. Forcing 60s cooldown...")
                        time.sleep(60)
                else:
                    print(f"❌ INTERNAL BRAIN ERROR: {e}")
                    return self._get_fallback_response(headline, current_zone)

        return self._get_fallback_response(headline, current_zone)

    def _log_intelligence(self, data):
        """Maintains the live log for the Streamlit feed."""
        logs = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                try: 
                    logs = json.load(f)
                except: 
                    logs = []
        
        # Keep it clean: Insert at top, cap at 50 entries
        logs.insert(0, data)
        with open(self.log_file, 'w') as f:
            json.dump(logs[:50], f, indent=4)

    def _get_fallback_response(self, headline, zone):
        """Ensures the dashboard doesn't break if API fails."""
        return {
            "is_relevant": True,
            "risk_score": 0.5,
            "sitrep": f"AUTO-ANALYSIS: Signal detected in {zone}. API offline/busy.",
            "integrity_score": 50,
            "flags": ["SYSTEM_FALLBACK_ACTIVE"],
            "strategic_options": ["Manual Source Verification", "Reset API Gateway"],
            "headline": headline,
            "timestamp": datetime.now().isoformat(),
            "zone": zone
        }

# Singleton instance to persist rotation state across the sentinel cycle
brain_instance = StrategicBrain()

def analyze_intelligence(headline, source, current_zone="Global"):
    return brain_instance.analyze_intelligence(headline, source, current_zone)