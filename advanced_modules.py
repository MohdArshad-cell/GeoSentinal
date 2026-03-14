import random
import datetime

class AdvancedFeatures:
    def __init__(self):
        self.market_baselines = {
            "NIFTY_50": 24500.00,
            "KSE_100": 65000.00,
            "BRENT_CRUDE": 85.50,
            "GOLD_10G": 68000.00,
            "USD_INR": 83.50,
            "EU_NAT_GAS": 35.00,
            "WHEAT_FUTURES": 600.00
        }

    # --- NEW: MULTI-DOMAIN THREAT MATRIX ---
    def generate_threat_matrix(self, gpti_score, conflict_zone):
        """Generates a 5-axis threat matrix for Radar plotting."""
        # Base threat scaled by overall tension
        kinetic = min(100, max(0, (gpti_score * 100) + random.uniform(-5, 10)))
        narrative = min(100, max(0, (gpti_score * 110) + random.uniform(-10, 5)))
        economic = min(100, max(0, (gpti_score * 90) + random.uniform(0, 15)))
        cyber = min(100, max(0, (gpti_score * 120) + random.uniform(-20, 20))) 
        diplomatic = min(100, max(0, (gpti_score * 100) + random.uniform(-5, 5)))

        # Context-specific domain adjustments
        if conflict_zone == "Russia-Ukraine":
            cyber = min(100, cyber + 15)  # Known for heavy cyber warfare
        elif "India-Pakistan" in conflict_zone:
            narrative = min(100, narrative + 15) # Heavy PsyOps and social media clashes
        elif "Iran-Israel" in conflict_zone:
            cyber = min(100, cyber + 20) # Infrastructure targeting

        return {
            "Kinetic (Military)": kinetic,
            "Narrative (PsyOps)": narrative,
            "Economic Linkage": economic,
            "Cyber & Grid": cyber,
            "Diplomatic Failure": diplomatic
        }

    # 1. Context-Aware Economic Impact (Upgraded with Risk Projections)
    def get_economic_impact(self, gpti_score, conflict_zone):
        impact = {}
        market_drop = gpti_score * 0.08  
        commodity_spike = gpti_score * 0.15 
        
        if conflict_zone in ["India-Pakistan", "India-Pakistan 2019"]:
            val_nifty = self.market_baselines["NIFTY_50"] * (1 - market_drop)
            impact["Indicator_1"] = {"name": "NIFTY 50 (India)", "value": val_nifty, "change": -market_drop*100, "symbol": "₹", "worst_case": val_nifty * 0.95}
            val_kse = self.market_baselines["KSE_100"] * (1 - (market_drop * 1.5))
            impact["Indicator_2"] = {"name": "KSE-100 (Pakistan)", "value": val_kse, "change": -(market_drop*1.5)*100, "symbol": "₨", "worst_case": val_kse * 0.92}
            val_usd = self.market_baselines["USD_INR"] * (1 + (gpti_score * 0.03))
            impact["Indicator_3"] = {"name": "USD/INR", "value": val_usd, "change": (gpti_score*0.03)*100, "symbol": "₹", "worst_case": val_usd * 1.02}
            
        elif conflict_zone == "Russia-Ukraine":
            val_gas = self.market_baselines["EU_NAT_GAS"] * (1 + commodity_spike)
            impact["Indicator_1"] = {"name": "EU Natural Gas", "value": val_gas, "change": commodity_spike*100, "symbol": "€", "worst_case": val_gas * 1.15}
            val_wheat = self.market_baselines["WHEAT_FUTURES"] * (1 + (commodity_spike * 0.8))
            impact["Indicator_2"] = {"name": "Wheat Futures", "value": val_wheat, "change": (commodity_spike*0.8)*100, "symbol": "$", "worst_case": val_wheat * 1.10}
            val_brent = self.market_baselines["BRENT_CRUDE"] * (1 + (commodity_spike * 0.5))
            impact["Indicator_3"] = {"name": "Brent Crude", "value": val_brent, "change": (commodity_spike*0.5)*100, "symbol": "$", "worst_case": val_brent * 1.08}
            
        else: 
            val_brent = self.market_baselines["BRENT_CRUDE"] * (1 + commodity_spike)
            impact["Indicator_1"] = {"name": "Brent Crude Oil", "value": val_brent, "change": commodity_spike*100, "symbol": "$", "worst_case": val_brent * 1.12}
            val_gold = self.market_baselines["GOLD_10G"] * (1 + (gpti_score * 0.10))
            impact["Indicator_2"] = {"name": "Gold (Safe Haven)", "value": val_gold, "change": (gpti_score*0.10)*100, "symbol": "₹", "worst_case": val_gold * 1.05}
            impact["Indicator_3"] = {"name": "Global Shipping Freight", "value": 2500 * (1 + commodity_spike), "change": commodity_spike*100, "symbol": "$", "worst_case": 2500 * (1 + commodity_spike) * 1.20}
            
        return impact

    # 2. Early Warning System
    def generate_alerts(self, current_gpti, gpti_trend):
        alerts = []
        if current_gpti >= 0.85:
            alerts.append("🔴 DEFCON 2: Extreme Kinetic Risk. Diplomatic channels failing.")
        elif current_gpti >= 0.70:
            alerts.append("🟠 ALERT: GPTI Threshold Exceeded. Military mobilization likely.")
        if gpti_trend > 0.10:
            alerts.append("⚠️ TREND VIOLATION: Flash-Crisis detected via anomalous acceleration.")
            
        status = "Accelerating" if gpti_trend > 0.05 else "Stable" if abs(gpti_trend) <= 0.05 else "De-escalating"
        return alerts, status

    # 3. Modern Information Integrity
    def analyze_information_integrity(self, text):
        text_lower = text.lower()
        score = 100 
        bot_signals = ["retweet", "trending", "viral", "copy paste"]
        state_media_signals = ["unprovoked", "glorious", "righteous", "destroy the enemy", "liberation"]
        ai_signals = ["deepfake", "ai-generated", "unverified footage", "synthetic audio"]
        
        flags = []
        if any(w in text_lower for w in bot_signals):
            score -= 15; flags.append("Bot Network Activity")
        if any(w in text_lower for w in state_media_signals):
            score -= 20; flags.append("State-Media Rhetoric")
        if any(w in text_lower for w in ai_signals):
            score -= 40; flags.append("Suspected AI/Deepfake")
            
        status = "Verified Organic" if score >= 85 else ("Disputed" if score >= 60 else "Highly Manipulated (PsyOp)")
        return {"integrity_score": max(0, score), "narrative_status": status, "flags": flags}

    # 4. Proactive Leading Indicators
    def analyze_leading_indicators(self, headline, conflict_zone):
        leading_keywords = ["notam", "airspace closure", "satellite imagery", "blood supplies", "evacuation", "strike group", "awacs", "cyberattack", "ddos"]
        is_leading = any(kw in headline.lower() for kw in leading_keywords)
        if "Israel" in conflict_zone and ("strait of hormuz" in headline.lower() or "gps jamming" in headline.lower()): is_leading = True
        if "Pakistan" in conflict_zone and ("leaves cancelled" in headline.lower() or "artillery moved" in headline.lower()): is_leading = True
        return is_leading

    # 5. Panic Index
    def get_public_panic_index(self, gpti_score):
        base_panic = gpti_score * 100 
        volatility = random.uniform(-10, 15)
        panic = min(100, max(0, base_panic + volatility))
        trending_terms = random.sample(["draft age", "nearest bunker", "flight cancellations", "grid outage", "stock crash", "emergency radio"], 2)
        return panic, trending_terms