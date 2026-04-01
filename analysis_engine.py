import config
from transformers import pipeline
import ai_brain 
import pandas as pd

class NarrativeAI:
    def __init__(self):
        # 1. Hardware acceleration check (GPU agar hai toh use karo)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1 # Set to 0 if you have NVIDIA GPU
        )
        self.confidence_threshold = 0.85 # [PRO] Sirf high-confidence signals uthayenge

    def process_batch_intelligence(self, df, current_zone):
        """
        [ELITE FEATURE] Poore news batch ko ek saath analyze karta hai.
        Relevance (Gemini) + Sentiment (DistilBERT) + Confidence Filtering.
        """
        results = []
        
        for _, row in df.iterrows():
            # Step 1: Smart Filter via Gemini
            is_relevant, intel = ai_brain.analyze_intelligence(row['title'], row['source'], current_zone)
            
            if is_relevant:
                # Step 2: Local Sentiment Engine
                sentiment = self.sentiment_analyzer(row['title'])[0]
                
                # Step 3: Confidence Filtering
                # Agar AI confused hai, toh hum 'Noise' risk nahi lenge.
                if sentiment['score'] >= self.confidence_threshold:
                    # Final Tension Score (Inverting if positive)
                    final_score = sentiment['score'] if sentiment['label'] == 'NEGATIVE' else (1 - sentiment['score'])
                    
                    results.append({
                        "date": row['date'],
                        "headline": row['title'],
                        "raw_intensity": final_score,
                        "confidence": sentiment['score'],
                        "sitrep": intel.get('sitrep', 'No SitRep available'),
                        "reasoning": intel.get('reasoning', 'Strategic context analyzed') # XAI Feature
                    })
        
        return pd.DataFrame(results)

# --- COMMANDER'S TEST DRIVE ---
if __name__ == "__main__":
    engine = NarrativeAI()
    test_data = pd.DataFrame([
        {"date": "2026-04-01", "title": "Massive artillery mobilization detected at Punjab border", "source": "OSINT_Sat"},
        {"date": "2026-04-01", "title": "India-Pak cricket fans celebrate peace match", "source": "Sports_Daily"}
    ])
    
    print("🚀 PRO-PROCESSING INITIATED...")
    intelligence_report = engine.process_batch_intelligence(test_data, "India-Pakistan 2019")
    print(intelligence_report)