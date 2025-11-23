import config
# We import 'pipeline' to use pre-made AI models easily
from transformers import pipeline

class NarrativeAI:
    def __init__(self):
        print("🧠 Initializing the AI Engine...")
        
        # 1. Load the "Judge" (Sentiment Analyzer)
        # We use DistilBERT because it's fast and accurate.
        # It downloads a pre-trained brain from HuggingFace.
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def llm_relevance_filter(self, text_snippet):
        """
        THE BOUNCER: Decides if text is relevant.
        In a real app, this calls GPT-4.
        """
        # [cite_start]The EXACT prompt from your research paper [cite: 644]
        system_prompt = """
        You are a geopolitical analyst specializing in South Asia. 
        Your task is to determine if the following text is relevant to genuine 
        geopolitical tension between India and Pakistan. 
        Geopolitical tension refers to military, diplomatic, or political conflicts. 
        It does NOT include cultural events, sports, or routine trade. 
        Does this article discuss genuine India-Pakistan geopolitical tension? 
        Answer ONLY with 'Yes' or 'No'.
        """
        
        # --- SIMULATION LOGIC ---
        # Since we are in "Dev Mode" without a live GPT-4 key, 
        # we simulate the Bouncer's logic with simple keywords for now.
        keywords_to_ignore = ["cricket", "match", "movie", "song", "festival"]
        
        # If any "noise" word is in the text, the Bouncer says NO.
        for word in keywords_to_ignore:
            if word in text_snippet.lower():
                return False # Kick it out!
                
        return True # Let it in!

    def get_sentiment_score(self, text_snippet):
        """
        THE JUDGE: Reads the text and gives a score.
        """
        # Ask the AI model to read the text
        result = self.sentiment_analyzer(text_snippet)[0]
        
        # The model returns 'POSITIVE' or 'NEGATIVE' with a confidence score.
        label = result['label']
        score = result['score']
        
        # We convert this to a simple number:
        # 0.0 = Very Hostile/Negative
        # 1.0 = Very Peaceful/Positive
        if label == 'NEGATIVE':
            return 1 - score  # Example: High confidence negative = low score (0.1)
        else:
            return score      # Example: High confidence positive = high score (0.9)

# --- TEST DRIVE ---
# Let's test our Brain on two fake headlines to see if it works.
if __name__ == "__main__":
    brain = NarrativeAI()
    
    # Example 1: Irrelevant Noise
    news1 = "India beats Pakistan in thrilling T20 Cricket Match!"
    
    # Example 2: Real Tension
    news2 = "Military forces exchange fire at the Line of Control, tensions rise."
    
    print(f"\n📰 News 1: '{news1}'")
    if brain.llm_relevance_filter(news1):
        print("❌ Bouncer: Let it in.")
    else:
        print("✅ Bouncer: Blocked! (Irrelevant)")
        
    print(f"\n📰 News 2: '{news2}'")
    if brain.llm_relevance_filter(news2):
        print("✅ Bouncer: Let it in. (Relevant)")
        score = brain.get_sentiment_score(news2)
        print(f"⚖️ Judge: Sentiment Score is {score:.2f} (Low means Hostile)")