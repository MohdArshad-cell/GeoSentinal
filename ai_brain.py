import os
import json
from groq import Groq
from dotenv import load_dotenv # Add this library

# 1. Load the variables from a hidden .env file
load_dotenv() 

# 2. Access the key securely
API_KEY = os.getenv("GROQ_API_KEY") 

if not API_KEY:
    raise ValueError("❌ Error: GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=API_KEY)

def analyze_intelligence(headline, source):
    """
    Feeds a headline to Llama 3 and requests a tactical assessment.
    Returns: JSON object with Summary, Hostility Score, and Options.
    """
    print(f"🧠 Analyzing Signal: '{headline}'...")
    
    # System Prompt: Defines the AI's Persona
    system_prompt = """
    You are a Strategic Defense Analyst. 
    Analyze the given news headline regarding a conflict zone.
    Return a valid JSON object with the following fields:
    1. "risk_score": A float between 0.0 (Peace) and 1.0 (War).
    2. "summary": A 1-sentence military-style situation report.
    3. "strategic_options": A list of 3 short tactical options for the government.
    
    Do not output any text other than the JSON.
    """
    
    user_prompt = f"Source: {source}\nHeadline: {headline}"

    try:
        completion = client.chat.completions.create(
            # UPDATED MODEL: Using the latest Llama 3.3 (Smarter & Fast)
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0, # Deterministic behavior
            stream=False,
            response_format={"type": "json_object"} # Force valid JSON
        )
        
        # Parse result
        result = json.loads(completion.choices[0].message.content)
        return result

    except Exception as e:
        print(f"❌ Brain Failure: {e}")
        return None

# --- TEST DRIVE ---
if __name__ == "__main__":
    # Test with a dummy headline
    fake_live_news = "Heavy artillery exchange reported in Uri Sector, villagers evacuated."
    source = "Times of India"
    
    print("\n--- INITIATING AI ANALYSIS ---")
    intel = analyze_intelligence(fake_live_news, source)
    
    if intel:
        print("\n📊 COMMANDER'S DASHBOARD OUTPUT:")
        print(f"⚠️ Risk Level: {intel['risk_score'] * 100:.1f}%")
        print(f"📝 SitRep: {intel['summary']}")
        print("\n🎯 STRATEGIC OPTIONS:")
        for i, opt in enumerate(intel['strategic_options'], 1):
            print(f"   {i}. {opt}")