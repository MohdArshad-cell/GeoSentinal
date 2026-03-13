import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Gemini Setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def analyze_with_gemini(headline, source):
    """
    LLM 'Bouncer' Logic: PDF ke according relevance check karta hai.
    """
    # PDF based Prompt
    prompt = f"""
    You are a Strategic Defense Analyst. 
    Task: Is this headline about India-Pakistan, Iran-Israel, or Russia-Ukraine GEOPOLITICAL tension?
    Geopolitical tension includes: Military moves, diplomatic rows, or war.
    It EXCLUDES: Sports, movies, and routine trade.
    
    Headline: "{headline}"
    Source: {source}
    
    Return a JSON object:
    {{
        "is_relevant": true/false,
        "risk_score": float (0.0 to 1.0),
        "summary": "1 sentence sitrep",
        "strategic_options": ["Option 1", "Option 2", "Option 3"]
    }}
    """
    try:
        response = model.generate_content(prompt)
        # Cleaning the response text to get valid JSON
        raw_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(raw_text)
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None