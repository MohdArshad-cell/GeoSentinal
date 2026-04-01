import os
from dotenv import load_dotenv

# .env file se keys load karne ke liye
load_dotenv()

# --- API CONFIGURATIONS ---
# ACLED Data (Kinetic Pillar)
ACLED_EMAIL = os.getenv("ACLED_EMAIL", "your_email@example.com")
ACLED_API_KEY = os.getenv("ACLED_API_KEY", "your_acled_key")

# Gemini AI (Relevance Filter)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_key")


# --- APP SETTINGS ---
APP_NAME = "GeoSentinal Commander"
REFRESH_RATE = 60  # seconds