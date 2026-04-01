import feedparser
import pandas as pd
import requests
from datetime import datetime
from urllib.parse import quote

class NewsIntel:
    def __init__(self):
        # Google News RSS is the most stable free source for search queries
        self.base_url = "https://news.google.com/rss/search?q="

    def fetch_news(self, query, limit=5):
        """
        Scrapes global news feeds for a specific geopolitical query.
        Returns a cleaned Pandas DataFrame.
        """
        print(f"📡 Accessing Global RSS Feeds for query: '{query}'...")
        
        # Encode query for URL (e.g., space becomes %20)
        encoded_query = quote(query)
        full_url = f"{self.base_url}{encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            # Parse the RSS feed
            feed = feedparser.parse(full_url)
            entries = feed.entries[:limit]
            
            processed_data = []
            for entry in entries:
                # Clean and structure the news packet
                processed_data.append({
                    "title": entry.title,
                    "source": entry.source.title if hasattr(entry, 'source') else "Global Intel",
                    "date": entry.published,
                    "link": entry.link
                })

            if not processed_data:
                return pd.DataFrame()

            df = pd.DataFrame(processed_data)
            # Standardizing date format for GPTI processing
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')
            
            print(f"✅ Extraction Successful: {len(df)} signals acquired.")
            return df

        except Exception as e:
            print(f"❌ Scraper Failure: {e}")
            return pd.DataFrame()

# --- COMMANDER'S FIELD TEST ---
if __name__ == "__main__":
    agent = NewsIntel()
    # Test query for your project
    test_signals = agent.fetch_news("India Pakistan border tension", limit=3)
    if not test_signals.empty:
        print("\n--- SAMPLE INTELLIGENCE ---")
        print(test_signals[['title', 'source', 'date']])
    else:
        print("📭 No signals found. Check your internet connection.")