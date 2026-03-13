import feedparser
import pandas as pd
import dateparser
from datetime import datetime
import urllib.parse
import time

class NewsIntel:
    def __init__(self):
        # We use the US edition (ceid=US:en) for English global coverage
        self.base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
        
    def fetch_news(self, query, limit=10):
        """
        Fetches live news headlines from Google News RSS.
        """
        # 1. Encode the query safely (e.g., "India Pakistan" -> "India%20Pakistan")
        encoded_query = urllib.parse.quote(query)
        feed_url = self.base_url.format(encoded_query)
        
        print(f"📡 Establishing Uplink... Scanning for: '{query}'")
        
        # 2. Parse the RSS feed (This goes to the internet)
        feed = feedparser.parse(feed_url)
        
        news_data = []
        
        # 3. Check if we got anything
        if not feed.entries:
            print(f"⚠️ No signals detected for '{query}'. Check internet connection.")
            return pd.DataFrame()

        # 4. Loop through entries and clean data
        for entry in feed.entries[:limit]:
            # Clean up the date (RSS dates are messy)
            pub_date = dateparser.parse(entry.published)
            date_str = pub_date.strftime("%Y-%m-%d %H:%M:%S") if pub_date else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            item = {
                "date": date_str,
                "title": entry.title,
                "link": entry.link,
                "source": entry.source.title if 'source' in entry else "Unknown Source",
                # Some feeds have a summary, some don't. We grab it if it exists.
                "summary": entry.summary if 'summary' in entry else ""
            }
            news_data.append(item)
            
        print(f"✅ Intelligence Gathered: {len(news_data)} reports secured.")
        return pd.DataFrame(news_data)

# --- COMMANDER'S TEST DRIVE ---
# Run this file directly to test if the "Eyes" are working.
if __name__ == "__main__":
    intel = NewsIntel()
    
    # Target: India-Pakistan Conflict
    target_query = "India Pakistan border conflict"
    
    df = intel.fetch_news(target_query, limit=5)
    
    if not df.empty:
        print("\n--- INCOMING INTEL FEED (LIVE) ---")
        # Print just the date, source, and title for a clean view
        print(df[['date', 'source', 'title']].to_string(index=False))
        
        # Save to CSV to prove we have data
        filename = "live_intel_dump.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Data archived to '{filename}'")
    else:
        print("❌ Mission Failed: No Intel Found.")