import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_grok():
    api_key = os.getenv("XAI_API_KEY")
    print(f"DEBUG: Key found: {api_key[:10]}...")
    
    # Try multiple base_url configurations
    urls = ["https://api.xai.com/v1", "https://api.xai.com/v1/"]
    
    for url in urls:
        print(f"\n--- Testing with URL: {url} ---")
        client = OpenAI(api_key=api_key, base_url=url)
        
        try:
            print("Listing models...")
            models = client.models.list()
            ids = [m.id for m in models.data]
            print(f"Available models: {ids}")
            
            # Use the first model in the list!
            if ids:
                target_model = ids[0]
                print(f"Trying embedding with model: {target_model}")
                res = client.embeddings.create(input=["test"], model=target_model)
                print(f"SUCCESS! Vector dim: {len(res.data[0].embedding)}")
                return
        except Exception as e:
            print(f"Failed with {url}: {e}")

if __name__ == "__main__":
    test_grok()
