import os
from typing import List, Optional
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class Embedder:
    def __init__(self, model: Optional[str] = None):
        google_key = os.getenv("GOOGLE_API_KEY")
        xai_key = os.getenv("XAI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if google_key:
            genai.configure(api_key=google_key)
            self.provider = "google"
            
            # Find an available embedding model
            try:
                available_embed_models = [m.name for m in genai.list_models() if 'embed' in m.name]
                if available_embed_models:
                    self.model = model or available_embed_models[0]
                else:
                    self.model = model or "models/embedding-001" # Fallback
            except:
                self.model = model or "models/embedding-001"
                
            print(f"Initialized Google Embedder with model: {self.model}")
        elif xai_key:
            self.provider = "openai-compatible"
            self.model = model or "v1-embeddings"
            self.client = OpenAI(api_key=xai_key, base_url="https://api.xai.com/v1")
            print(f"Initialized xAI Embedder with model: {self.model}")
        elif openai_key:
            self.provider = "openai"
            self.model = model or "text-embedding-3-small"
            self.client = OpenAI(api_key=openai_key)
            print(f"Initialized OpenAI Embedder with model: {self.model}")
        else:
            raise ValueError("No API Key found (GOOGLE_API_KEY, XAI_API_KEY, or OPENAI_API_KEY).")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            if self.provider == "google":
                # Google Generative AI embeddings
                result = genai.embed_content(
                    model=self.model,
                    content=texts,
                    task_type="retrieval_document"
                )
                return result['embedding']
            else:
                # OpenAI or xAI
                response = self.client.embeddings.create(input=texts, model=self.model)
                if not response.data:
                    raise ValueError(f"No embedding data received from model '{self.model}'.")
                return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Embedding error with {self.provider} model {self.model}: {e}")
            raise e

    def get_query_embedding(self, query: str) -> List[float]:
        try:
            if self.provider == "google":
                result = genai.embed_content(
                    model=self.model,
                    content=query,
                    task_type="retrieval_query"
                )
                return result['embedding']
            else:
                res = self.get_embeddings([query])
                return res[0]
        except Exception as e:
            print(f"Query embedding error: {e}")
            raise e
