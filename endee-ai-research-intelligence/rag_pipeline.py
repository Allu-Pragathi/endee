import os
from typing import List, Dict, Any, Optional
from endee import Endee, Precision
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from embeddings import Embedder
import json
import numpy as np

load_dotenv()

class RAGPipeline:
    def __init__(self, index_name="research_v5"):
        self.embedder = Embedder()
        self.mode = "endee" # Default mode
        
        # Configure Endee Client with Fallback
        try:
            self.client = Endee()  # Defaults to localhost:8080
            # Test connection with a simple list_indexes call
            self.client.list_indexes()
            print("Successfully connected to Endee Vector Database.")
        except Exception as e:
            print(f"Warning: Could not connect to Endee server ({e}). Switching to Simulation Mode.")
            self.mode = "mock"
            self.mock_db = [] # Local in-memory storage for deployment/demo

        # Configure LLM Client (Gemini, xAI or OpenAI)
        google_key = os.getenv("GOOGLE_API_KEY")
        xai_key = os.getenv("XAI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if google_key:
            genai.configure(api_key=google_key)
            self.llm_provider = "google"
            try:
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                flash_models = [m for m in models if 'flash' in m.lower()]
                selected_model = flash_models[0] if flash_models else models[0]
                self.llm_model = genai.GenerativeModel(selected_model)
            except:
                self.llm_model = genai.GenerativeModel("gemini-1.5-flash")
        elif xai_key:
            self.openai_client = OpenAI(api_key=xai_key, base_url="https://api.xai.com/v1")
            self.llm_provider = "openai-compatible"
            self.llm_model = "grok-2"
        elif openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
            self.llm_provider = "openai"
            self.llm_model = "gpt-4o"
        else:
            raise ValueError("API Key not found.")
            
        self.index_name = f"{self.llm_provider}_v1"
        
        # Get dimensions
        test_vec = self.embedder.get_query_embedding("test")
        self.dimension = len(test_vec)
        
        if self.mode == "endee":
            self._ensure_index_exists()

    def _ensure_index_exists(self):
        try:
            indices = self.client.list_indexes()
            index_names = [idx.name if hasattr(idx, 'name') else str(idx) for idx in indices]
            if self.index_name not in index_names:
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    space_type="cosine",
                    precision=Precision.FLOAT32
                )
        except Exception as e:
            if "Conflict" not in str(e):
                print(f"Index check skipped: {e}")

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Upserts document chunks into Endee or Mock DB."""
        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.get_embeddings(texts)
        
        upsert_data = []
        for i, chunk in enumerate(chunks):
            source_clean = "".join(c if c.isalnum() else "_" for c in chunk['metadata']['source'])
            item = {
                "id": f"{source_clean}_{chunk['metadata']['chunk_id']}",
                "vector": embeddings[i],
                "meta": {
                    "text": chunk["content"],
                    "source": chunk["metadata"]["source"]
                },
                "filter": {
                    "source": chunk["metadata"]["source"]
                }
            }
            upsert_data.append(item)

        if self.mode == "endee":
            index = self.client.get_index(self.index_name)
            index.upsert(upsert_data)
        else:
            # Simple in-memory storage for Simulation Mode
            self.mock_db.extend(upsert_data)

    def retrieve(self, query: str, top_k=5, filter_sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieves relevant chunks from Endee or Mock DB."""
        query_vec = self.embedder.get_query_embedding(query)
        
        if self.mode == "endee":
            index = self.client.get_index(self.index_name)
            filter_query = [{"source": {"$eq": filter_sources[0]}}] if filter_sources and len(filter_sources) == 1 else None
            results = index.query(vector=query_vec, top_k=top_k, filter=filter_query)
        else:
            # Basic Cosine Similarity Simulation for In-Memory Mode
            results = []
            for item in self.mock_db:
                # Apply filter
                if filter_sources and item["filter"]["source"] not in filter_sources:
                    continue
                
                # Compute Cosine Similarity
                v1 = np.array(query_vec)
                v2 = np.array(item["vector"])
                score = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                
                match = item.copy()
                match["score"] = float(score)
                results.append(match)
            
            # Sort by score and take top_k
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        retrieved = []
        for match in results:
            meta = match.get("meta", {}) if isinstance(match, dict) else getattr(match, "meta", {})
            score = match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)
            retrieved.append({
                "content": meta.get("text", ""),
                "source": meta.get("source", "Unknown"),
                "score": score
            })
        return retrieved

    def _generate(self, prompt: str, temperature: float = 0.0) -> str:
        if self.llm_provider == "google":
            config = genai.types.GenerationConfig(temperature=temperature)
            response = self.llm_model.generate_content(prompt, generation_config=config)
            return response.text
        else:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content

    def answer_question(self, query: str, filter_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        context_chunks = self.retrieve(query, top_k=5, filter_sources=filter_sources)
        if not context_chunks:
            return {"answer": "I couldn't find any relevant information.", "sources": []}

        context_text = "\n\n".join([f"Source: {c['source']}\nContent: {c['content']}" for c in context_chunks])
        prompt = f"""You are an AI Research Assistant. Answer strictly based on the provided context.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"""
        answer = self._generate(prompt)
        return {"answer": answer, "sources": context_chunks}

    def compare_documents(self, doc_a: str, doc_b: str, aspect: str = "methodology") -> str:
        chunks_a = self.retrieve(f"{aspect} of {doc_a}", top_k=5, filter_sources=[doc_a])
        chunks_b = self.retrieve(f"{aspect} of {doc_b}", top_k=5, filter_sources=[doc_b])
        context_a = "\n".join([c['content'] for c in chunks_a])
        context_b = "\n".join([c['content'] for c in chunks_b])
        prompt = f"""Compare the {aspect} of:\nPaper A: {doc_a}\nPaper B: {doc_b}\n\nContext A:\n{context_a}\n\nContext B:\n{context_b}\n\nProvide structural comparison."""
        return self._generate(prompt)

    def generate_literature_review(self, doc_names: List[str]) -> str:
        all_contexts = []
        for doc in doc_names:
            chunks = self.retrieve(f"content of {doc}", top_k=8, filter_sources=[doc])
            summary = "\n".join([c['content'] for c in chunks])
            all_contexts.append(f"Paper: {doc}\nContext: {summary}")
        
        full_context = "\n\n---\n\n".join(all_contexts)
        prompt = f"""Generate a structured literature review for: {', '.join(doc_names)}.\n\nContext:\n{full_context}"""
        return self._generate(prompt, temperature=0.2)
