import os
from typing import List, Dict, Any, Optional
from endee import Endee, Precision
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from embeddings import Embedder
import json

load_dotenv()

class RAGPipeline:
    def __init__(self, index_name="research_v5"):
        self.client = Endee()  # Defaults to localhost:8080
        self.embedder = Embedder()
        
        # Configure LLM Client (Gemini, xAI or OpenAI)
        google_key = os.getenv("GOOGLE_API_KEY")
        xai_key = os.getenv("XAI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if google_key:
            genai.configure(api_key=google_key)
            self.llm_provider = "google"
            
            # Use 1.5-flash or discover latest
            try:
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                flash_models = [m for m in models if 'flash' in m.lower()]
                selected_model = flash_models[0] if flash_models else models[0]
                self.llm_model = genai.GenerativeModel(selected_model)
                print(f"Using Google Gemini LLM: {selected_model}")
            except:
                self.llm_model = genai.GenerativeModel("gemini-1.5-flash")
                print("Using fallback Gemini LLM: gemini-1.5-flash")
        elif xai_key:
            self.openai_client = OpenAI(api_key=xai_key, base_url="https://api.xai.com/v1")
            self.llm_provider = "openai-compatible"
            self.llm_model = "grok-2"
            print("Using xAI (Grok) LLM")
        elif openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
            self.llm_provider = "openai"
            self.llm_model = "gpt-4o"
            print("Using OpenAI GPT-4o")
        else:
            raise ValueError("API Key (GOOGLE_API_KEY, XAI_API_KEY, or OPENAI_API_KEY) not found.")
            
        self.index_name = f"{self.llm_provider}_v1"
        # Dynamic dimension detection
        test_vec = self.embedder.get_query_embedding("test")
        self.dimension = len(test_vec)
        print(f"Detected embedding dimension: {self.dimension}")
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Checks if index exists, and creates it if not."""
        try:
            print(f"Checking for index: {self.index_name}")
            indices = self.client.list_indexes()
            
            # Note: Endee list_indexes might return objects or strings.
            # Convert objects to names if needed.
            index_names = [idx.name if hasattr(idx, 'name') else str(idx) for idx in indices]
            
            if self.index_name not in index_names:
                print(f"Creating index {self.index_name} with dim {self.dimension}")
                try:
                    self.client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        space_type="cosine",
                        precision=Precision.FLOAT32
                    )
                except Exception as ex:
                    if "Conflict" in str(ex):
                        print(f"Index {self.index_name} already exists (Handled Conflict)")
                    else:
                        raise ex
            else:
                print(f"Index {self.index_name} already exists.")
        except Exception as e:
            print(f"Error in index management: {e}")
            # Don't let index checks crash the app if they are non-fatal
            if "Conflict" not in str(e):
                raise e

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Upserts document chunks into Endee."""
        print(f"Adding {len(chunks)} chunks to Endee index {self.index_name}...")
        try:
            index = self.client.get_index(self.index_name)
            
            # Prepare content for batch embedding
            texts = [c["content"] for c in chunks]
            embeddings = self.embedder.get_embeddings(texts)
            
            upsert_data = []
            for i, chunk in enumerate(chunks):
                # Sanitize ID to be alphanumeric + underscores/dashes/dots
                source_clean = "".join(c if c.isalnum() else "_" for c in chunk['metadata']['source'])
                safe_id = f"{source_clean}_{chunk['metadata']['chunk_id']}"
                
                upsert_data.append({
                    "id": safe_id,
                    "vector": embeddings[i],
                    "meta": {
                        "text": chunk["content"],
                        "source": chunk["metadata"]["source"]
                    },
                    "filter": {
                        "source": chunk["metadata"]["source"]
                    }
                })
            
            print(f"Performing batch upsert of {len(upsert_data)} items...")
            index.upsert(upsert_data)
            print("Upsert successful.")
        except Exception as e:
            print(f"FATAL error in add_documents: {e}")
            raise e

    def retrieve(self, query: str, top_k=5, filter_sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieves relevant chunks from Endee."""
        query_vec = self.embedder.get_query_embedding(query)
        index = self.client.get_index(self.index_name)
        
        filter_query = []
        if filter_sources and len(filter_sources) == 1:
            filter_query.append({"source": {"$eq": filter_sources[0]}})
        
        results = index.query(
            vector=query_vec,
            top_k=top_k,
            filter=filter_query if filter_query else None
        )
        
        retrieved = []
        for match in results:
            # Handle both object and dict styles
            meta = match.get("meta", {}) if isinstance(match, dict) else getattr(match, "meta", {})
            score = match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)
            
            retrieved.append({
                "content": meta.get("text", ""),
                "source": meta.get("source", "Unknown"),
                "score": score
            })
        return retrieved

    def _generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Centralized generation helper."""
        if self.llm_provider == "google":
            # Gemini generation
            config = genai.types.GenerationConfig(temperature=temperature)
            response = self.llm_model.generate_content(prompt, generation_config=config)
            return response.text
        else:
            # OpenAI / xAI generation
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content

    def answer_question(self, query: str, filter_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generates an answer based on retrieved chunks."""
        context_chunks = self.retrieve(query, top_k=5, filter_sources=filter_sources)
        
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents.",
                "sources": []
            }

        context_text = "\n\n".join([f"Source: {c['source']}\nContent: {c['content']}" for c in context_chunks])
        
        prompt = f"""You are an AI Research Assistant. Answer the user's question based strictly on the provided context.
If the answer is not in the context, say you don't know based on the documents.
Always cite your sources (e.g., [Source Name]).

Context:
{context_text}

Question: {query}

Answer:"""

        answer = self._generate(prompt)
        
        return {
            "answer": answer,
            "sources": context_chunks
        }

    def compare_documents(self, doc_a: str, doc_b: str, aspect: str = "methodology") -> str:
        """Compares two documents on a specific aspect."""
        chunks_a = self.retrieve(f"{aspect} of {doc_a}", top_k=3, filter_sources=[doc_a])
        chunks_b = self.retrieve(f"{aspect} of {doc_b}", top_k=3, filter_sources=[doc_b])
        
        context_a = "\n".join([c['content'] for c in chunks_a])
        context_b = "\n".join([c['content'] for c in chunks_b])
        
        prompt = f"""Compare the {aspect} of the following two research papers:
Paper A: {doc_a}
Paper B: {doc_b}

Context for Paper A:
{context_a}

Context for Paper B:
{context_b}

Provide a structured comparison highlighting key differences and similarities."""

        return self._generate(prompt)

    def generate_literature_review(self, doc_names: List[str]) -> str:
        """Generates a literature review from multiple papers."""
        all_contexts = []
        for doc in doc_names:
            chunks = self.retrieve(f"main findings and summary of {doc}", top_k=3, filter_sources=[doc])
            summary = "\n".join([c['content'] for c in chunks])
            all_contexts.append(f"Paper: {doc}\nSummary/Context: {summary}")
        
        full_context = "\n\n---\n\n".join(all_contexts)
        
        prompt = f"""Generate a structured literature review based on the following research papers: {', '.join(doc_names)}.

The review should include:
1. Introduction
2. Key Themes
3. Comparative Insights (similarities and contradictions)
4. Conclusion

Context from papers:
{full_context}"""

        return self._generate(prompt, temperature=0.2)
