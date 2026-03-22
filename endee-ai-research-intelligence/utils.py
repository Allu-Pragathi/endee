import hashlib

def generate_chunk_id(source: str, index: int) -> str:
    """Generates a unique ID for a document chunk."""
    return hashlib.md5(f"{source}_{index}".encode()).hexdigest()

def format_citation(source_name: str, snippet: str) -> str:
    """Formats a source citation for the UI."""
    return f"**Source:** {source_name}\n\n> {snippet}"

def validate_openai_key(api_key: str) -> bool:
    """Validates if the OpenAI API key is present."""
    return bool(api_key and api_key.startswith("sk-"))
