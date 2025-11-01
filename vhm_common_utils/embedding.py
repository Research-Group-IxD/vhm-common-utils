import hashlib
import numpy as np
import requests
from typing import List

from .config import settings

# ... (Copy and refactor all embedding functions from the original utils.py) ...
# ... (deterministic_embed, portkey_embed, ollama_embed, get_embedding, etc.) ...

# IMPORTANT: Replace all `os.getenv` calls with `settings.*` attributes.
# For example, `os.getenv("OLLAMA_BASE_URL")` becomes `settings.ollama_base_url`.

def get_embedding(text: str) -> List[float]:
    """Main embedding function that respects EMBEDDING_MODEL env var."""
    # ... (refactor this function to use settings.embedding_model)
    pass
