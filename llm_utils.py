from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from config import SETTINGS

# Only Gemini is supported right now

def get_llm_model():
    """Load the LLM client based on settings. Defaults to the Gemini API."""
    cfg = SETTINGS.get("LLM_model", {})
    api_key = cfg.get("api_key", "")
    api_type = cfg.get("api_type", "gemini").lower()
    local_path = cfg.get("local_path", "")

    if api_key:
        if api_type != "gemini":
            raise ValueError(f"Unsupported API type: {api_type}")
        # Using Client for more versatile use cases like file uploads
        client = genai.Client(api_key=api_key)
        return client
    if local_path:
        print(f"⚠️ Local LLM path '{local_path}' provided but loading is not implemented.")
        return None
    print("🔑 Please set your Gemini API key in settings.json. Free as of 7-21-2025 See https://ai.google.dev/gemini-api")
    return None


def call_llm(client, prompt_text, temperature=None, max_tokens=None, top_p=None):
    """Send ``prompt_text`` to the provided LLM client."""
    if not client:
        return "❌ Generative model client not initialized."
    
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    api_cfg = SETTINGS.get("api_settings", {})
    if temperature is None:
        temperature = api_cfg.get("temperature", 0.6)
    if max_tokens is None:
        max_tokens = api_cfg.get("max_output_tokens", 5000)
    top_p = api_cfg.get("top_p", 1.0)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
            ),
        )
        raw_text = (
            response.candidates[0].content.parts[0].text
            if response and response.candidates
            and response.candidates[0].content.parts
            and hasattr(response.candidates[0].content.parts[0], 'text')
            else None
        )
        if raw_text is None:
            return "💥 Gemini query failed: No valid response from model."
        return raw_text.strip()
    except Exception as e:
        return f"💥 Gemini query failed: {e}"




def load_embedding_model(model_path: str | None):
    """Load a ``SentenceTransformer`` model from ``model_path`` or download a default."""
    if not model_path:
        print(
            "encoder_model_path is not set; downloading 'sentence-transformers/all-MiniLM-L6-v2'"
        )
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model_path)

