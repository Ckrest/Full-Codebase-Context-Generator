from config import SETTINGS
from lazy_loader import lazy_import
import json


def get_example_json(n: int) -> str:
    """Return comma separated example JSON values."""
    return ",\n  ".join(f'"query suggestion {i+1}"' for i in range(n))


PROMPT_GEN_TEMPLATE = """You are an expert in semantic code search.

Given the user’s problem statement below, generate {n} recommended queries that are each:
- Short (5-12 words)
- Technically focused
- Different in angle or phrasing
- Useful for embedding-based code search

Respond only with a JSON list of strings — no commentary, no markdown.

# Problem Statement
{problem}

# Output Format
[
  {get_example_json}
]"""


PROMPT_NEW_QUERY = """You previously generated the following recommended queries for a problem. Now generate a single, new query that:
- Is different in phrasing or focus
- Still relevant to the original problem
- Is useful for code search
- Is short and specific

Respond only with the query string.

# Problem Statement
{problem}

# Existing Queries
{existing}

# New Query"""

# Only Gemini is supported right now

def get_llm_model():
    """Load the LLM client based on settings. Supports Gemini or a local model."""
    cfg = SETTINGS.get("LLM_model", {})
    api_key = cfg.get("api_key", "")
    api_type = cfg.get("api_type", "gemini").lower()
    local_path = cfg.get("local_path", "")
    model_type = cfg.get("model_type", "causal")
    device = cfg.get("device", "cpu")

    if local_path:
        trans = lazy_import("transformers")
        torch = lazy_import("torch")
        try:
            tokenizer = trans.AutoTokenizer.from_pretrained(local_path)
            model_cls = trans.AutoModelForCausalLM if model_type == "causal" else trans.AutoModel
            model = model_cls.from_pretrained(local_path)
            model.to(device)
            model.eval()
            return {"model": model, "tokenizer": tokenizer, "device": device}
        except Exception as exc:
            print(f"❌ Failed to load local model from {local_path}: {exc}")
            return None

    if api_key:
        if api_type != "gemini":
            raise ValueError(f"Unsupported API type: {api_type}")
        genai = lazy_import("google.generativeai")
        client = genai.Client(api_key=api_key)
        return client

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

    if isinstance(client, dict) and "model" in client and "tokenizer" in client:
        torch = lazy_import("torch")
        tokenizer = client["tokenizer"]
        model = client["model"]
        device = client.get("device", "cpu")
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if text.startswith(prompt_text):
            text = text[len(prompt_text):]
        return text.strip()

    types = lazy_import("google.generativeai.types")
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





