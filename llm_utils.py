import google.generativeai as genai
from Start import SETTINGS

# Only Gemini is supported right now

def get_llm_model():
    """Load the LLM model based on settings. Defaults to the Gemini API."""
    cfg = SETTINGS.get("LLM_model", {})
    api_key = cfg.get("api_key", "")
    api_type = cfg.get("api_type", "gemini").lower()
    local_path = cfg.get("local_path", "")

    if api_key:
        if api_type != "gemini":
            raise ValueError(f"Unsupported API type: {api_type}")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-pro")
    if local_path:
        print(f"⚠️ Local LLM path '{local_path}' provided but loading is not implemented.")
        return None
    print("🔑 Please set your Gemini API key in settings.json. See https://ai.google.dev/gemini-api")
    return None


def call_llm(model, prompt_text, temperature=0.6):
    """Send ``prompt_text`` to the provided LLM model."""
    if not model:
        return "❌ Generative model not initialized."
    try:
        response = model.generate_content(
            prompt_text,
            generation_config={
                "temperature": temperature,
                "top_p": 1.0,
                "max_output_tokens": 1000,
            },
        )
        return response.text.strip()
    except Exception as e:
        return f"💥 Gemini query failed: {e}"
