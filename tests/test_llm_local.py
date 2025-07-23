import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import SETTINGS


def setup_dummy_transformers(monkeypatch, captured):
    class DummyTokenizer:
        def __call__(self, text, return_tensors=None):
            import torch
            captured['prompt'] = text
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, tokens, skip_special_tokens=True):
            return "dummy output"

    class DummyModel:
        def to(self, device):
            self.device = device

        def eval(self):
            pass

        def generate(self, **kwargs):
            import torch
            return torch.tensor([[1, 2, 3]])

    dummy = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: DummyTokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda path: DummyModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", dummy)


def test_local_model_loading_and_generation(monkeypatch):
    captured = {}
    setup_dummy_transformers(monkeypatch, captured)
    monkeypatch.setitem(SETTINGS, "LLM_model", {"local_path": "dummy", "api_key": "", "api_type": "gemini"})
    import llm
    importlib.reload(llm)
    model = llm.get_llm_model()
    assert isinstance(model, llm.LocalLLM)
    text = llm.call_llm(model, "Hello")
    assert text.strip() == "dummy output"
    assert captured["prompt"].startswith("Your job is to process and format data.")


def test_call_llm_with_instruction(monkeypatch):
    captured = {}
    setup_dummy_transformers(monkeypatch, captured)
    monkeypatch.setitem(SETTINGS, "LLM_model", {"local_path": "dummy", "api_key": "", "api_type": "gemini"})
    import llm
    importlib.reload(llm)
    model = llm.get_llm_model()
    text = llm.call_llm(model, "Hello", instruction="TEST INSTRUCT")
    assert captured["prompt"].startswith("TEST INSTRUCT")

