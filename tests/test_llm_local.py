import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import SETTINGS
import llm

class DummyTokenizer:
    def encode(self, text, return_tensors="pt"):
        import torch
        return torch.tensor([[1, 2, 3]])

    def decode(self, tokens, skip_special_tokens=True):
        return "prompt response"

class DummyModel:
    def to(self, device):
        return self
    def eval(self):
        pass
    def generate(self, input_ids, max_new_tokens=20, do_sample=False, temperature=0.0, top_p=1.0):
        import torch
        return torch.tensor([[1, 2, 3, 4]])


def test_local_llm_load_and_generate(monkeypatch):
    monkeypatch.setitem(SETTINGS["LLM_model"], "local_path", "dummy")
    monkeypatch.setitem(SETTINGS["LLM_model"], "api_key", "")
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: DummyTokenizer())
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda p: DummyModel())

    client = llm.get_llm_model()
    out = llm.call_llm(client, "hello")
    assert out == "prompt response"
