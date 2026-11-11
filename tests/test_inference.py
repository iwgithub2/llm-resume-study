import sys
from types import SimpleNamespace

import pytest

from resume_study.config import Model
from resume_study.inference import ChatBackend, InferenceError, VLLMBackend, preflight, snapshot


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "official:" + messages[0]["content"]

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(range(len(text)))


def test_chat_format_and_no_silent_truncation(study):
    cfg, _, _ = study
    engine = ChatBackend()
    engine.cfg = cfg.model
    engine.tokenizer = Tokenizer()
    assert engine.format([{"role": "user", "content": "hello"}]) == "official:hello"
    with pytest.raises(InferenceError, match="No truncation"):
        engine.validate_context(["x" * cfg.model.max_model_len])


def test_vllm_batched_parameters_and_seeds_without_gpu(study, monkeypatch):
    cfg, _, _ = study
    calls = []
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))

    def generate(prompts, params, **kwargs):
        calls.append((prompts, params, kwargs))
        return [SimpleNamespace(outputs=[SimpleNamespace(text="raw")]) for _ in prompts]

    engine = VLLMBackend.__new__(VLLMBackend)
    engine.cfg = cfg.model
    engine.tokenizer = Tokenizer()
    engine.engine = SimpleNamespace(generate=generate)
    messages = [[{"role": "user", "content": text}] for text in ("one", "two")]
    assert engine.generate(messages, [101, 102]) == ["raw", "raw"]
    prompts, params, options = calls[0]
    assert prompts == ["official:one", "official:two"]
    assert [p["seed"] for p in params] == [101, 102]
    assert all(p["top_k"] == -1 and p["repetition_penalty"] == 1.0 for p in params)
    assert options == {"use_tqdm": False}


def test_snapshot_pins_revision_and_respects_offline(study, monkeypatch):
    cfg, _, _ = study
    calls = []

    def download(name, **kwargs):
        calls.append((name, kwargs))
        return "local-snapshot"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    data = cfg.model.model_dump()
    data["local_files_only"] = True
    assert snapshot(Model.model_validate(data)) == "local-snapshot"
    assert calls[0][0] == cfg.model.name
    assert calls[0][1]["revision"] == cfg.model.revision
    assert calls[0][1]["local_files_only"] is True


def test_missing_cuda_has_actionable_error_and_no_fallback(study, monkeypatch):
    cfg, _, _ = study
    data = cfg.model.model_dump()
    data.update(backend="vllm", device="cuda")
    monkeypatch.setattr("resume_study.inference.hardware", lambda: {"cuda_available": False})
    monkeypatch.setattr("resume_study.inference.platform.system", lambda: "Linux")
    monkeypatch.setattr("resume_study.inference.importlib.util.find_spec", lambda _: object())
    with pytest.raises(InferenceError, match="CUDA unavailable"):
        preflight(Model.model_validate(data))  
