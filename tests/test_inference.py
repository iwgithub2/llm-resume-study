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


def test_explicit_greedy_settings_override_recent_model_defaults(study):
    # Transformers >=4.50 can replace even explicitly set values equal to global
    # defaults. A recent Qwen config must not turn our temperature-zero run stochastic.
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from resume_study.inference import TransformersBackend

    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    model.generation_config = transformers.GenerationConfig(
        do_sample=True, temperature=0.7, top_k=20, top_p=0.8, transformers_version="4.57.1"
    )
    calls = []
    original = model._prepare_generation_config

    def capture(*args, **kwargs):
        generation, remaining = original(*args, **kwargs)
        calls.append(generation)
        return generation, remaining

    model._prepare_generation_config = capture

    class Encoding(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            return self

    class TinyTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            return Encoding(
                input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2, dtype=torch.long)
            )

        def decode(self, tokens, **kwargs):
            return "raw"

    engine = TransformersBackend.__new__(TransformersBackend)
    engine.cfg = study[0].model
    engine.model = model
    engine.tokenizer = TinyTokenizer()
    engine.generation = transformers.GenerationConfig(
        max_new_tokens=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=31,
    )
    assert engine.generate([[{"role": "user", "content": "test"}]], [101]) == ["raw"]
    assert calls and calls[0].do_sample is False
    assert calls[0].top_p == 1.0 and calls[0].temperature == 1.0  
