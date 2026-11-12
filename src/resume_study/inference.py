"""Explicit offline engines. No automatic fallback, services, or paid APIs."""

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from .config import Model, digest


class InferenceError(RuntimeError):
    pass


def prepare_environment(cfg: Model) -> None:
    if cfg.backend == "transformers":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    elif cfg.backend == "vllm":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"


def execution_environment() -> dict:
    keys = {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
    }
    for key in os.environ:
        if key.startswith(("VLLM_", "TORCH_", "PYTORCH_", "CUDA_", "CUDNN_", "NCCL_")):
            if not any(
                secret in key
                for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "URL", "ENDPOINT")
            ):
                keys.add(key)
    return {key: os.environ.get(key) for key in sorted(keys)}


def hardware() -> dict:
    report = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cuda_available": False,
        "mps_available": False,
        "cpu_count": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        info = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        report["cpu_and_memory"] = info.stdout.strip()
    elif Path("/proc/cpuinfo").is_file():
        report["cpu_model"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
    if importlib.util.find_spec("torch"):
        try:
            import torch

            report.update(
                torch=torch.__version__,
                cuda_runtime=torch.version.cuda,
                cuda_available=torch.cuda.is_available(),
                mps_available=torch.backends.mps.is_available(),
                torch_num_threads=torch.get_num_threads(),
                torch_num_interop_threads=torch.get_num_interop_threads(),
            )
            report["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ]
            if report["cuda_available"] and shutil.which("nvidia-smi"):
                driver = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                report["nvidia_driver"] = driver.stdout.strip()
        except Exception as exc:
            report["torch_error"] = str(exc)
    return report


def versions() -> dict:
    # Include kernel libraries and optional-package presence, not only direct pins.
    result = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    for package in (
        "llm-resume-study",
        "torch",
        "transformers",
        "vllm",
        "huggingface-hub",
        "Jinja2",
        "PyYAML",
        "pydantic",
        "numpy",
        "tokenizers",
        "safetensors",
        "triton",
        "xformers",
        "flash-attn",
        "flashinfer-python",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return dict(sorted(result.items()))


def preflight(cfg: Model) -> dict:
    report = hardware()
    if cfg.backend == "mock":
        return report
    required = ["torch", "transformers", "huggingface_hub"]
    if cfg.backend == "vllm":
        required.append("vllm")
        if platform.system() != "Linux" or cfg.device != "cuda":
            raise InferenceError(
                "vLLM requires Linux + NVIDIA CUDA here. Use a Linux GPU host, "
                "or explicitly select --backend transformers --device cpu."
            )
    missing = [module for module in required if importlib.util.find_spec(module) is None]
    if missing:
        raise InferenceError(
            f"Missing {', '.join(missing)}. Run uv sync --locked --extra {cfg.backend}."
        )
    if cfg.device == "cuda" and not report["cuda_available"]:
        raise InferenceError(
            "CUDA unavailable. Check the NVIDIA driver and PyTorch CUDA build; "
            "or explicitly select --backend transformers --device cpu."
        )
    if cfg.device == "mps" and not report["mps_available"]:
        raise InferenceError("MPS unavailable. Select --device cpu or use a supported Apple GPU.")
    if cfg.device == "cpu" and cfg.dtype != "float32":
        raise InferenceError("Use dtype: float32 for this CPU implementation.")
    if cfg.device == "mps" and cfg.dtype == "bfloat16":
        raise InferenceError("Use float32 or float16 for this MPS implementation.")
    return report


class Backend(Protocol):
    name: str
    metadata: dict

    def format(self, messages: list[dict]) -> str: ...

    def generate(self, messages: list[list[dict]], seeds: list[int]) -> list[str]: ...


class MockBackend:
    name = "mock"
    metadata = {"engine": "synthetic constant-output mock; NOT Qwen", "chat_template_hash": None}

    def format(self, messages: list[dict]) -> str:
        return json.dumps(messages, ensure_ascii=False, sort_keys=True)

    def generate(self, messages: list[list[dict]], seeds: list[int]) -> list[str]:
        return [
            json.dumps(
                {
                    "score": 3,
                    "recommendation": "maybe",
                    "reason": "Synthetic mock response; not model evidence.",
                }
            )
            for _ in messages
        ]


class ChatBackend:
    def format(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def validate_context(self, prompts: list[str]) -> None:
        for prompt in prompts:
            size = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            if size + self.cfg.max_new_tokens > self.cfg.max_model_len:
                raise InferenceError(
                    f"Prompt {size} tokens + output budget exceeds max_model_len="
                    f"{self.cfg.max_model_len}; shorten inputs or increase it. No truncation."
                )


def snapshot(cfg: Model) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        cfg.name,
        revision=cfg.revision,
        local_files_only=cfg.local_files_only,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )


class TransformersBackend(ChatBackend):
    name = "transformers"

    def __init__(self, cfg: Model):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.cfg = cfg
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(cfg.seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        directory = snapshot(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                directory,
                local_files_only=True,
                trust_remote_code=False,
                dtype=getattr(torch, cfg.dtype),
                attn_implementation=cfg.attention_implementation,
            )
            .to(cfg.device)
            .eval()
        )
        # Fresh config prevents Qwen's top_k/repetition defaults changing this design.
        self.generation = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature if cfg.temperature > 0 else 1.0,
            top_p=cfg.top_p if cfg.temperature > 0 else 1.0,
            top_k=0 if cfg.temperature > 0 else 50,  # Greedy has no sampling filters.
            repetition_penalty=1.0,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        self.metadata = {
            "chat_template_hash": digest(self.tokenizer.chat_template),
            "generation_config": self.generation.to_dict(),
            "effective_device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "attention": cfg.attention_implementation,
            "deterministic_algorithms": True,
            "execution": "one request at a time; per-condition seed",
            "use_model_defaults": False,
        }

    def generate(self, messages: list[list[dict]], seeds: list[int]) -> list[str]:
        import torch

        prompts = [self.format(message) for message in messages]
        self.validate_context(prompts)
        responses = []
        for prompt, seed in zip(prompts, seeds, strict=True):
            torch.manual_seed(seed)
            encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
                self.cfg.device
            )
            with torch.inference_mode():
                tokens = self.model.generate(
                    **encoded, generation_config=self.generation, use_model_defaults=False
                )
            responses.append(
                self.tokenizer.decode(
                    tokens[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
                )
            )
        return responses


class VLLMBackend(ChatBackend):
    name = "vllm"

    def __init__(self, cfg: Model):
        # Set before importing vLLM. A local snapshot guarantees revision and offline loading.
        prepare_environment(cfg)
        from vllm import LLM

        self.cfg = cfg
        self.engine = LLM(
            model=snapshot(cfg),
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype=cfg.dtype,
            seed=cfg.seed,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enforce_eager=True,
            enable_prefix_caching=False,
            generation_config="vllm",
            tensor_parallel_size=1,
        )
        self.tokenizer = self.engine.get_tokenizer()
        self.metadata = {
            "chat_template_hash": digest(self.tokenizer.chat_template),
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_V1": "1",
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "generation_config": "vllm",
            "tensor_parallel_size": 1,
            "attention_backend_request": os.environ.get("VLLM_ATTENTION_BACKEND", "auto"),
        }

    def generate(self, messages: list[list[dict]], seeds: list[int]) -> list[str]:
        from vllm import SamplingParams

        prompts = [self.format(message) for message in messages]
        self.validate_context(prompts)
        params = [
            SamplingParams(
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_new_tokens,
                seed=seed,
                top_k=-1,
                repetition_penalty=1.0,
            )
            for seed in seeds
        ]
        results = self.engine.generate(prompts, params, use_tqdm=False)
        return [output.outputs[0].text for output in results]


def load_backend(cfg: Model) -> Backend:
    prepare_environment(cfg)
    preflight(cfg)
    try:
        return {
            "mock": lambda _: MockBackend(),
            "transformers": TransformersBackend,
            "vllm": VLLMBackend,
        }[cfg.backend](cfg)
    except Exception as exc:
        raise InferenceError(
            f"Cannot load {cfg.backend}: {exc}. Verify cache/network, available memory "
            "and pinned packages. For memory errors reduce batch_size/max_model_len; "
            "select another backend explicitly in a NEW run."
        ) from exc  
