"""Local self-test only: registers a minimal `vllm` module (SamplingParams) when vLLM is not installed (Windows)."""
import sys
import types

if "vllm" not in sys.modules:
    try:
        import vllm  # noqa: F401
    except ImportError:
        m = types.ModuleType("vllm")

        class SamplingParams:
            def __init__(self, temperature=1.0, max_tokens=16, logprobs=None):
                self.temperature, self.max_tokens, self.logprobs = temperature, max_tokens, logprobs

        m.SamplingParams = SamplingParams
        sys.modules["vllm"] = m
