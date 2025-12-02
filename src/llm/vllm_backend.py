import gc
from typing import Dict, Any, AsyncIterator, Optional
import torch
from loguru import logger


class VLLMBackend:
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.llm = None
        self.sampling_params = None
        try:
            from vllm import LLM, SamplingParams
            self.LLM = LLM
            self.SamplingParams = SamplingParams
            self.available = True
        except ImportError:
            self.LLM = None
            self.SamplingParams = None
            self.available = False
            return
        self._load_model(tensor_parallel_size=tensor_parallel_size, quantization=quantization, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, trust_remote_code=trust_remote_code, **kwargs)

    def _load_model(self, **kwargs) -> None:
        if not self.available:
            raise RuntimeError("vLLM not available")
        try:
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            self.llm = self.LLM(model=self.model, download_dir="models/llm", **filtered_kwargs)
            self.sampling_params = self.SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("vLLM model not loaded")
        try:
            sampling_params = self.sampling_params
            if kwargs:
                sampling_params = self.SamplingParams(max_tokens=kwargs.get("max_tokens", self.max_tokens), temperature=kwargs.get("temperature", self.temperature), top_p=kwargs.get("top_p", self.top_p))
            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0]
            return {"text": output.outputs[0].text, "tokens": len(output.outputs[0].token_ids), "finish_reason": output.outputs[0].finish_reason}
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        result = self.generate(prompt, **kwargs)
        yield {"text": result["text"], "tokens": result["tokens"], "is_final": True}

    def cleanup(self) -> None:
        if self.llm is not None:
            del self.llm
            self.llm = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
