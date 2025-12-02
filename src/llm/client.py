from typing import Dict, Any, AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from src.llm.vllm_backend import VLLMBackend
from src.llm.tgi_backend import TGIBackend
from src.llm.groq_backend import GroqBackend
from src.utils.metrics import metrics


class LLMClient:
    def __init__(
        self,
        backend: str = "vllm",
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_retries: int = 3,
        retry_delay: int = 2,
        timeout: int = 120,
        **backend_kwargs
    ):
        self.backend_type = backend.lower()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        if self.backend_type == "vllm":
            self.backend = VLLMBackend(model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, **backend_kwargs)
        elif self.backend_type == "tgi":
            self.backend = TGIBackend(model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, timeout=timeout, **backend_kwargs)
        elif self.backend_type == "groq":
            api_key = backend_kwargs.pop('api_key', None) or backend_kwargs.pop('groq_api_key', None)
            if not api_key:
                import os
                api_key = os.environ.get('GROQ_API_KEY')
            self.backend = GroqBackend(model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, api_key=api_key, timeout=timeout, **backend_kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        logger.info(f"LLM client initialized: {backend}, model: {model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        with metrics.llm_latency.time():
            try:
                result = self.backend.generate(prompt, **kwargs)
                metrics.llm_requests.inc()
                metrics.llm_tokens_generated.inc(result.get("tokens", 0))
                return result
            except Exception as e:
                metrics.llm_errors.inc()
                logger.error(f"LLM generation failed: {e}")
                raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        token_count = 0
        try:
            async for chunk in self.backend.generate_stream(prompt, **kwargs):
                token_count += chunk.get("tokens", 0)
                yield chunk
            metrics.llm_requests.inc()
            metrics.llm_tokens_generated.inc(token_count)
        except Exception as e:
            metrics.llm_errors.inc()
            logger.error(f"Streaming failed: {e}")
            raise

    def cleanup(self) -> None:
        if hasattr(self.backend, 'cleanup'):
            self.backend.cleanup()
