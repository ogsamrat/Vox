from typing import Dict, Any, AsyncIterator
from loguru import logger


class GroqBackend:
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
        api_key: str = None,
        timeout: int = 120,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key
        self.timeout = timeout
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self.available = True
        except ImportError:
            self.client = None
            self.available = False
            raise ImportError("Please install groq: pip install groq")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("Groq client not available")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
            )
            return {
                "text": response.choices[0].message.content,
                "tokens": response.usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
            }
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        if not self.available:
            raise RuntimeError("Groq client not available")
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {"text": chunk.choices[0].delta.content, "tokens": 1, "is_final": chunk.choices[0].finish_reason is not None}
        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            raise

    def cleanup(self) -> None:
        pass
