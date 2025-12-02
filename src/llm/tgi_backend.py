from typing import Dict, Any, AsyncIterator
import httpx
from loguru import logger


class TGIBackend:
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
        endpoint: str = "http://localhost:8080",
        timeout: int = 120,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                    "top_p": kwargs.get("top_p", self.top_p),
                    "do_sample": True,
                    "return_full_text": False,
                }
            }
            response = self.client.post(f"{self.endpoint}/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            generated_text = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
            return {"text": generated_text, "tokens": len(generated_text.split()), "finish_reason": "stop"}
        except Exception as e:
            logger.error(f"TGI generation failed: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        try:
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": kwargs.get("max_tokens", self.max_tokens), "temperature": kwargs.get("temperature", self.temperature), "top_p": kwargs.get("top_p", self.top_p), "do_sample": True}, "stream": True}
            async with self.async_client.stream("POST", f"{self.endpoint}/generate_stream", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        import json
                        chunk = json.loads(line[5:].strip())
                        yield {"text": chunk.get("token", {}).get("text", ""), "tokens": 1, "is_final": chunk.get("generated_text") is not None}
        except Exception as e:
            logger.error(f"TGI streaming failed: {e}")
            raise

    def cleanup(self) -> None:
        self.client.close()
