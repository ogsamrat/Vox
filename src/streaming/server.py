import asyncio
import json
from typing import Dict, Any, Optional
import yaml
from loguru import logger

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from src.asr import WhisperTranscriber
from src.streaming.vad import VoiceActivityDetector
from src.llm import LLMClient
from src.prompts import PromptTemplates


class StreamingServer:
    def __init__(self, config_path: str = "config.yaml", host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.config = self._load_config(config_path)
        self.transcriber: Optional[WhisperTranscriber] = None
        self.vad: Optional[VoiceActivityDetector] = None
        self.llm_client: Optional[LLMClient] = None
        self.active_connections: Dict[str, Any] = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    async def start(self) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets not available")
        self._initialize_components()
        async with websockets.serve(self._handle_connection, self.host, self.port):
            logger.info(f"Streaming server started on ws://{self.host}:{self.port}")
            await asyncio.Future()

    def _initialize_components(self) -> None:
        asr_config = self.config.get("asr", {})
        self.transcriber = WhisperTranscriber(
            model_size=asr_config.get("model", "small"),
            device=asr_config.get("device", "cpu"),
            compute_type=asr_config.get("compute_type", "int8"),
            language=asr_config.get("language", "en")
        )
        streaming_config = self.config.get("streaming", {})
        self.vad = VoiceActivityDetector(
            sample_rate=streaming_config.get("sample_rate", 16000),
            aggressiveness=streaming_config.get("vad_aggressiveness", 3)
        )
        llm_config = self.config.get("llm", {})
        api_key = llm_config.get("api_key")
        backend_kwargs = {}
        if api_key:
            backend_kwargs["api_key"] = api_key
        self.llm_client = LLMClient(
            backend=llm_config.get("backend", "groq"),
            model=llm_config.get("model", "llama-3.3-70b-versatile"),
            **backend_kwargs
        )

    async def _handle_connection(self, websocket, path) -> None:
        connection_id = id(websocket)
        self.active_connections[connection_id] = {"websocket": websocket, "audio_buffer": b"", "transcript_context": ""}
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._process_audio_chunk(connection_id, message)
                else:
                    await self._handle_control_message(connection_id, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            del self.active_connections[connection_id]

    async def _process_audio_chunk(self, connection_id: str, audio_data: bytes) -> None:
        conn = self.active_connections[connection_id]
        conn["audio_buffer"] += audio_data
        streaming_config = self.config.get("streaming", {})
        chunk_duration = streaming_config.get("chunk_duration_seconds", 5)
        sample_rate = streaming_config.get("sample_rate", 16000)
        chunk_size = chunk_duration * sample_rate * 2
        if len(conn["audio_buffer"]) >= chunk_size:
            chunk = conn["audio_buffer"][:chunk_size]
            conn["audio_buffer"] = conn["audio_buffer"][chunk_size:]
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import wave
                with wave.open(f.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(chunk)
                temp_path = f.name
            transcription = self.transcriber.transcribe(temp_path)
            chunk_text = transcription.get("text", "")
            conn["transcript_context"] += " " + chunk_text
            prompt = PromptTemplates.build_streaming_prompt(chunk_text, conn["transcript_context"][-2000:])
            response = self.llm_client.generate(prompt, max_tokens=512)
            await conn["websocket"].send(json.dumps({"type": "transcription", "text": chunk_text, "analysis": response.get("text", "")}))

    async def _handle_control_message(self, connection_id: str, message: str) -> None:
        try:
            data = json.loads(message)
            if data.get("type") == "end_stream":
                conn = self.active_connections[connection_id]
                await conn["websocket"].send(json.dumps({"type": "stream_ended", "final_transcript": conn["transcript_context"]}))
        except json.JSONDecodeError:
            pass

    def cleanup(self) -> None:
        if self.transcriber:
            self.transcriber.cleanup()
        if self.llm_client:
            self.llm_client.cleanup()
