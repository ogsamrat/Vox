import gc
from typing import Dict, List, Optional, Any
import torch
from faster_whisper import WhisperModel
from loguru import logger

from src.utils.metrics import metrics


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict[str, Any]] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {}
        self.model: Optional[WhisperModel] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            download_root = "D:/task/models/whisper"
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=download_root,
            )
            logger.success(f"Whisper model loaded: {self.model_size}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            if self.device == "cuda":
                logger.warning("Fallback to CPU...")
                self.device = "cpu"
                self.compute_type = "int8"
                try:
                    download_root = "D:/task/models/whisper"
                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root=download_root,
                    )
                    logger.success("Whisper model loaded on CPU")
                except Exception as e2:
                    logger.critical(f"Failed on CPU: {e2}")
                    raise
            else:
                raise

    def transcribe(self, audio_path: str, task: str = "transcribe", **kwargs) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        logger.info(f"Transcribing: {audio_path}")
        with metrics.asr_latency.time():
            try:
                segments, info = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    task=task,
                    beam_size=self.beam_size,
                    vad_filter=self.vad_filter,
                    vad_parameters=self.vad_parameters,
                    word_timestamps=True,
                    **kwargs
                )
                segments_list = list(segments)
                result = {
                    "text": "",
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "segments": [],
                    "words": [],
                }
                full_text_parts = []
                for segment in segments_list:
                    segment_data = {
                        "id": segment.id,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "avg_logprob": segment.avg_logprob,
                        "no_speech_prob": segment.no_speech_prob,
                        "confidence": self._logprob_to_confidence(segment.avg_logprob),
                        "words": [],
                    }
                    if segment.words:
                        for word in segment.words:
                            word_data = {
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "probability": word.probability,
                                "confidence": word.probability,
                            }
                            segment_data["words"].append(word_data)
                            result["words"].append(word_data)
                    result["segments"].append(segment_data)
                    full_text_parts.append(segment.text)
                result["text"] = " ".join(full_text_parts).strip()
                metrics.asr_requests.inc()
                metrics.audio_duration_seconds.observe(info.duration)
                metrics.transcribed_words.inc(len(result["words"]))
                logger.success(f"Transcription complete: {len(result['words'])} words")
                return result
            except Exception as e:
                metrics.asr_errors.inc()
                logger.error(f"Transcription failed: {e}")
                raise

    @staticmethod
    def _logprob_to_confidence(avg_logprob: float) -> float:
        import math
        confidence = math.exp(max(avg_logprob, -3.0))
        return min(max(confidence, 0.0), 1.0)

    def cleanup(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            logger.info("Whisper model cleaned up")
