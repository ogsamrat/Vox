from typing import Dict, Optional, Any
import torch
from loguru import logger


class WhisperXRefiner:
    def __init__(
        self,
        device: str = "cuda",
        align_model: Optional[str] = None,
        interpolate_method: str = "linear",
        return_char_alignments: bool = False,
    ):
        self.device = device
        self.align_model = align_model
        self.interpolate_method = interpolate_method
        self.return_char_alignments = return_char_alignments
        self.alignment_model = None
        self.metadata = None
        try:
            import whisperx
            self.whisperx = whisperx
            self.available = True
        except ImportError:
            self.whisperx = None
            self.available = False

    def refine(self, audio_path: str, transcription: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            return transcription
        try:
            language = transcription.get("language", "en")
            if self.alignment_model is None or self.metadata is None:
                self.alignment_model, self.metadata = self.whisperx.load_align_model(
                    language_code=language,
                    device=self.device,
                )
            audio = self.whisperx.load_audio(audio_path)
            segments_for_alignment = [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                for seg in transcription["segments"]
            ]
            result = self.whisperx.align(
                segments_for_alignment,
                self.alignment_model,
                self.metadata,
                audio,
                self.device,
                interpolate_method=self.interpolate_method,
                return_char_alignments=self.return_char_alignments,
            )
            refined_transcription = transcription.copy()
            refined_transcription["segments"] = []
            refined_transcription["words"] = []
            for segment in result["segments"]:
                refined_segment = {
                    "id": len(refined_transcription["segments"]),
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "words": [],
                    "avg_logprob": -1.0,
                    "no_speech_prob": 0.0,
                    "confidence": 1.0,
                }
                if "words" in segment:
                    for word in segment["words"]:
                        word_data = {
                            "word": word["word"],
                            "start": word["start"],
                            "end": word["end"],
                            "probability": word.get("score", 1.0),
                            "confidence": word.get("score", 1.0),
                        }
                        refined_segment["words"].append(word_data)
                        refined_transcription["words"].append(word_data)
                refined_transcription["segments"].append(refined_segment)
            return refined_transcription
        except Exception as e:
            logger.error(f"WhisperX refinement failed: {e}")
            return transcription

    def cleanup(self) -> None:
        if self.alignment_model is not None:
            del self.alignment_model
            self.alignment_model = None
            self.metadata = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
