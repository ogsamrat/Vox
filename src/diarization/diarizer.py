from typing import Dict, List, Optional, Any
import torch
from loguru import logger


class SpeakerDiarizer:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "cuda",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        self.hf_token = hf_token
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.pipeline = None
        try:
            from pyannote.audio import Pipeline
            self.Pipeline = Pipeline
            self.available = True
        except ImportError:
            self.Pipeline = None
            self.available = False

    def _load_pipeline(self) -> None:
        if not self.available or self.pipeline is not None:
            return
        try:
            self.pipeline = self.Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            if torch.cuda.is_available() and self.device == "cuda":
                self.pipeline.to(torch.device("cuda"))
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            self.available = False

    def diarize(self, audio_path: str) -> Dict[str, Any]:
        if not self.available:
            return {"speakers": [], "segments": []}
        self._load_pipeline()
        try:
            diarization = self.pipeline(
                audio_path,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
            segments = []
            speakers = set()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
                speakers.add(speaker)
            return {"speakers": sorted(list(speakers)), "segments": segments}
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return {"speakers": [], "segments": []}

    def assign_speakers_to_words(
        self, transcription: Dict[str, Any], diarization: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not diarization["segments"]:
            return transcription
        result = transcription.copy()
        for word in result.get("words", []):
            word["speaker"] = self._find_speaker_at_time(word["start"], diarization["segments"])
        for segment in result.get("segments", []):
            segment_words = [
                w for w in result.get("words", [])
                if w["start"] >= segment["start"] and w["end"] <= segment["end"]
            ]
            if segment_words:
                speakers = [w.get("speaker") for w in segment_words if w.get("speaker")]
                segment["speaker"] = max(set(speakers), key=speakers.count) if speakers else None
            else:
                segment["speaker"] = None
        return result

    @staticmethod
    def _find_speaker_at_time(time: float, speaker_segments: List[Dict[str, Any]]) -> Optional[str]:
        for segment in speaker_segments:
            if segment["start"] <= time <= segment["end"]:
                return segment["speaker"]
        return None

    def cleanup(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
