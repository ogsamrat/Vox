import numpy as np
from typing import Optional
from loguru import logger

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False


class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30, aggressiveness: int = 3, energy_threshold: float = 0.01):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.aggressiveness = aggressiveness
        self.energy_threshold = energy_threshold
        self.vad = None
        if WEBRTCVAD_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(aggressiveness)
                self.use_webrtc = True
            except Exception:
                self.use_webrtc = False
        else:
            self.use_webrtc = False
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

    def is_speech(self, audio_frame: bytes) -> bool:
        if self.use_webrtc and self.vad:
            try:
                return self.vad.is_speech(audio_frame, self.sample_rate)
            except Exception:
                pass
        return self._energy_based_vad(audio_frame)

    def _energy_based_vad(self, audio_frame: bytes) -> bool:
        audio_array = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32) / 32768.0
        energy = np.sqrt(np.mean(audio_array ** 2))
        return energy > self.energy_threshold

    def process_audio(self, audio_data: np.ndarray) -> list:
        if audio_data.dtype == np.float32:
            audio_data = (audio_data * 32767).astype(np.int16)
        speech_segments = []
        current_segment_start = None
        for i in range(0, len(audio_data) - self.frame_size, self.frame_size):
            frame = audio_data[i:i + self.frame_size].tobytes()
            is_speech = self.is_speech(frame)
            time_offset = i / self.sample_rate
            if is_speech and current_segment_start is None:
                current_segment_start = time_offset
            elif not is_speech and current_segment_start is not None:
                speech_segments.append({"start": current_segment_start, "end": time_offset})
                current_segment_start = None
        if current_segment_start is not None:
            speech_segments.append({"start": current_segment_start, "end": len(audio_data) / self.sample_rate})
        return speech_segments
