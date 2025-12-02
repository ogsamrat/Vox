import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from loguru import logger


def convert_audio(
    input_path: str,
    output_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1,
    output_path: Optional[str] = None
) -> str:
    input_path = Path(input_path)
    if output_path:
        output_file = Path(output_path)
    else:
        output_file = Path(tempfile.mktemp(suffix=f".{output_format}"))
    cmd = ["ffmpeg", "-i", str(input_path), "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le" if output_format == "wav" else "libmp3lame", "-y", str(output_file)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            raise RuntimeError(f"Audio conversion failed: {result.stderr}")
        return str(output_file)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio conversion timeout")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found")


def get_audio_duration(audio_path: str) -> float:
    cmd = ["ffprobe", "-i", str(audio_path), "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0


def load_audio(audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype='float32')
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise
