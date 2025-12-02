import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from loguru import logger

from src.asr import WhisperTranscriber, WhisperXRefiner
from src.llm import LLMClient
from src.prompts import PromptTemplates, SPEAKER_IDENTIFICATION_PROMPT
from src.validation import OutputValidator, repair_json
from src.utils.audio import convert_audio, get_audio_duration
from src.utils.metrics import metrics


class BatchPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.transcriber: Optional[WhisperTranscriber] = None
        self.refiner: Optional[WhisperXRefiner] = None
        self.llm_client: Optional[LLMClient] = None
        self.validator = OutputValidator()
        self._initialize_components()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        local_config = Path("config.local.yaml")
        if local_config.exists():
            with open(local_config) as f:
                return yaml.safe_load(f)
        config_file = Path(config_path)
        if not config_file.exists():
            return self._default_config()
        with open(config_file) as f:
            return yaml.safe_load(f)

    def _default_config(self) -> Dict[str, Any]:
        return {
            "asr": {"model": "small", "device": "cpu", "compute_type": "int8", "language": "en"},
            "llm": {"backend": "groq", "model": "llama-3.3-70b-versatile", "max_tokens": 4096},
            "pipeline": {"enable_timestamps": True, "enable_diarization": True}
        }

    def _initialize_components(self) -> None:
        asr_config = self.config.get("asr", {})
        self.transcriber = WhisperTranscriber(
            model_size=asr_config.get("model", "small"),
            device=asr_config.get("device", "cpu"),
            compute_type=asr_config.get("compute_type", "int8"),
            language=asr_config.get("language", "en"),
            beam_size=asr_config.get("beam_size", 5),
            vad_filter=asr_config.get("vad_filter", True)
        )
        llm_config = self.config.get("llm", {})
        api_key = llm_config.get("api_key")
        backend_kwargs = {k: v for k, v in llm_config.items() if k not in ["backend", "model", "max_tokens", "temperature", "top_p"]}
        if api_key:
            backend_kwargs["api_key"] = api_key
        self.llm_client = LLMClient(
            backend=llm_config.get("backend", "groq"),
            model=llm_config.get("model", "llama-3.3-70b-versatile"),
            max_tokens=llm_config.get("max_tokens", 4096),
            temperature=llm_config.get("temperature", 0.1),
            top_p=llm_config.get("top_p", 0.9),
            **backend_kwargs
        )

    def process(self, audio_path: str, output_path: Optional[str] = None, progress_callback=None) -> Dict[str, Any]:
        start_time = time.time()
        result = {"metadata": {"source_file": str(audio_path), "processing_time_seconds": 0}, "transcript": {}, "analysis": {}}
        try:
            if progress_callback:
                progress_callback(10, "Converting audio...")
            processed_audio = self._prepare_audio(audio_path)
            result["metadata"]["duration_seconds"] = get_audio_duration(processed_audio)
            if progress_callback:
                progress_callback(25, "Transcribing audio...")
            transcription = self.transcriber.transcribe(processed_audio)
            result["transcript"] = transcription
            result["metadata"]["language"] = transcription.get("language", "unknown")
            if progress_callback:
                progress_callback(50, "Identifying speakers...")
            if self.config.get("diarization", {}).get("enabled", True):
                speaker_result = self._identify_speakers_with_llm(transcription)
                if speaker_result:
                    result["transcript"]["segments"] = speaker_result.get("segments", transcription.get("segments", []))
                    result["speaker_profiles"] = speaker_result.get("speaker_profiles", {})
            if progress_callback:
                progress_callback(75, "Analyzing content...")
            analysis = self._analyze_transcript(transcription)
            result.update(analysis)
            result["metadata"]["processing_time_seconds"] = round(time.time() - start_time, 2)
            if output_path:
                self._save_result(result, output_path)
            if progress_callback:
                progress_callback(100, "Complete")
            return result
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result["error"] = str(e)
            return result

    def _prepare_audio(self, audio_path: str) -> str:
        audio_file = Path(audio_path)
        if audio_file.suffix.lower() in ['.wav'] and audio_file.stat().st_size < 100 * 1024 * 1024:
            return str(audio_path)
        return convert_audio(audio_path)

    def _identify_speakers_with_llm(self, transcription: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        segments = transcription.get("segments", [])
        if not segments:
            return None
        transcript_lines = []
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            if text:
                transcript_lines.append(f"[{start:.2f}s - {end:.2f}s]: {text}")
        transcript_with_timestamps = "\n".join(transcript_lines)
        prompt = SPEAKER_IDENTIFICATION_PROMPT.format(transcript_with_timestamps=transcript_with_timestamps)
        try:
            response = self.llm_client.generate(prompt, max_tokens=8000)
            response_text = response.get("text", "")
            parsed = repair_json(response_text)
            if parsed and "segments" in parsed:
                return parsed
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
        return None

    def _analyze_transcript(self, transcription: Dict[str, Any]) -> Dict[str, Any]:
        full_text = transcription.get("text", "")
        segments = transcription.get("segments", [])
        timestamps = [{"start": s.get("start"), "end": s.get("end"), "text": s.get("text")} for s in segments]
        prompt = PromptTemplates.build_analysis_prompt(transcript=full_text, timestamps=timestamps)
        try:
            response = self.llm_client.generate(prompt)
            response_text = response.get("text", "")
            analysis = self.validator.validate_and_repair(response_text)
            if analysis:
                return analysis
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
        return {"summary": full_text[:500] if full_text else "", "action_items": [], "decisions": [], "key_points": []}

    def _save_result(self, result: Dict[str, Any], output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def cleanup(self) -> None:
        if self.transcriber:
            self.transcriber.cleanup()
        if self.refiner:
            self.refiner.cleanup()
        if self.llm_client:
            self.llm_client.cleanup()
