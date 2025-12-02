from typing import Dict, List, Any
from loguru import logger
import re


class SmartSpeakerSeparator:
    def __init__(self):
        self.sales_patterns = {
            "greeting": [r"\b(hello|hi|good morning|good afternoon)\b", r"\b(this is|my name is|calling from)\b"],
            "questions": [r"\?$", r"\b(are you|do you|would you|can you)\b", r"\b(what|when|where|why|how)\b"],
            "offers": [r"\b(offer|deal|discount|promotion)\b", r"\b(credit card|benefits|rewards)\b"],
            "agreement": [r"\b(okay|ok|yes|sure|alright)\b"],
        }
        self.customer_patterns = {
            "questions_about_product": [r"\b(how much|what's the|cost|price)\b"],
            "hesitation": [r"\b(i don't know|not sure|maybe)\b"],
        }

    def separate_speakers(self, transcription: Dict[str, Any], min_silence_gap: float = 1.0) -> Dict[str, Any]:
        segments = transcription.get("segments", [])
        if not segments:
            return transcription
        segments_with_gaps = self._add_silence_gaps(segments)
        segments_with_speakers = self._assign_speakers_by_pattern(segments_with_gaps, min_silence_gap)
        refined_segments = self._refine_speaker_assignment(segments_with_speakers)
        final_segments = self._remove_overlaps(refined_segments)
        result = transcription.copy()
        result["segments"] = final_segments
        if "words" in result and result["words"]:
            result["words"] = self._assign_speakers_to_words(result["words"], final_segments)
        result["speakers"] = self._generate_speaker_summary(final_segments)
        return result

    def _add_silence_gaps(self, segments: List[Dict]) -> List[Dict]:
        result = []
        for i, seg in enumerate(segments):
            seg_copy = seg.copy()
            seg_copy["silence_before"] = segments[i-1]["end"] - seg["start"] if i > 0 else 0.0
            result.append(seg_copy)
        return result

    def _assign_speakers_by_pattern(self, segments: List[Dict], min_silence_gap: float) -> List[Dict]:
        result = []
        current_speaker = "sales_person"
        for i, seg in enumerate(segments):
            text = seg["text"].lower().strip()
            silence = seg.get("silence_before", 0.0)
            if i > 0 and silence >= min_silence_gap:
                if current_speaker == "sales_person":
                    if self._matches_patterns(text, self.customer_patterns.get("questions_about_product", [])):
                        current_speaker = "customer"
                    elif self._matches_patterns(text, self.sales_patterns.get("agreement", [])):
                        current_speaker = "customer"
                else:
                    if self._matches_patterns(text, self.sales_patterns.get("questions", [])):
                        current_speaker = "sales_person"
            seg_copy = seg.copy()
            seg_copy["speaker"] = "SPEAKER_01" if current_speaker == "sales_person" else "SPEAKER_02"
            seg_copy["speaker_role"] = current_speaker
            result.append(seg_copy)
        return result

    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _refine_speaker_assignment(self, segments: List[Dict]) -> List[Dict]:
        result = []
        for i, seg in enumerate(segments):
            seg_copy = seg.copy()
            if 0 < i < len(segments) - 1:
                prev_speaker = segments[i-1]["speaker"]
                next_speaker = segments[i+1]["speaker"]
                if prev_speaker == next_speaker and prev_speaker != seg["speaker"]:
                    if seg["end"] - seg["start"] < 2.0:
                        seg_copy["speaker"] = prev_speaker
                        seg_copy["speaker_role"] = segments[i-1]["speaker_role"]
            result.append(seg_copy)
        return result

    def _remove_overlaps(self, segments: List[Dict]) -> List[Dict]:
        result = []
        for i, seg in enumerate(segments):
            if i == 0:
                result.append(seg)
                continue
            prev_seg = result[-1]
            if seg["start"] < prev_seg["end"]:
                overlap = prev_seg["end"] - seg["start"]
                if overlap < 0.5:
                    midpoint = (prev_seg["end"] + seg["start"]) / 2
                    prev_seg["end"] = midpoint
                    seg["start"] = midpoint
                else:
                    if seg.get("confidence", 0) > prev_seg.get("confidence", 0):
                        prev_seg["end"] = seg["start"]
                    else:
                        seg["start"] = prev_seg["end"]
            result.append(seg)
        return result

    def _assign_speakers_to_words(self, words: List[Dict], segments: List[Dict]) -> List[Dict]:
        result = []
        for word in words:
            word_copy = word.copy()
            for seg in segments:
                if seg["start"] <= word["start"] <= seg["end"]:
                    word_copy["speaker"] = seg["speaker"]
                    word_copy["speaker_role"] = seg.get("speaker_role")
                    break
            result.append(word_copy)
        return result

    def _generate_speaker_summary(self, segments: List[Dict]) -> Dict[str, Any]:
        s1 = [s for s in segments if s["speaker"] == "SPEAKER_01"]
        s2 = [s for s in segments if s["speaker"] == "SPEAKER_02"]
        return {
            "speaker_1": {"label": "SPEAKER_01", "role": "Sales Person", "segment_count": len(s1)},
            "speaker_2": {"label": "SPEAKER_02", "role": "Customer", "segment_count": len(s2)},
            "total_speakers": 2,
        }
