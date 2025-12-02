from typing import Dict, Any, List


class PromptTemplates:
    @staticmethod
    def build_analysis_prompt(transcript: str, timestamps: List[Dict[str, Any]] = None, speakers: Dict[str, Any] = None, confidence_scores: List[float] = None, context: str = None) -> str:
        prompt_parts = ["Analyze the following audio transcript and provide a structured analysis.", ""]
        if context:
            prompt_parts.extend([f"Context: {context}", ""])
        prompt_parts.extend(["TRANSCRIPT:", "```", transcript if transcript else "[No transcript provided]", "```", ""])
        if timestamps:
            prompt_parts.extend(["TIMESTAMPS:", "```"])
            for ts in timestamps[:20]:
                prompt_parts.append(f"[{ts.get('start', 0):.2f}s - {ts.get('end', 0):.2f}s]: {ts.get('text', '')[:100]}")
            if len(timestamps) > 20:
                prompt_parts.append(f"... and {len(timestamps) - 20} more segments")
            prompt_parts.extend(["```", ""])
        if speakers:
            prompt_parts.extend(["SPEAKERS:", "```"])
            for speaker_id, info in speakers.items():
                prompt_parts.append(f"{speaker_id}: {info}")
            prompt_parts.extend(["```", ""])
        prompt_parts.extend([
            "Provide your analysis in the following JSON format:",
            "```json",
            "{",
            '  "summary": "A comprehensive summary of the conversation (2-4 sentences)",',
            '  "action_items": [',
            '    {"item": "Action item description", "confidence": 0.95, "assignee": "Person responsible"}',
            "  ],",
            '  "decisions": [',
            '    {"decision": "Decision made", "confidence": 0.90, "context": "Brief context"}',
            "  ],",
            '  "key_points": [',
            '    {"point": "Key point or insight", "confidence": 0.85}',
            "  ],",
            '  "sentiment": "overall|positive|negative|neutral|mixed",',
            '  "topics": ["topic1", "topic2"]',
            "}",
            "```",
            "",
            "Rules:",
            "- Extract only information explicitly stated in the transcript",
            "- Mark uncertain information with confidence < 0.7",
            "- Use [UNSURE] prefix for any assumptions",
            "- Keep the summary concise but comprehensive",
            "- Return ONLY valid JSON, no additional text"
        ])
        return "\n".join(prompt_parts)

    @staticmethod
    def build_streaming_prompt(transcript_chunk: str, previous_context: str = None, is_final: bool = False) -> str:
        prompt_parts = []
        if previous_context:
            prompt_parts.extend(["Previous context:", previous_context, ""])
        prompt_parts.extend(["New transcript chunk:", transcript_chunk, ""])
        if is_final:
            prompt_parts.extend(["This is the FINAL chunk. Provide complete analysis.", ""])
        prompt_parts.extend([
            "Provide incremental analysis as JSON:",
            '{"new_insights": [], "updated_summary": "", "confidence": 0.0}'
        ])
        return "\n".join(prompt_parts)
