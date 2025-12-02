SPEAKER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker_profiles": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "likely_role": {"type": "string"},
                    "characteristics": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["likely_role"]
            }
        },
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["start", "end", "speaker", "text"]
            }
        },
        "conversation_summary": {"type": "string"}
    },
    "required": ["speaker_profiles", "segments"]
}
