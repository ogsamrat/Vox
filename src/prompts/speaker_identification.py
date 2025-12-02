SPEAKER_IDENTIFICATION_PROMPT = """You are an expert conversational analyst. Analyze this transcript and identify distinct speakers.

TRANSCRIPT WITH TIMESTAMPS:
{transcript_with_timestamps}

TASK:
1. Identify exactly 2 speakers in this conversation
2. Determine their roles (e.g., Sales Person, Customer, Interviewer, etc.)
3. Assign each transcript segment to the correct speaker
4. Ensure NO overlapping timestamps between speakers

ANALYSIS GUIDELINES:
- Look at conversation flow and turn-taking patterns
- Sales person typically: initiates, explains products, asks questions about needs
- Customer typically: responds, asks about details, expresses interest/concerns
- Consider who introduces themselves and their purpose
- Consider question/answer patterns
- Consider formal vs informal speech patterns

OUTPUT FORMAT (JSON only):
{{
  "speaker_profiles": {{
    "SPEAKER_01": {{
      "likely_role": "Sales Person",
      "characteristics": "Professional tone, product knowledge",
      "confidence": 0.95
    }},
    "SPEAKER_02": {{
      "likely_role": "Customer", 
      "characteristics": "Asks questions, responds to offers",
      "confidence": 0.95
    }}
  }},
  "segments": [
    {{
      "start": 0.00,
      "end": 5.50,
      "speaker": "SPEAKER_01",
      "text": "exact text from transcript",
      "confidence": 0.9
    }}
  ],
  "conversation_summary": "Brief description of the conversation"
}}

RULES:
- Return ONLY valid JSON
- Each segment must have exactly one speaker
- Timestamps must not overlap
- Maintain original text exactly
- Assign confidence scores (0.0-1.0)"""
