import json
import re
from typing import Dict, Any, Optional
from loguru import logger

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "assignee": {"type": "string"}
                },
                "required": ["item"]
            }
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "context": {"type": "string"}
                },
                "required": ["decision"]
            }
        },
        "key_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["point"]
            }
        },
        "sentiment": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["summary"]
}


def repair_json(text: str) -> Optional[Dict[str, Any]]:
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        text = json_match.group(1)
    text = text.strip()
    if not text.startswith('{'):
        start = text.find('{')
        if start != -1:
            text = text[start:]
    if not text.endswith('}'):
        end = text.rfind('}')
        if end != -1:
            text = text[:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r'}\s*{', '},{', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = re.sub(r"'([^']*)':", r'"\1":', text)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("JSON repair failed")
        return None


class OutputValidator:
    def __init__(self, schema: Dict[str, Any] = None):
        self.schema = schema or ANALYSIS_SCHEMA
        try:
            import jsonschema
            self.jsonschema = jsonschema
            self.available = True
        except ImportError:
            self.jsonschema = None
            self.available = False

    def validate(self, data: Dict[str, Any]) -> bool:
        if not self.available:
            return True
        try:
            self.jsonschema.validate(data, self.schema)
            return True
        except self.jsonschema.ValidationError as e:
            logger.warning(f"Validation error: {e.message}")
            return False

    def validate_and_repair(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = repair_json(text)
        if data is None:
            return None
        if self.validate(data):
            return data
        return self._apply_defaults(data)

    def _apply_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            "summary": "",
            "action_items": [],
            "decisions": [],
            "key_points": [],
            "sentiment": "neutral",
            "topics": []
        }
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        return data
