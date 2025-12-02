import pytest
from src.validation.schema import repair_json, OutputValidator


class TestRepairJson:
    def test_clean_json(self):
        result = repair_json('{"summary": "test"}')
        assert result == {"summary": "test"}

    def test_json_in_markdown(self):
        result = repair_json('```json\n{"summary": "test"}\n```')
        assert result == {"summary": "test"}

    def test_trailing_comma(self):
        result = repair_json('{"summary": "test",}')
        assert result == {"summary": "test"}

    def test_invalid_json(self):
        result = repair_json('not json at all')
        assert result is None


class TestOutputValidator:
    def test_valid_output(self):
        validator = OutputValidator()
        data = {"summary": "Test summary", "action_items": [], "decisions": []}
        assert validator.validate(data) == True

    def test_apply_defaults(self):
        validator = OutputValidator()
        data = {"summary": "Test"}
        result = validator._apply_defaults(data)
        assert "action_items" in result
        assert "decisions" in result
