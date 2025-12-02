import pytest
from unittest.mock import Mock, patch, MagicMock


class TestBatchPipeline:
    @patch('src.pipeline.batch.WhisperTranscriber')
    @patch('src.pipeline.batch.LLMClient')
    def test_pipeline_init(self, mock_llm, mock_transcriber):
        from src.pipeline.batch import BatchPipeline
        pipeline = BatchPipeline(config_path="config.yaml")
        assert pipeline.transcriber is not None

    def test_default_config(self):
        from src.pipeline.batch import BatchPipeline
        pipeline = BatchPipeline.__new__(BatchPipeline)
        config = pipeline._default_config()
        assert "asr" in config
        assert "llm" in config
