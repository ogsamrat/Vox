import pytest
from unittest.mock import Mock, patch


class TestWhisperTranscriber:
    @patch('src.asr.transcriber.WhisperModel')
    def test_transcriber_init(self, mock_model):
        from src.asr.transcriber import WhisperTranscriber
        transcriber = WhisperTranscriber(model_size="small", device="cpu")
        assert transcriber.model_size == "small"
        assert transcriber.device == "cpu"

    def test_logprob_to_confidence(self):
        from src.asr.transcriber import WhisperTranscriber
        assert WhisperTranscriber._logprob_to_confidence(-0.5) == pytest.approx(0.606, rel=0.01)
        assert WhisperTranscriber._logprob_to_confidence(-3.0) == pytest.approx(0.05, rel=0.01)
        assert WhisperTranscriber._logprob_to_confidence(0.0) == 1.0
