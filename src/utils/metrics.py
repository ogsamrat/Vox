from prometheus_client import Counter, Histogram, Gauge, start_http_server
from loguru import logger


class MetricsCollector:
    def __init__(self):
        self.asr_latency = Histogram('asr_transcription_seconds', 'ASR transcription latency', buckets=[0.5, 1, 2, 5, 10, 30, 60, 120])
        self.llm_latency = Histogram('llm_inference_seconds', 'LLM inference latency', buckets=[0.5, 1, 2, 5, 10, 30, 60])
        self.pipeline_latency = Histogram('pipeline_total_seconds', 'Total pipeline latency', buckets=[1, 5, 10, 30, 60, 120, 300])
        self.asr_requests = Counter('asr_requests_total', 'Total ASR requests')
        self.llm_requests = Counter('llm_requests_total', 'Total LLM requests')
        self.asr_errors = Counter('asr_errors_total', 'ASR errors')
        self.llm_errors = Counter('llm_errors_total', 'LLM errors')
        self.audio_duration_seconds = Histogram('audio_duration_seconds', 'Audio duration', buckets=[10, 30, 60, 120, 300, 600, 1800])
        self.transcribed_words = Counter('transcribed_words_total', 'Total transcribed words')
        self.llm_tokens_generated = Counter('llm_tokens_generated_total', 'LLM tokens generated')
        self.gpu_memory_used = Gauge('gpu_memory_used_gb', 'GPU memory used in GB')
        self.active_jobs = Gauge('active_jobs', 'Number of active processing jobs')
        self._server_started = False

    def start_server(self, port: int = 8001) -> None:
        if not self._server_started:
            try:
                start_http_server(port)
                self._server_started = True
                logger.info(f"Prometheus metrics server started on port {port}")
            except Exception as e:
                logger.warning(f"Could not start metrics server: {e}")


metrics = MetricsCollector()
