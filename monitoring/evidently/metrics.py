from prometheus_client import Counter, Histogram, Gauge

MODEL_CHECK_COUNTER = Counter('model_checks_total', 'Number of model checks performed')
MODEL_FAILURES = Counter('model_check_failures', 'Number of failed model checks')
CHECK_DURATION = Histogram('model_check_duration_seconds', 'Time spent performing model check')