from traceloop.sdk import Traceloop
from opentelemetry import trace

# Initialize OpenLLMetry (built on OpenTelemetry)
Traceloop.init(disable_batch=True)

# Create tracer for EcoBERT-X
tracer = trace.get_tracer("EcoBERT-X-Prediction")
