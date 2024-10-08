import logging


DEFAULT_LOG_FORMAT = "[%(asctime)s][%(filename)s]- %(levelname)s: %(message)s"
logger = logging.getLogger("graphembedding")
logging.basicConfig(
    level=logging.INFO,
    format=DEFAULT_LOG_FORMAT,
    datefmt='%H:%M:%S',
)
