import logging

# Log to stderr by default
logging.getLogger(__name__).addHandler(logging.NullHandler())