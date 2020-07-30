import logging

__import__("pkg_resources").declare_namespace(__name__)
# Log to stderr by default
logging.getLogger(__name__).addHandler(logging.NullHandler())