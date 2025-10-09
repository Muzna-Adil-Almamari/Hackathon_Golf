import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger


# If directory doesn't exist then create
os.makedirs('./logs', exist_ok=True)

FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'

# Create formatters
standard_formatter = logging.Formatter(FORMAT)
# Fix: Use correct field names or rename them properly
json_formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(levelname)s %(message)s %(name)s',
    rename_fields={'asctime': 'timestamp', 'levelname': 'level'}
)

# Create handlers
file_handler = RotatingFileHandler(
    './logs/myapp.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=7
)
file_handler.setFormatter(standard_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(standard_formatter)

# Create JSON file handler
json_handler = RotatingFileHandler(
    './logs/myapp.json',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=7
)
json_handler.setFormatter(json_formatter)

# Create error-specific handlers
error_file_handler = RotatingFileHandler(
    './logs/myapp.error.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=7
)
error_file_handler.setFormatter(standard_formatter)
error_file_handler.setLevel(logging.ERROR)  # Only capture ERROR and above

# Create JSON error file handler
error_json_handler = RotatingFileHandler(
    './logs/myapp.error.json',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=7
)
error_json_handler.setFormatter(json_formatter)
error_json_handler.setLevel(logging.ERROR)  # Only capture ERROR and above

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.addHandler(json_handler)
root_logger.addHandler(error_file_handler)
root_logger.addHandler(error_json_handler)

def get_logger(name=None):
    """Get a configured logger instance"""
    logger = logging.getLogger(name or __name__)
    return logger