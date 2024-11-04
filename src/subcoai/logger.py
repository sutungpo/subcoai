import logging
import logging.config
import os
from logging.handlers import RotatingFileHandler

def setup_logging(
    default_level=logging.INFO,
    logs_dir="logs",
    default_filename="app.log"
):
    """
    Configure logging for the entire application
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Logging configuration dictionary
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': os.path.join(logs_dir, default_filename),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': os.path.join(logs_dir, 'error.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file', 'error_file'],
                'level': default_level,
                'propagate': True
            },
            'subcoai.asr': {  # Asr module logger
                'handlers': ['console', 'file', 'error_file'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(config)

setup_logging()