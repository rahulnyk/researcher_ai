import logging
from yachalk import chalk
import os
import logging


class AgentLogger:
    def __init__(self, name = 'Agent Logger'):
        "Set the log level (optional, can be DEBUG, INFO, WARNING, ERROR, CRITICAL)"
        self.name = name
        self.LOGLEVEL = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
        self.time_format = "%Y-%m-%d %H:%M:%S"
        self.formatter = logging.Formatter(
            fmt=chalk.gray("▶︎ %(name)s - %(asctime)s - %(levelname)s \n%(message)s\n"),
            datefmt=self.time_format,
        )

    def getLogger(self):
        logging.basicConfig(level=self.LOGLEVEL)
        # Create a logger and set the custom formatter
        logger = logging.getLogger(self.name)
        handler = logging.StreamHandler()
        handler.setFormatter(self.formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger