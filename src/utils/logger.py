import logging
from typing import Optional, Type


class Logger:
    """
    A utility class for creating logging.Logger instances with optional NewRelic
    integration. This class provides static methods to configure loggers with
    appropriate formatting and handlers for console output and optional external monitoring.
    """

    @staticmethod
    def get_logger(
        log_level: int = logging.INFO,
        caller: Optional[Type] = None,
    ) -> logging.Logger:
        """
        Factory method to create a logging.Logger instance.

        Args:
            log_level: Logging level for the logger (default: logging.INFO)
            caller: Optional class type that is calling the logger, used for naming the logger

        Returns:
            logging.Logger: A configured logging.Logger instance
        """
        return Logger._create_logger(
            level=log_level, custom_handler=None, caller=caller)

    @staticmethod
    def _create_logger(
        level: int,
        custom_handler: Optional[logging.Handler] = None,
        caller: Optional[Type] = None,
    ) -> logging.Logger:
        """
        Create a logging.Logger instance for debugging purposes.

        Args:
            level: Logging level
            custom_handler: Optional custom logging handler (e.g., NewRelic handler)
            caller: Optional class type that is calling the logger, used for naming the logger

        Returns:
            logging.Logger: A configured logging.Logger instance
        """
        # Set logger name based on caller
        if caller is not None:
            if isinstance(caller, type):
                name = caller.__name__
            else:
                name = caller.__class__.__name__
        else:
            name = Logger.__name__

        logger = logging.getLogger(name)
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s level=%(levelname)-7s - %(name)s.%(funcName)s(): %(message)s"
        )

        logger.propagate = False
        logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if custom_handler:
            custom_handler.setFormatter(formatter)
            logger.addHandler(custom_handler)

        return logger