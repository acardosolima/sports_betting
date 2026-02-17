# Copyright (C) 2026 Adriano Lima
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
            level=log_level, custom_handler=None, caller=caller
        )

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
