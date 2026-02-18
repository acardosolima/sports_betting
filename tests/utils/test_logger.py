import logging
import unittest
from io import StringIO

from ssa.utils.logger import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.log_output = StringIO()
        self.handler = logging.StreamHandler(self.log_output)
        self.handler.setLevel(logging.DEBUG)

    def test_default_logger(self):
        """Test creating a logger with default parameters."""
        logger = Logger.get_logger()
        self.assertEqual(logger.name, "Logger")
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

    def test_custom_log_level(self):
        """Test creating a logger with a custom log level."""
        logger = Logger.get_logger(log_level=logging.CRITICAL)
        self.assertEqual(logger.level, logging.CRITICAL)

    def test_caller_naming(self):
        """Test logger naming based on caller class."""

        class TestClass:
            pass

        logger = Logger.get_logger(caller=TestClass)
        self.assertEqual(logger.name, "TestClass")

    def test_custom_handler(self):
        """Test logger with custom handler."""
        logger = Logger._create_logger(level=logging.INFO, custom_handler=self.handler)
        self.assertEqual(len(logger.handlers), 2)
        self.assertIn(self.handler, logger.handlers)

    def test_logging_output(self):
        """Test actual logging output format."""
        # Create a logger with our test handler
        logger = Logger._create_logger(level=logging.INFO, custom_handler=self.handler)
        test_message = "Test log message"
        logger.info(test_message)

        # Get the output
        output = self.log_output.getvalue()
        self.assertIn(test_message, output)
        self.assertIn("level=INFO", output)
        self.assertIn("Logger", output)

    def test_logger_propagation(self):
        """Test that logger propagation is disabled."""
        logger = Logger.get_logger()
        self.assertFalse(logger.propagate)

    def test_multiple_handlers(self):
        """Test that multiple handlers can be added."""
        logger = Logger._create_logger(level=logging.INFO, custom_handler=self.handler)
        second_handler = logging.StreamHandler(StringIO())
        logger.addHandler(second_handler)
        self.assertEqual(len(logger.handlers), 3)
