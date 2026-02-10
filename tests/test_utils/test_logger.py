"""
Comprehensive tests for DeepBridge logging utilities.

This test suite validates:
1. DeepBridgeLogger initialization
2. Logging level management
3. Verbosity control
4. All logging methods (debug, info, warning, error, critical, exception)
5. get_logger singleton pattern

Coverage Target: ~100%
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from io import StringIO

from deepbridge.utils.logger import DeepBridgeLogger, get_logger, _logger_instance


# ==================== Fixtures ====================


@pytest.fixture(autouse=True)
def reset_logger_singleton():
    """Reset the global logger instance before each test"""
    import deepbridge.utils.logger as logger_module
    logger_module._logger_instance = None
    yield
    logger_module._logger_instance = None


@pytest.fixture
def logger():
    """Create a fresh DeepBridgeLogger instance"""
    return DeepBridgeLogger(name='test_logger', level=logging.INFO)


@pytest.fixture
def captured_logs():
    """Capture log output"""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    return log_stream, handler


# ==================== Initialization Tests ====================


class TestDeepBridgeLoggerInitialization:
    """Tests for DeepBridgeLogger initialization"""

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        logger = DeepBridgeLogger()

        assert logger.logger.name == 'deepbridge'
        assert logger.logger.level == logging.INFO
        assert logger.verbose is True
        assert len(logger.logger.handlers) > 0

    def test_init_with_custom_name(self):
        """Test initialization with custom name"""
        logger = DeepBridgeLogger(name='custom_logger')

        assert logger.logger.name == 'custom_logger'

    def test_init_with_custom_level(self):
        """Test initialization with custom level"""
        logger = DeepBridgeLogger(level=logging.DEBUG)

        assert logger.logger.level == logging.DEBUG
        assert logger.verbose is True  # DEBUG <= INFO

    def test_init_with_warning_level(self):
        """Test initialization with WARNING level"""
        logger = DeepBridgeLogger(level=logging.WARNING)

        assert logger.logger.level == logging.WARNING
        assert logger.verbose is False  # WARNING > INFO

    def test_handler_not_duplicated(self):
        """Test that handlers are not duplicated on re-initialization"""
        # Create logger
        logger1 = DeepBridgeLogger(name='shared_logger')
        initial_handlers = len(logger1.logger.handlers)

        # Create another logger with same name (gets same logging.Logger)
        logger2 = DeepBridgeLogger(name='shared_logger')

        # Should not add duplicate handlers
        assert len(logger2.logger.handlers) == initial_handlers


# ==================== Level Property Tests ====================


class TestLevelProperty:
    """Tests for level property"""

    def test_level_getter(self, logger):
        """Test getting current level"""
        assert logger.level == logging.INFO

    def test_level_reflects_changes(self, logger):
        """Test that level property reflects changes"""
        logger.set_level(logging.DEBUG)
        assert logger.level == logging.DEBUG

        logger.set_level(logging.ERROR)
        assert logger.level == logging.ERROR


# ==================== set_level Tests ====================


class TestSetLevel:
    """Tests for set_level method"""

    def test_set_level_debug(self, logger):
        """Test setting level to DEBUG"""
        logger.set_level(logging.DEBUG)

        assert logger.logger.level == logging.DEBUG
        assert logger.verbose is True

    def test_set_level_info(self, logger):
        """Test setting level to INFO"""
        logger.set_level(logging.INFO)

        assert logger.logger.level == logging.INFO
        assert logger.verbose is True

    def test_set_level_warning(self, logger):
        """Test setting level to WARNING"""
        logger.set_level(logging.WARNING)

        assert logger.logger.level == logging.WARNING
        assert logger.verbose is False

    def test_set_level_error(self, logger):
        """Test setting level to ERROR"""
        logger.set_level(logging.ERROR)

        assert logger.logger.level == logging.ERROR
        assert logger.verbose is False

    def test_set_level_critical(self, logger):
        """Test setting level to CRITICAL"""
        logger.set_level(logging.CRITICAL)

        assert logger.logger.level == logging.CRITICAL
        assert logger.verbose is False


# ==================== set_verbose Tests ====================


class TestSetVerbose:
    """Tests for set_verbose method"""

    def test_set_verbose_true(self, logger):
        """Test setting verbose to True"""
        logger.set_level(logging.WARNING)  # Start with non-verbose
        logger.set_verbose(True)

        assert logger.verbose is True
        assert logger.logger.level == logging.INFO

    def test_set_verbose_false(self, logger):
        """Test setting verbose to False"""
        logger.set_level(logging.DEBUG)  # Start with verbose
        logger.set_verbose(False)

        assert logger.verbose is False
        assert logger.logger.level == logging.WARNING

    def test_set_verbose_idempotent(self, logger):
        """Test that setting verbose twice has no side effects"""
        logger.set_verbose(True)
        level1 = logger.logger.level
        logger.set_verbose(True)
        level2 = logger.logger.level

        assert level1 == level2 == logging.INFO


# ==================== Logging Methods Tests ====================


class TestLoggingMethods:
    """Tests for logging methods (debug, info, warning, error, critical)"""

    def test_debug(self, logger):
        """Test debug logging"""
        with patch.object(logger.logger, 'debug') as mock_debug:
            logger.debug('Test debug message')
            mock_debug.assert_called_once_with('Test debug message')

    def test_info(self, logger):
        """Test info logging"""
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info('Test info message')
            mock_info.assert_called_once_with('Test info message')

    def test_warning(self, logger):
        """Test warning logging"""
        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.warning('Test warning message')
            mock_warning.assert_called_once_with('Test warning message')

    def test_error(self, logger):
        """Test error logging"""
        with patch.object(logger.logger, 'error') as mock_error:
            logger.error('Test error message')
            mock_error.assert_called_once_with('Test error message')

    def test_critical(self, logger):
        """Test critical logging"""
        with patch.object(logger.logger, 'critical') as mock_critical:
            logger.critical('Test critical message')
            mock_critical.assert_called_once_with('Test critical message')

    def test_exception(self, logger):
        """Test exception logging"""
        with patch.object(logger.logger, 'exception') as mock_exception:
            logger.exception('Test exception message')
            mock_exception.assert_called_once_with('Test exception message')


# ==================== get_logger Tests ====================


class TestGetLogger:
    """Tests for get_logger function (singleton pattern)"""

    def test_get_logger_creates_instance(self):
        """Test that get_logger creates a new instance if none exists"""
        logger = get_logger()

        assert logger is not None
        assert isinstance(logger, DeepBridgeLogger)

    def test_get_logger_returns_same_instance(self):
        """Test singleton pattern - same instance returned"""
        logger1 = get_logger()
        logger2 = get_logger()

        assert logger1 is logger2

    def test_get_logger_with_custom_name(self):
        """Test get_logger with custom name on first call"""
        logger = get_logger(name='custom_logger')

        assert logger.logger.name == 'custom_logger'

    def test_get_logger_with_custom_level(self):
        """Test get_logger with custom level on first call"""
        logger = get_logger(level=logging.DEBUG)

        assert logger.logger.level == logging.DEBUG

    def test_get_logger_updates_level_on_existing(self):
        """Test that get_logger updates level on existing instance"""
        logger1 = get_logger(level=logging.INFO)
        logger2 = get_logger(level=logging.DEBUG)

        assert logger1 is logger2
        assert logger2.logger.level == logging.DEBUG

    def test_get_logger_preserves_level_when_none(self):
        """Test that get_logger preserves level when level=None"""
        logger1 = get_logger(level=logging.DEBUG)
        logger2 = get_logger(level=None)

        assert logger1 is logger2
        assert logger2.logger.level == logging.DEBUG

    def test_get_logger_default_level(self):
        """Test get_logger with no level uses INFO"""
        logger = get_logger()

        assert logger.logger.level == logging.INFO


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_verbose_mode(self):
        """Test complete workflow in verbose mode"""
        logger = get_logger(level=logging.DEBUG)

        # Set verbose
        logger.set_verbose(True)
        assert logger.verbose is True

        # Log messages
        with patch.object(logger.logger, 'debug') as mock_debug:
            with patch.object(logger.logger, 'info') as mock_info:
                logger.debug('Debug msg')
                logger.info('Info msg')

                mock_debug.assert_called_once()
                mock_info.assert_called_once()

    def test_full_workflow_quiet_mode(self):
        """Test complete workflow in quiet mode"""
        logger = get_logger(level=logging.WARNING)

        # Set non-verbose
        logger.set_verbose(False)
        assert logger.verbose is False
        assert logger.logger.level == logging.WARNING

    def test_level_changes_affect_singleton(self):
        """Test that level changes affect the singleton instance"""
        logger1 = get_logger(level=logging.INFO)
        logger1.set_level(logging.DEBUG)

        logger2 = get_logger()
        assert logger2.logger.level == logging.DEBUG

    def test_multiple_loggers_same_instance(self):
        """Test that multiple calls to get_logger return same instance"""
        loggers = [get_logger() for _ in range(5)]

        # All should be the same instance
        assert all(logger is loggers[0] for logger in loggers)


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_empty_message(self, logger):
        """Test logging empty message"""
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info('')
            mock_info.assert_called_once_with('')

    def test_very_long_message(self, logger):
        """Test logging very long message"""
        long_message = 'A' * 10000

        with patch.object(logger.logger, 'info') as mock_info:
            logger.info(long_message)
            mock_info.assert_called_once_with(long_message)

    def test_message_with_special_characters(self, logger):
        """Test logging message with special characters"""
        special_msg = 'Test: \n\t\r\\\"\'%s %d'

        with patch.object(logger.logger, 'info') as mock_info:
            logger.info(special_msg)
            mock_info.assert_called_once_with(special_msg)

    def test_multiple_level_changes(self, logger):
        """Test multiple rapid level changes"""
        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

        for level in levels:
            logger.set_level(level)
            assert logger.logger.level == level

    def test_verbose_toggle_multiple_times(self, logger):
        """Test toggling verbose multiple times"""
        for _ in range(10):
            logger.set_verbose(True)
            assert logger.verbose is True

            logger.set_verbose(False)
            assert logger.verbose is False

    def test_all_logging_methods_work(self, logger):
        """Test that all logging methods work without errors"""
        # Should not raise any exceptions
        logger.debug('debug')
        logger.info('info')
        logger.warning('warning')
        logger.error('error')
        logger.critical('critical')

        # Exception method
        try:
            raise ValueError("test error")
        except ValueError:
            logger.exception('exception occurred')


# ==================== Handler Tests ====================


class TestHandlers:
    """Tests for handler management"""

    def test_default_handler_added(self):
        """Test that default StreamHandler is added"""
        logger = DeepBridgeLogger()

        assert len(logger.logger.handlers) > 0
        assert any(isinstance(h, logging.StreamHandler) for h in logger.logger.handlers)

    def test_handler_has_formatter(self):
        """Test that handler has formatter set"""
        logger = DeepBridgeLogger()

        for handler in logger.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                assert handler.formatter is not None

    def test_no_duplicate_handlers_on_same_logger_name(self):
        """Test that creating loggers with same name doesn't duplicate handlers"""
        # First logger
        logger1 = DeepBridgeLogger(name='shared')
        handler_count1 = len(logger1.logger.handlers)

        # Second logger with same name
        logger2 = DeepBridgeLogger(name='shared')
        handler_count2 = len(logger2.logger.handlers)

        # Should have same number of handlers
        assert handler_count1 == handler_count2
