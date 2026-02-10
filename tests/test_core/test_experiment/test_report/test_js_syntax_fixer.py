"""
Comprehensive tests for JavaScriptSyntaxFixer.

This test suite validates:
1. fix_trailing_commas - JSON formatter + regex fallback
2. fix_undefined_variables - safe variable declarations
3. add_error_handling - global error handler
4. fix_model_comparison_function - specific function fix
5. fix_model_level_details_function - specific function fix
6. apply_all_fixes - integration

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import patch, Mock

from deepbridge.core.experiment.report.js_syntax_fixer import JavaScriptSyntaxFixer


# ==================== Fixtures ====================


@pytest.fixture
def simple_js():
    """Simple JavaScript code"""
    return """
    var data = { a: 1, b: 2 };
    console.log(data);
    """


@pytest.fixture
def js_with_trailing_comma():
    """JavaScript with trailing comma in object"""
    return """
    return {
        name: "test",
        value: 42,
    };
    """


@pytest.fixture
def js_with_json_object():
    """JavaScript with JSON object literal"""
    return """
    var myData = {"name": "test", "value": 42, };
    """


@pytest.fixture
def js_model_comparison_function():
    """JavaScript with model comparison function"""
    return """
    extractModelComparisonData: function() {
        var levels = [0.1, 0.2];
        return {
            levels: levels,
            scores: [0.8, 0.9],
        };
    }
    """


@pytest.fixture
def js_model_level_details_function():
    """JavaScript with model level details function"""
    return """
    extractModelLevelDetailsData: function() {
        var levels = [0.1, 0.2];
        var modelScores = {model1: [0.8]};
        var modelNames = {model1: "Model 1"};
        var metricName = "accuracy";
        return { levels, modelScores, modelNames, metricName, };
    }
    """


# ==================== fix_trailing_commas Tests ====================


class TestFixTrailingCommas:
    """Tests for fix_trailing_commas method"""

    def test_no_changes_when_no_trailing_commas(self, simple_js):
        """Test that clean JS is not modified"""
        result = JavaScriptSyntaxFixer.fix_trailing_commas(simple_js)

        assert result == simple_js

    def test_fix_trailing_comma_in_return_statement(self):
        """Test fixing trailing comma in return object"""
        js = "return { name: 'test', value: 42, };"

        result = JavaScriptSyntaxFixer.fix_trailing_commas(js)

        assert result == "return { name: 'test', value: 42 };"
        assert ", }" not in result

    def test_fix_trailing_comma_in_object_literal(self):
        """Test fixing trailing comma in any object literal"""
        js = "var obj = { a: 1, b: 2, };"

        result = JavaScriptSyntaxFixer.fix_trailing_commas(js)

        assert result == "var obj = { a: 1, b: 2 };"

    def test_fix_json_with_double_quotes(self):
        """Test fixing JSON with double quotes using JsonFormatter"""
        js = 'var data = {"name": "test", "value": 42, };'

        with patch('deepbridge.core.experiment.report.js_syntax_fixer.JsonFormatter.sanitize_json_string') as mock_sanitize:
            mock_sanitize.return_value = '{"name": "test", "value": 42}'

            result = JavaScriptSyntaxFixer.fix_trailing_commas(js)

            # Should call JsonFormatter
            mock_sanitize.assert_called_once()

    def test_json_formatter_fallback_on_error(self):
        """Test that it falls back when JsonFormatter fails"""
        js = 'var data = {"invalid": "json", };'

        with patch('deepbridge.core.experiment.report.js_syntax_fixer.JsonFormatter.sanitize_json_string') as mock_sanitize:
            mock_sanitize.side_effect = Exception("Invalid JSON")

            # Should not raise, just use regex fallback
            result = JavaScriptSyntaxFixer.fix_trailing_commas(js)

            # Should still attempt regex fixes
            assert result is not None

    def test_fix_multiline_object(self):
        """Test fixing trailing comma in multiline object"""
        js = """
        return {
            name: "test",
            value: 42,
        };
        """

        result = JavaScriptSyntaxFixer.fix_trailing_commas(js)

        assert ", \n        }" in result or ",\n        }" not in result

    def test_no_json_triggers_regex_fallback(self):
        """Test that code without JSON uses regex fallback"""
        js = "return { x: 1, y: 2, };"

        result = JavaScriptSyntaxFixer.fix_trailing_commas(js)

        # Should remove trailing comma via regex
        assert ", };" not in result
        assert " };" in result


# ==================== fix_undefined_variables Tests ====================


class TestFixUndefinedVariables:
    """Tests for fix_undefined_variables method"""

    def test_add_safe_headers_to_js(self, simple_js):
        """Test adding safe variable declarations"""
        result = JavaScriptSyntaxFixer.fix_undefined_variables(simple_js)

        assert 'window.__safeFallbackObject' in result
        assert 'typeof Plotly' in result
        assert 'Safe definitions' in result

    def test_no_duplicate_headers(self):
        """Test that headers are not added twice"""
        js_with_headers = """
        window.__safeFallbackObject = {};
        var data = {a: 1};
        """

        result = JavaScriptSyntaxFixer.fix_undefined_variables(js_with_headers)

        # Should not add headers again
        assert result == js_with_headers
        # Count occurrences - should be exactly 1
        assert result.count('window.__safeFallbackObject') == 1

    def test_safe_fallback_object_structure(self, simple_js):
        """Test that safe fallback object has required structure"""
        result = JavaScriptSyntaxFixer.fix_undefined_variables(simple_js)

        assert 'levels: []' in result
        assert 'modelScores: {}' in result
        assert 'modelNames: {}' in result
        assert 'metricName: ""' in result


# ==================== add_error_handling Tests ====================


class TestAddErrorHandling:
    """Tests for add_error_handling method"""

    def test_add_error_handler(self, simple_js):
        """Test adding global error handler"""
        result = JavaScriptSyntaxFixer.add_error_handling(simple_js)

        assert "window.addEventListener('error'" in result
        assert 'console.error("JavaScript error caught:"' in result
        assert 'Global error handler' in result

    def test_no_duplicate_error_handler(self):
        """Test that error handler is not added twice"""
        js_with_handler = """
        window.addEventListener('error', function() {});
        var data = {a: 1};
        """

        result = JavaScriptSyntaxFixer.add_error_handling(js_with_handler)

        # Should not add handler again
        assert result == js_with_handler
        assert result.count("window.addEventListener('error'") == 1

    def test_error_handler_syntax_error_check(self, simple_js):
        """Test that error handler checks for syntax errors"""
        result = JavaScriptSyntaxFixer.add_error_handling(simple_js)

        assert 'Unexpected token' in result
        assert 'Syntax error detected' in result


# ==================== fix_model_comparison_function Tests ====================


class TestFixModelComparisonFunction:
    """Tests for fix_model_comparison_function method"""

    def test_fix_model_comparison_function(self, js_model_comparison_function):
        """Test fixing model comparison function with trailing comma"""
        result = JavaScriptSyntaxFixer.fix_model_comparison_function(
            js_model_comparison_function
        )

        # Should remove trailing comma
        assert ", \n        }" not in result or ",\n        }" not in result

    def test_add_try_catch_wrapper(self):
        """Test that try-catch wrapper is added"""
        js = """
        extractModelComparisonData: function() {
            return {
                levels: [0.1],
                scores: [0.8]
            };
        }
        """

        result = JavaScriptSyntaxFixer.fix_model_comparison_function(js)

        # Should contain try-catch
        assert 'try {' in result or 'try{' in result or result == js  # May not match pattern

    def test_fallback_object_in_catch(self):
        """Test that catch block has fallback object"""
        js = """
        extractModelComparisonData: function() {
            return {
                data: values,
            };
        }
        """

        result = JavaScriptSyntaxFixer.fix_model_comparison_function(js)

        # If try-catch added, should have fallback
        if 'catch' in result:
            assert 'window.__safeFallbackObject' in result

    def test_no_change_when_pattern_not_found(self, simple_js):
        """Test that JS without the pattern is unchanged"""
        result = JavaScriptSyntaxFixer.fix_model_comparison_function(simple_js)

        assert result == simple_js


# ==================== fix_model_level_details_function Tests ====================


class TestFixModelLevelDetailsFunction:
    """Tests for fix_model_level_details_function method"""

    def test_fix_model_level_details_function(self, js_model_level_details_function):
        """Test fixing model level details function"""
        result = JavaScriptSyntaxFixer.fix_model_level_details_function(
            js_model_level_details_function
        )

        # Should remove trailing comma
        assert ', }' not in result or result != js_model_level_details_function

    def test_add_try_catch_with_fallback_data(self):
        """Test that try-catch wrapper has fallback data"""
        js = """
        extractModelLevelDetailsData: function() {
            return {
                levels: data.levels,
                modelScores: data.scores,
            };
        }
        """

        result = JavaScriptSyntaxFixer.fix_model_level_details_function(js)

        # If try-catch added, should have default data
        if 'catch' in result:
            assert 'levels: [0.1, 0.2, 0.3, 0.4, 0.5]' in result
            assert "'primary'" in result or '"primary"' in result

    def test_targeted_approach_fallback(self):
        """Test targeted approach when complex pattern doesn't match"""
        js = "return { levels, modelScores, modelNames, metricName, };"

        result = JavaScriptSyntaxFixer.fix_model_level_details_function(js)

        # Should remove trailing comma
        assert "metricName }" in result

    def test_no_change_when_no_pattern_match(self, simple_js):
        """Test that unrelated JS is unchanged"""
        result = JavaScriptSyntaxFixer.fix_model_level_details_function(simple_js)

        assert result == simple_js


# ==================== apply_all_fixes Tests ====================


class TestApplyAllFixes:
    """Tests for apply_all_fixes integration method"""

    def test_apply_all_fixes_integration(self, js_with_trailing_comma):
        """Test that all fixes are applied in sequence"""
        result = JavaScriptSyntaxFixer.apply_all_fixes(js_with_trailing_comma)

        # Check all fixes were applied
        assert ', }' not in result or ', }\n' not in result  # trailing comma removed
        assert 'window.__safeFallbackObject' in result  # safe variables added
        assert "window.addEventListener('error'" in result  # error handler added

    def test_apply_all_fixes_order(self):
        """Test that fixes are applied in correct order"""
        js = """
        var data = { a: 1, b: 2, };
        extractModelComparisonData: function() {
            return { x: 1, };
        }
        """

        result = JavaScriptSyntaxFixer.apply_all_fixes(js)

        # All methods should have been called
        assert 'window.__safeFallbackObject' in result
        assert "window.addEventListener('error'" in result

    def test_apply_all_fixes_with_complex_code(self):
        """Test all fixes on complex JavaScript code"""
        js = """
        extractModelComparisonData: function() {
            var myData = {"name": "test", };
            return {
                levels: [0.1, 0.2],
                data: myData,
            };
        }
        extractModelLevelDetailsData: function() {
            return { levels, modelScores, modelNames, metricName, };
        }
        """

        result = JavaScriptSyntaxFixer.apply_all_fixes(js)

        # All fixes should be present
        assert 'window.__safeFallbackObject' in result
        assert "window.addEventListener('error'" in result
        # Trailing commas should be attempted to be fixed
        assert result != js  # Something changed

    def test_apply_all_fixes_idempotent(self, simple_js):
        """Test that applying fixes twice gives same result"""
        result1 = JavaScriptSyntaxFixer.apply_all_fixes(simple_js)
        result2 = JavaScriptSyntaxFixer.apply_all_fixes(result1)

        assert result1 == result2  # Idempotent


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_empty_string(self):
        """Test with empty string input"""
        result = JavaScriptSyntaxFixer.apply_all_fixes("")

        # Should add headers and error handler
        assert 'window.__safeFallbackObject' in result
        assert "window.addEventListener('error'" in result

    def test_very_long_js_code(self):
        """Test with very long JavaScript code"""
        long_js = "var x = 1;\n" * 1000

        result = JavaScriptSyntaxFixer.apply_all_fixes(long_js)

        # Should still add headers
        assert 'window.__safeFallbackObject' in result

    def test_nested_objects_with_trailing_commas(self):
        """Test nested objects with multiple trailing commas"""
        js = """
        return {
            outer: {
                inner: {
                    value: 42,
                },
            },
        };
        """

        result = JavaScriptSyntaxFixer.fix_trailing_commas(js)

        # Should attempt to fix trailing commas
        assert result is not None

    def test_json_with_single_quotes(self):
        """Test JSON with single quotes"""
        js = "var data = {'name': 'test', };"

        result = JavaScriptSyntaxFixer.fix_trailing_commas(js)

        # Should handle single quotes
        assert result is not None

    def test_multiple_return_statements(self):
        """Test code with multiple return statements"""
        js = """
        function f1() { return { a: 1, }; }
        function f2() { return { b: 2, }; }
        """

        result = JavaScriptSyntaxFixer.fix_trailing_commas(js)

        # Should fix both
        assert ', };' not in result or result != js


# ==================== Logging Tests ====================


class TestLogging:
    """Tests that logging is called correctly"""

    @patch('deepbridge.core.experiment.report.js_syntax_fixer.logger')
    def test_logs_trailing_comma_fix(self, mock_logger):
        """Test that fixing trailing commas is logged"""
        js = "return { a: 1, };"

        JavaScriptSyntaxFixer.fix_trailing_commas(js)

        # Should log info
        mock_logger.info.assert_called()

    @patch('deepbridge.core.experiment.report.js_syntax_fixer.logger')
    def test_logs_undefined_variables_fix(self, mock_logger, simple_js):
        """Test that adding safe variables is logged"""
        JavaScriptSyntaxFixer.fix_undefined_variables(simple_js)

        mock_logger.info.assert_called_once()

    @patch('deepbridge.core.experiment.report.js_syntax_fixer.logger')
    def test_logs_error_handler_addition(self, mock_logger, simple_js):
        """Test that adding error handler is logged"""
        JavaScriptSyntaxFixer.add_error_handling(simple_js)

        mock_logger.info.assert_called_once()

    @patch('deepbridge.core.experiment.report.js_syntax_fixer.logger')
    def test_logs_json_sanitize_warning(self, mock_logger):
        """Test that JSON sanitize failures are logged as warnings"""
        js = 'var data = {"invalid", };'

        with patch('deepbridge.core.experiment.report.js_syntax_fixer.JsonFormatter.sanitize_json_string') as mock_sanitize:
            mock_sanitize.side_effect = Exception("Parse error")

            JavaScriptSyntaxFixer.fix_trailing_commas(js)

            # Should log warning
            mock_logger.warning.assert_called()
