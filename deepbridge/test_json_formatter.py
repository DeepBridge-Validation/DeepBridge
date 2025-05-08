"""
Test for the JSON formatter utility.
"""

import unittest
import json
from core.experiment.report.utils.json_formatter import JsonFormatter

class TestJsonFormatter(unittest.TestCase):
    """Test cases for the JsonFormatter utility."""
    
    def test_sanitize_json_string(self):
        """Test sanitizing JSON strings with trailing commas."""
        # Test with trailing comma in object
        json_with_trailing_comma = '{"name": "test", "value": 123,}'
        sanitized = JsonFormatter.sanitize_json_string(json_with_trailing_comma)
        self.assertEqual(sanitized, '{"name": "test", "value": 123}')
        
        # Test with trailing comma in array
        json_with_trailing_comma_array = '["a", "b", "c",]'
        sanitized = JsonFormatter.sanitize_json_string(json_with_trailing_comma_array)
        self.assertEqual(sanitized, '["a", "b", "c"]')
        
        # Test with nested structures
        nested_json = '{"data": {"items": [1, 2, 3,], "meta": {"count": 3,}}}'
        sanitized = JsonFormatter.sanitize_json_string(nested_json)
        self.assertEqual(sanitized, '{"data": {"items": [1, 2, 3], "meta": {"count": 3}}}')
        
        # Ensure valid JSON remains unchanged
        valid_json = '{"name": "test", "values": [1, 2, 3]}'
        sanitized = JsonFormatter.sanitize_json_string(valid_json)
        self.assertEqual(sanitized, valid_json)
    
    def test_format_for_javascript(self):
        """Test formatting Python data structures for JavaScript."""
        # Test with dict
        data = {"name": "test", "values": [1, 2, 3]}
        formatted = JsonFormatter.format_for_javascript(data)
        # Ensure the result is valid JSON
        parsed = json.loads(formatted)
        self.assertEqual(parsed, data)
        
        # Test with special values
        data_with_special = {"name": "test", "value": float('nan'), "inf": float('inf')}
        formatted = JsonFormatter.format_for_javascript(data_with_special)
        # Should replace NaN and Infinity with null
        parsed = json.loads(formatted)
        self.assertEqual(parsed, {"name": "test", "value": None, "inf": None})
    
    def test_validate_and_format(self):
        """Test validation and formatting of various inputs."""
        # Test with valid JSON string
        json_str = '{"name": "test"}'
        formatted = JsonFormatter.validate_and_format(json_str)
        self.assertEqual(json.loads(formatted), {"name": "test"})
        
        # Test with invalid JSON string that can be fixed
        invalid_json = '{"name": "test",}'
        formatted = JsonFormatter.validate_and_format(invalid_json)
        self.assertEqual(json.loads(formatted), {"name": "test"})
        
        # Test with Python dict
        data = {"name": "test", "values": [1, 2, 3]}
        formatted = JsonFormatter.validate_and_format(data)
        self.assertEqual(json.loads(formatted), data)
    
    def test_embed_in_html(self):
        """Test embedding JSON in HTML script tag."""
        data = {"name": "test", "values": [1, 2, 3]}
        html = JsonFormatter.embed_in_html(data, "testData")
        # Verify it contains the variable name and JSON data
        self.assertIn("const testData =", html)
        self.assertIn('"name":"test"', html.replace(" ", ""))
        self.assertIn('"values":[1,2,3]', html.replace(" ", ""))

if __name__ == '__main__':
    unittest.main()