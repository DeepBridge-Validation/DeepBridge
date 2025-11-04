"""
CSS Manager for DeepBridge Reports

Manages the three-layer CSS architecture:
  Layer 1: Base Styles (design tokens, reset, typography, utilities)
  Layer 2: Report Components (shared UI components)
  Layer 3: Custom Styles (report-specific overrides)

Version: 1.0
Date: 2025-10-29
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CSSManager:
    """
    Manages CSS compilation for DeepBridge reports.

    Compiles CSS from three layers:
    1. base_styles.css - Foundation styles
    2. report_components.css - Shared components
    3. {report_type}_custom.css - Report-specific styles

    Usage:
        css_manager = CSSManager()
        compiled_css = css_manager.get_compiled_css('robustness')
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize CSS Manager.

        Args:
            templates_dir: Path to templates directory. If None, uses default.
        """
        if templates_dir is None:
            # Get default templates directory
            current_file = Path(__file__).resolve()
            deepbridge_root = current_file.parent.parent.parent.parent
            templates_dir = deepbridge_root / 'templates'

        self.templates_dir = Path(templates_dir)
        self.base_css_path = self.templates_dir / 'base_styles.css'
        self.components_css_path = self.templates_dir / 'report_components.css'

        logger.info(f"CSSManager initialized with templates_dir: {self.templates_dir}")

    def get_compiled_css(self, report_type: str) -> str:
        """
        Compile CSS for a specific report type.

        Combines:
        1. Base styles (always included)
        2. Component styles (always included)
        3. Custom styles (report-specific, if exists)

        Args:
            report_type: Type of report ('robustness', 'resilience', 'uncertainty', etc.)

        Returns:
            Compiled CSS as a string

        Raises:
            FileNotFoundError: If base or components CSS not found
        """
        logger.info(f"Compiling CSS for report type: {report_type}")

        # Layer 1: Base Styles (required)
        base_css = self._read_css_file(self.base_css_path, required=True)

        # Layer 2: Component Styles (required)
        components_css = self._read_css_file(self.components_css_path, required=True)

        # Layer 3: Custom Styles (optional)
        custom_css_path = self._get_custom_css_path(report_type)
        custom_css = self._read_css_file(custom_css_path, required=False)

        # Compile all layers
        compiled_css = self._compile_layers(
            base_css,
            components_css,
            custom_css,
            report_type
        )

        logger.info(f"CSS compiled successfully. Total size: {len(compiled_css)} chars")
        return compiled_css

    def _get_custom_css_path(self, report_type: str) -> Path:
        """
        Get path to custom CSS file for report type.

        Checks in:
        1. /templates/report_types/{report_type}/interactive/{report_type}_custom.css
        2. /templates/report_types/{report_type}/interactive/css/{report_type}_custom.css

        Args:
            report_type: Type of report

        Returns:
            Path to custom CSS file
        """
        # Try path 1: direct in interactive folder
        path1 = (self.templates_dir / 'report_types' / report_type /
                 'interactive' / f'{report_type}_custom.css')

        if path1.exists():
            return path1

        # Try path 2: in css subfolder
        path2 = (self.templates_dir / 'report_types' / report_type /
                 'interactive' / 'css' / f'{report_type}_custom.css')

        if path2.exists():
            return path2

        # Return path1 as default (will be logged as not found)
        return path1

    def _read_css_file(self, css_path: Path, required: bool = False) -> str:
        """
        Read CSS file content.

        Args:
            css_path: Path to CSS file
            required: If True, raise error if file not found

        Returns:
            CSS content as string, or empty string if not found (and not required)

        Raises:
            FileNotFoundError: If file not found and required=True
        """
        if not css_path.exists():
            if required:
                logger.error(f"Required CSS file not found: {css_path}")
                raise FileNotFoundError(f"Required CSS file not found: {css_path}")
            else:
                logger.warning(f"Optional CSS file not found: {css_path}")
                return ""

        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.debug(f"Read CSS file: {css_path} ({len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"Error reading CSS file {css_path}: {e}")
            if required:
                raise
            return ""

    def _compile_layers(
        self,
        base_css: str,
        components_css: str,
        custom_css: str,
        report_type: str
    ) -> str:
        """
        Compile CSS layers into single stylesheet.

        Args:
            base_css: Layer 1 - Base styles
            components_css: Layer 2 - Component styles
            custom_css: Layer 3 - Custom styles
            report_type: Type of report (for comments)

        Returns:
            Compiled CSS with layer separators
        """
        separator = "\n\n/* " + "=" * 76 + " */\n"

        layers = [
            f"/*\n * DeepBridge Report - {report_type.capitalize()}\n"
            f" * CSS compiled from three layers\n"
            f" * Generated: {self._get_timestamp()}\n"
            f" */\n",

            base_css,
            separator,
            "/* LAYER 2: SHARED COMPONENTS */",
            separator,
            components_css
        ]

        # Add custom layer if exists
        if custom_css.strip():
            layers.extend([
                separator,
                f"/* LAYER 3: {report_type.upper()} CUSTOM STYLES */",
                separator,
                custom_css
            ])

        return "\n".join(layers)

    def _get_timestamp(self) -> str:
        """
        Get current timestamp for CSS compilation.

        Returns:
            Timestamp string
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def validate_css_files(self) -> dict:
        """
        Validate that required CSS files exist.

        Returns:
            Dictionary with validation results:
            {
                'base_styles': bool,
                'components': bool,
                'errors': list
            }
        """
        validation = {
            'base_styles': self.base_css_path.exists(),
            'components': self.components_css_path.exists(),
            'errors': []
        }

        if not validation['base_styles']:
            validation['errors'].append(f"Base styles not found: {self.base_css_path}")

        if not validation['components']:
            validation['errors'].append(f"Components not found: {self.components_css_path}")

        return validation

    def get_custom_css_info(self, report_type: str) -> dict:
        """
        Get information about custom CSS for report type.

        Args:
            report_type: Type of report

        Returns:
            Dictionary with custom CSS info:
            {
                'exists': bool,
                'path': str,
                'size': int (bytes)
            }
        """
        custom_css_path = self._get_custom_css_path(report_type)

        info = {
            'exists': custom_css_path.exists(),
            'path': str(custom_css_path),
            'size': 0
        }

        if info['exists']:
            info['size'] = custom_css_path.stat().st_size

        return info


# Convenience function for quick usage
def compile_report_css(report_type: str, templates_dir: Optional[str] = None) -> str:
    """
    Quick function to compile CSS for a report type.

    Args:
        report_type: Type of report ('robustness', 'resilience', 'uncertainty')
        templates_dir: Optional custom templates directory

    Returns:
        Compiled CSS as string

    Example:
        css = compile_report_css('robustness')
    """
    manager = CSSManager(templates_dir)
    return manager.get_compiled_css(report_type)


# Example usage and testing
if __name__ == '__main__':
    # Set up logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("CSS Manager Test")
    print("=" * 80)

    # Initialize manager
    manager = CSSManager()

    # Validate CSS files
    print("\n1. Validating CSS files...")
    validation = manager.validate_css_files()
    print(f"   Base styles: {'✓' if validation['base_styles'] else '✗'}")
    print(f"   Components: {'✓' if validation['components'] else '✗'}")
    if validation['errors']:
        print(f"   Errors: {validation['errors']}")

    # Test compilation for each report type
    report_types = ['robustness', 'resilience', 'uncertainty']

    for report_type in report_types:
        print(f"\n2. Testing {report_type} report...")

        # Get custom CSS info
        custom_info = manager.get_custom_css_info(report_type)
        print(f"   Custom CSS exists: {'✓' if custom_info['exists'] else '✗'}")
        if custom_info['exists']:
            print(f"   Custom CSS size: {custom_info['size']} bytes")

        # Compile CSS
        try:
            compiled = manager.get_compiled_css(report_type)
            print(f"   Compiled CSS size: {len(compiled)} chars")
            print(f"   Success: ✓")
        except Exception as e:
            print(f"   Error: {e}")
            print(f"   Success: ✗")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
