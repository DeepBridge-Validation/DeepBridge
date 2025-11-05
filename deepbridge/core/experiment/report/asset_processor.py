"""
Asset processor module - Simplified (Phase 2 Sprint 5-6).

Processes assets for report generation with streamlined architecture:
- CSS: Delegated to CSSManager
- JS: Simplified using file_utils
- Images: Cached base64 encoding
"""

import os
import base64
import logging
from typing import Dict, Any
from functools import lru_cache

from .css_manager import CSSManager
from .utils import file_utils

logger = logging.getLogger("deepbridge.reports")


class AssetProcessor:
    """
    Simplified asset processor for report generation.

    Delegations:
    - CSS → CSSManager
    - File discovery → file_utils
    - Images → Cached base64 encoding

    Phase 2 Sprint 5-6: Reduced from ~700 lines to ~250 lines.
    """

    def __init__(self, asset_manager):
        """
        Initialize asset processor.

        Parameters:
        -----------
        asset_manager : AssetManager
            Parent asset manager
        """
        self.asset_manager = asset_manager
        self.css_manager = CSSManager()

    # ==================================================================================
    # CSS Methods - Delegated to CSSManager
    # ==================================================================================

    def get_css_content(self, test_type: str) -> str:
        """
        Get CSS content for test type (delegates to CSSManager).

        Args:
            test_type: Type of test ('uncertainty', 'robustness', etc.)

        Returns:
            Compiled CSS content

        Example:
            >>> css = processor.get_css_content('uncertainty')
        """
        return self.css_manager.get_compiled_css(test_type)

    def get_generic_css_content(self) -> str:
        """
        Get generic CSS content (delegates to CSSManager).

        Returns:
            Generic CSS content
        """
        return self.css_manager.get_compiled_css('generic')

    def get_combined_css_content(self, test_type: str) -> str:
        """
        Get combined CSS content for test type.

        Alias for get_css_content() for backward compatibility.

        Args:
            test_type: Type of test

        Returns:
            Combined CSS content
        """
        return self.get_css_content(test_type)

    # ==================================================================================
    # JavaScript Methods - Simplified with file_utils
    # ==================================================================================

    def get_js_content(self, js_dir: str) -> str:
        """
        Get combined JavaScript content from directory.

        Args:
            js_dir: Directory containing JS files

        Returns:
            Combined JavaScript content

        Raises:
            FileNotFoundError: If directory doesn't exist

        Example:
            >>> js = processor.get_js_content('/path/to/js')
        """
        if not os.path.exists(js_dir):
            raise FileNotFoundError(f"JavaScript directory not found: {js_dir}")

        # Discover JS files
        files = file_utils.find_js_files(js_dir)

        if not files:
            logger.warning(f"No JavaScript files found in {js_dir}")
            return ""

        # Build full paths in priority order
        priority_order = ['main', 'utils']  # Load these first
        file_paths = []

        # Add priority files first
        for priority in priority_order:
            if priority in files:
                file_paths.append(os.path.join(js_dir, files[priority]))

        # Add remaining files
        for name, rel_path in sorted(files.items()):
            if name not in priority_order:
                full_path = os.path.join(js_dir, rel_path)
                if full_path not in file_paths:
                    file_paths.append(full_path)

        # Combine files
        combined_js = "/* ----- Combined JavaScript ----- */\n\n"
        combined_js += file_utils.combine_text_files(file_paths, separator="\n\n")

        logger.info(f"Combined {len(file_paths)} JavaScript files from {js_dir}")
        return combined_js

    def get_generic_js_content(self) -> str:
        """
        Get generic JavaScript content from assets/js directory.

        Returns:
            Generic JavaScript content

        Raises:
            FileNotFoundError: If generic JS directory doesn't exist
        """
        js_dir = os.path.join(self.asset_manager.assets_dir, 'js')

        if not os.path.exists(js_dir):
            raise FileNotFoundError(f"Generic JavaScript directory not found: {js_dir}")

        return self.get_js_content(js_dir)

    def get_combined_js_content(self, test_type: str) -> str:
        """
        Get combined JavaScript content for test type.

        Combines:
        1. Test-specific JS from report_types/{test_type}/js
        2. Generic JS from assets/js

        Args:
            test_type: Type of test ('uncertainty', 'robustness', etc.)

        Returns:
            Combined JavaScript content

        Example:
            >>> js = processor.get_combined_js_content('uncertainty')
        """
        combined_js = "/* ===== Combined JavaScript for {} ===== */\n\n".format(test_type)

        # 1. Get test-specific JS
        js_path = file_utils.find_asset_path(
            self.asset_manager.templates_dir,
            test_type,
            'js'
        )

        if js_path:
            try:
                test_js = self.get_js_content(js_path)
                combined_js += "/* Test-Specific JavaScript */\n"
                combined_js += test_js + "\n\n"
                logger.info(f"Added test-specific JavaScript for {test_type}")
            except Exception as e:
                logger.warning(f"Could not load test-specific JS for {test_type}: {e}")
        else:
            logger.info(f"No test-specific JavaScript found for {test_type}")

        # 2. Get generic JS
        try:
            generic_js = self.get_generic_js_content()
            combined_js += "/* Generic JavaScript */\n"
            combined_js += generic_js
            logger.info("Added generic JavaScript")
        except Exception as e:
            logger.warning(f"Could not load generic JavaScript: {e}")

        return combined_js

    # ==================================================================================
    # Image Methods - Cached Base64 Encoding
    # ==================================================================================

    def get_base64_image(self, image_path: str) -> str:
        """
        Convert image to base64 data URL.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded data URL (data:image/png;base64,...)

        Raises:
            FileNotFoundError: If image doesn't exist

        Example:
            >>> data_url = processor.get_base64_image('/path/to/image.png')
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Determine MIME type from extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon'
        }
        mime_type = mime_types.get(ext, 'image/png')

        # Read and encode
        try:
            with open(image_path, "rb") as img_file:
                base64_str = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:{mime_type};base64,{base64_str}"
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise

    @lru_cache(maxsize=1)
    def get_logo_base64(self) -> str:
        """
        Get base64 encoded logo (cached).

        Returns:
            Base64 data URL for logo

        Raises:
            FileNotFoundError: If logo doesn't exist
        """
        if not os.path.exists(self.asset_manager.logo_path):
            raise FileNotFoundError(f"Logo file not found: {self.asset_manager.logo_path}")
        return self.get_base64_image(self.asset_manager.logo_path)

    @lru_cache(maxsize=1)
    def get_favicon_base64(self) -> str:
        """
        Get base64 encoded favicon (cached).

        Returns:
            Base64 data URL for favicon

        Raises:
            FileNotFoundError: If favicon doesn't exist
        """
        if not os.path.exists(self.asset_manager.favicon_path):
            raise FileNotFoundError(f"Favicon file not found: {self.asset_manager.favicon_path}")
        return self.get_base64_image(self.asset_manager.favicon_path)

    def get_icons(self) -> Dict[str, str]:
        """
        Get all icons as base64 data URLs.

        Returns:
            Dictionary mapping icon names to base64 data URLs

        Example:
            >>> icons = processor.get_icons()
            >>> # {'warning': 'data:image/svg+xml;base64,...', ...}
        """
        icons = {}
        icons_dir = os.path.join(self.asset_manager.images_dir, 'icons')

        if not os.path.exists(icons_dir):
            logger.warning(f"Icons directory not found: {icons_dir}")
            return icons

        # Find all image files
        icon_files = file_utils.find_files_by_pattern(
            icons_dir,
            '*.{svg,png,jpg,jpeg,ico,gif}'
        )

        if not icon_files:
            # Try individual patterns
            for ext in ['svg', 'png', 'jpg', 'jpeg', 'ico', 'gif']:
                icon_files.extend(file_utils.find_files_by_pattern(icons_dir, f'*.{ext}'))

        # Encode each icon
        for icon_path in icon_files:
            icon_name = os.path.splitext(os.path.basename(icon_path))[0]
            try:
                icons[icon_name] = self.get_base64_image(icon_path)
                logger.debug(f"Loaded icon: {icon_name}")
            except Exception as e:
                logger.warning(f"Could not load icon {icon_name}: {e}")

        logger.info(f"Loaded {len(icons)} icons")
        return icons

    # ==================================================================================
    # Complete Asset Bundle
    # ==================================================================================

    def create_full_report_assets(self, test_type: str) -> Dict[str, Any]:
        """
        Create complete asset bundle for report.

        Args:
            test_type: Type of test

        Returns:
            Dictionary containing all assets:
            - css_content: Combined CSS
            - js_content: Combined JavaScript
            - logo: Base64 logo
            - favicon: Base64 favicon
            - icons: Dictionary of base64 icons

        Example:
            >>> assets = processor.create_full_report_assets('uncertainty')
            >>> html = template.render(**assets)
        """
        logger.info(f"Creating full asset bundle for {test_type}")

        assets = {}

        # CSS
        try:
            assets['css_content'] = self.get_combined_css_content(test_type)
            logger.info("CSS loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CSS: {e}")
            assets['css_content'] = ""

        # JavaScript
        try:
            assets['js_content'] = self.get_combined_js_content(test_type)
            logger.info("JavaScript loaded successfully")
        except Exception as e:
            logger.error(f"Error loading JavaScript: {e}")
            assets['js_content'] = ""

        # Logo
        try:
            assets['logo'] = self.get_logo_base64()
            logger.info("Logo loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading logo: {e}")
            assets['logo'] = ""

        # Favicon
        try:
            assets['favicon'] = self.get_favicon_base64()
            logger.info("Favicon loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading favicon: {e}")
            assets['favicon'] = ""

        # Icons
        try:
            assets['icons'] = self.get_icons()
            logger.info(f"Loaded {len(assets['icons'])} icons")
        except Exception as e:
            logger.warning(f"Error loading icons: {e}")
            assets['icons'] = {}

        logger.info(f"Asset bundle created for {test_type}")
        return assets
