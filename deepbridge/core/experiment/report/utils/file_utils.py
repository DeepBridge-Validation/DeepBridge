"""
Simple utilities for file discovery.

Replaces FileDiscoveryManager with lightweight functions (Phase 2 Sprint 5-6).
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger("deepbridge.reports")


def find_files_by_pattern(directory: str, pattern: str) -> List[str]:
    """
    Find files matching pattern in directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.css", "**/*.js")

    Returns:
        List of absolute file paths

    Example:
        >>> css_files = find_files_by_pattern('/path/to/css', '*.css')
        >>> js_files = find_files_by_pattern('/path/to/js', '**/*.js')
    """
    path = Path(directory)
    if not path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    return sorted([str(f) for f in path.glob(pattern)])


def find_css_files(directory: str) -> Dict[str, str]:
    """
    Discover CSS files in directory with logical names.

    Looks for:
    - main.css, styles.css, or index.css as 'main'
    - components/*.css as component files
    - Other *.css files in root

    Args:
        directory: CSS directory to search

    Returns:
        Dictionary: {logical_name: relative_path}

    Example:
        >>> files = find_css_files('/path/to/css')
        >>> # {'main': 'main.css', 'buttons': 'components/buttons.css'}
    """
    path = Path(directory)
    if not path.exists():
        logger.warning(f"CSS directory does not exist: {directory}")
        return {}

    css_files = {}

    # Main CSS (required) - try common names
    main_candidates = ['main.css', 'styles.css', 'index.css', 'style.css']
    for name in main_candidates:
        main_path = path / name
        if main_path.exists():
            css_files['main'] = name
            logger.debug(f"Found main CSS: {name}")
            break

    # Components subdirectory
    components_dir = path / 'components'
    if components_dir.exists() and components_dir.is_dir():
        for css_file in components_dir.glob('*.css'):
            css_files[css_file.stem] = f"components/{css_file.name}"
            logger.debug(f"Found component CSS: {css_file.stem}")

    # Other CSS files in root (excluding main candidates)
    for css_file in path.glob('*.css'):
        if css_file.name not in main_candidates and css_file.stem not in css_files:
            css_files[css_file.stem] = css_file.name
            logger.debug(f"Found additional CSS: {css_file.stem}")

    return css_files


def find_js_files(directory: str) -> Dict[str, str]:
    """
    Discover JavaScript files in directory with logical names.

    Looks for:
    - main.js, utils.js in root
    - charts/*.js, controllers/*.js, components/*.js
    - Other *.js files in root

    Args:
        directory: JavaScript directory to search

    Returns:
        Dictionary: {logical_name: relative_path}

    Example:
        >>> files = find_js_files('/path/to/js')
        >>> # {'main': 'main.js', 'charts_line': 'charts/line.js'}
    """
    path = Path(directory)
    if not path.exists():
        logger.warning(f"JavaScript directory does not exist: {directory}")
        return {}

    js_files = {}

    # Main JS files in root
    special_files = ['main.js', 'utils.js']
    for special in special_files:
        special_path = path / special
        if special_path.exists():
            js_files[special_path.stem] = special
            logger.debug(f"Found {special}")

    # Common subdirectories
    subdirs = ['charts', 'controllers', 'components']
    for subdir in subdirs:
        subdir_path = path / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            for js_file in subdir_path.glob('*.js'):
                logical_name = f"{subdir}_{js_file.stem}"
                js_files[logical_name] = f"{subdir}/{js_file.name}"
                logger.debug(f"Found JS: {logical_name}")

    # Other JS files in root (excluding special files)
    for js_file in path.glob('*.js'):
        if js_file.name not in special_files and js_file.stem not in js_files:
            js_files[js_file.stem] = js_file.name
            logger.debug(f"Found additional JS: {js_file.stem}")

    return js_files


def find_asset_path(
    base_dir: str,
    test_type: str,
    asset_type: str,
    report_type: Optional[str] = None
) -> Optional[str]:
    """
    Find asset directory for test type (CSS or JS).

    Tries multiple common locations:
    - report_types/{test_type}/{asset_type}/
    - report_types/{test_type}/interactive/{asset_type}/
    - report_types/{test_type}/static/{asset_type}/

    Args:
        base_dir: Base templates directory
        test_type: Type of test ('uncertainty', 'robustness', etc.)
        asset_type: Type of asset ('css' or 'js')
        report_type: Optional report type to prioritize ('interactive' or 'static')

    Returns:
        Absolute path to asset directory, or None if not found

    Example:
        >>> path = find_asset_path('/templates', 'uncertainty', 'css', 'interactive')
        >>> # '/templates/report_types/uncertainty/interactive/css'
    """
    base_path = Path(base_dir)

    # Priority path if report_type specified
    if report_type and report_type.lower() in ['interactive', 'static']:
        priority = base_path / 'report_types' / test_type / report_type.lower() / asset_type
        if priority.exists():
            logger.debug(f"Using {report_type} {asset_type} path: {priority}")
            return str(priority)

    # Try common locations
    candidates = [
        base_path / 'report_types' / test_type / asset_type,
        base_path / 'report_types' / test_type / 'interactive' / asset_type,
        base_path / 'report_types' / test_type / 'static' / asset_type,
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.debug(f"Using {asset_type} path: {candidate}")
            return str(candidate)

    logger.warning(f"No {asset_type} directory found for {test_type}")
    return None


def read_html_files(directory: str) -> Dict[str, str]:
    """
    Read all HTML files in directory.

    Args:
        directory: Directory containing HTML files

    Returns:
        Dictionary: {filename: content}

    Example:
        >>> files = read_html_files('/path/to/partials')
        >>> # {'header.html': '<header>...</header>'}
    """
    path = Path(directory)
    if not path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return {}

    html_files = {}
    for html_file in path.glob('*.html'):
        try:
            content = html_file.read_text(encoding='utf-8')
            html_files[html_file.name] = content
            logger.debug(f"Loaded HTML file: {html_file.name}")
        except Exception as e:
            logger.warning(f"Error reading {html_file}: {e}")

    return html_files


def combine_text_files(file_paths: List[str], separator: str = "\n\n") -> str:
    """
    Combine multiple text files into one string.

    Args:
        file_paths: List of file paths to combine
        separator: String to use between files

    Returns:
        Combined content

    Example:
        >>> css = combine_text_files(['/path/base.css', '/path/components.css'])
    """
    combined = []

    for file_path in sorted(file_paths):
        try:
            content = Path(file_path).read_text(encoding='utf-8')
            combined.append(content)
            logger.debug(f"Added content from: {file_path}")
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")

    return separator.join(combined)
