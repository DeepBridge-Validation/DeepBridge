"""
Dependency management system for DeepBridge experiments.
This module provides utilities to check, manage and dynamically load required dependencies.
"""

import importlib
import importlib.metadata
import sys
import os
import subprocess
import logging
import json
import platform
import warnings
from functools import lru_cache
from typing import List, Dict, Tuple, Set, Optional, Any, Callable, Union

# Setup logging
logger = logging.getLogger(__name__)

# Define dependencies by component
DEPENDENCIES = {
    # Core dependencies required for basic functionality
    'core': {
        'required': {
            'pandas': 'pandas',       # For data manipulation
            'numpy': 'numpy',         # For numerical operations
            'scikit-learn': 'sklearn',  # For machine learning algorithms
        },
        'optional': {
            'scipy': 'scipy',         # For scientific computing
        }
    },
    
    # Report generation dependencies
    'reporting': {
        'required': {
            'jinja2': 'jinja2',       # For template rendering
            'plotly': 'plotly',       # For interactive visualizations
        },
        'optional': {
            'matplotlib': 'matplotlib',  # For basic plotting
        }
    },
    
    # Visualization dependencies
    'visualization': {
        'required': {
            'plotly': 'plotly',       # For interactive visualizations
        },
        'optional': {
            'matplotlib': 'matplotlib',  # For basic plotting
            'seaborn': 'seaborn',      # For enhanced visualizations
        }
    },
    
    # Test-specific dependencies
    'robustness': {
        'required': {},
        'optional': {
            'torch': 'torch',         # For adversarial perturbations
        }
    },
    
    'uncertainty': {
        'required': {},
        'optional': {
            'tensorflow': 'tensorflow',  # For some uncertainty methods
        }
    }
}

# Consolidate all dependencies for easy access
ALL_REQUIRED_PACKAGES = {}
ALL_OPTIONAL_PACKAGES = {}

for component, deps in DEPENDENCIES.items():
    ALL_REQUIRED_PACKAGES.update(deps['required'])
    ALL_OPTIONAL_PACKAGES.update(deps['optional'])

# Define minimum versions for critical packages
MIN_VERSIONS = {
    'pandas': '1.0.0',
    'numpy': '1.18.0',
    'scikit-learn': '0.22.0',
    'jinja2': '2.10.0',
    'plotly': '4.0.0',
}

# Cache for package versions and availability
_package_cache = {}

@lru_cache(maxsize=128)
def get_package_version(package_name: str) -> Optional[str]:
    """
    Get the installed version of a package.
    
    Args:
        package_name: Name of the package
        
    Returns:
        Version string or None if package is not installed
    """
    # Check if the result is already in the cache
    if package_name in _package_cache:
        return _package_cache[package_name]
        
    try:
        version = importlib.metadata.version(package_name)
        _package_cache[package_name] = version
        return version
    except importlib.metadata.PackageNotFoundError:
        _package_cache[package_name] = None
        return None
    except Exception as e:
        logger.warning(f"Error getting version for {package_name}: {str(e)}")
        return None

@lru_cache(maxsize=128)
def is_package_installed(import_name: str) -> bool:
    """
    Check if a package is installed by attempting to import it.
    
    Args:
        import_name: Import name of the package to check
        
    Returns:
        Boolean indicating if the package is installed
    """
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def check_version_compatibility(package_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if the installed version of a package meets the minimum requirement.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        Tuple containing:
        - Boolean indicating if the version is compatible
        - Installed version or None if not installed
        - Minimum required version or None if no minimum specified
    """
    from packaging import version
    
    installed_version = get_package_version(package_name)
    required_version = MIN_VERSIONS.get(package_name)
    
    if not installed_version:
        return False, None, required_version
        
    if not required_version:
        return True, installed_version, None
        
    is_compatible = version.parse(installed_version) >= version.parse(required_version)
    return is_compatible, installed_version, required_version

def check_component_dependencies(component: str) -> Dict[str, Any]:
    """
    Check dependencies for a specific component.
    
    Args:
        component: Name of the component
        
    Returns:
        Dictionary with dependency status
    """
    if component not in DEPENDENCIES:
        raise ValueError(f"Unknown component: {component}")
        
    result = {
        'component': component,
        'all_required_installed': True,
        'missing_required': [],
        'missing_optional': [],
        'version_issues': [],
    }
    
    # Check required packages
    for package_name, import_name in DEPENDENCIES[component]['required'].items():
        if not is_package_installed(import_name):
            result['missing_required'].append(package_name)
            result['all_required_installed'] = False
        else:
            # Check version compatibility
            is_compatible, installed_version, required_version = check_version_compatibility(package_name)
            if required_version and not is_compatible:
                result['version_issues'].append({
                    'package': package_name,
                    'installed': installed_version,
                    'required': required_version
                })
                # Version issues with required packages are considered blocking
                result['all_required_installed'] = False
    
    # Check optional packages
    for package_name, import_name in DEPENDENCIES[component]['optional'].items():
        if not is_package_installed(import_name):
            result['missing_optional'].append(package_name)
        else:
            # Check version compatibility (but don't affect all_required_installed)
            is_compatible, installed_version, required_version = check_version_compatibility(package_name)
            if required_version and not is_compatible:
                result['version_issues'].append({
                    'package': package_name,
                    'installed': installed_version,
                    'required': required_version
                })
    
    return result

def check_dependencies(components: Optional[List[str]] = None) -> Tuple[bool, List[str], List[str], List[Dict]]:
    """
    Check if required dependencies are installed for all or specific components.
    
    Args:
        components: List of component names to check (None for all components)
        
    Returns:
        Tuple containing:
        - Boolean indicating if all required dependencies are installed
        - List of missing required dependencies
        - List of missing optional dependencies
        - List of version compatibility issues
    """
    all_components = DEPENDENCIES.keys() if components is None else components
    
    # Validate component names
    for component in all_components:
        if component not in DEPENDENCIES:
            raise ValueError(f"Unknown component: {component}")
    
    all_required_installed = True
    missing_required = []
    missing_optional = []
    version_issues = []
    
    # Check each component
    for component in all_components:
        result = check_component_dependencies(component)
        
        # Update overall status
        if not result['all_required_installed']:
            all_required_installed = False
            
        # Add unique missing packages
        for pkg in result['missing_required']:
            if pkg not in missing_required:
                missing_required.append(pkg)
                
        for pkg in result['missing_optional']:
            if pkg not in missing_optional:
                missing_optional.append(pkg)
                
        # Add version issues
        for issue in result['version_issues']:
            if issue not in version_issues:
                version_issues.append(issue)
    
    return all_required_installed, missing_required, missing_optional, version_issues

def get_install_command(missing_packages: List[str], upgrade: bool = False) -> str:
    """
    Generate pip install command for missing packages.
    
    Args:
        missing_packages: List of missing package names
        upgrade: Whether to upgrade existing packages
        
    Returns:
        String with pip install command
    """
    if not missing_packages:
        return "No packages to install"
    
    # Consider minimum versions for the packages
    packages_with_versions = []
    for pkg in missing_packages:
        min_version = MIN_VERSIONS.get(pkg)
        if min_version:
            packages_with_versions.append(f"{pkg}>={min_version}")
        else:
            packages_with_versions.append(pkg)
    
    upgrade_flag = "--upgrade" if upgrade else ""
    return f"pip install {upgrade_flag} {' '.join(packages_with_versions)}"

class DependencyError(Exception):
    """Exception raised for dependency-related errors."""
    pass

class IncompatibleVersionError(DependencyError):
    """Exception raised when a package version is incompatible."""
    pass

class MissingDependencyError(DependencyError):
    """Exception raised when a required dependency is missing."""
    pass

def try_import(package_name: str, import_name: Optional[str] = None) -> Optional[Any]:
    """
    Try to import a package safely, returning None if it fails.
    
    Args:
        package_name: Name of the package for error messages
        import_name: Name to import (defaults to package_name)
        
    Returns:
        The imported module or None if import fails
    """
    import_name = import_name or package_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        logger.debug(f"Failed to import {package_name}")
        return None

def require_package(package_name: str, import_name: Optional[str] = None, 
                   min_version: Optional[str] = None) -> Any:
    """
    Import a required package, raising an error if not available.
    
    Args:
        package_name: Name of the package for error messages
        import_name: Name to import (defaults to package_name)
        min_version: Minimum required version
        
    Returns:
        The imported module
        
    Raises:
        MissingDependencyError: If the package is not installed
        IncompatibleVersionError: If the package version is incompatible
    """
    import_name = import_name or package_name
    min_version = min_version or MIN_VERSIONS.get(package_name)
    
    # Try to import
    module = try_import(package_name, import_name)
    if module is None:
        raise MissingDependencyError(
            f"Required package '{package_name}' is not installed. "
            f"Please install it with: pip install {package_name}"
        )
    
    # Check version if required
    if min_version:
        from packaging import version
        installed_version = get_package_version(package_name)
        if installed_version and version.parse(installed_version) < version.parse(min_version):
            raise IncompatibleVersionError(
                f"Installed version of {package_name} ({installed_version}) "
                f"is older than required version ({min_version}). "
                f"Please upgrade with: pip install --upgrade {package_name}>={min_version}"
            )
    
    return module

def optional_import(package_name: str, import_name: Optional[str] = None, 
                   warning: bool = True) -> Optional[Any]:
    """
    Import an optional package, returning None and optionally warning if not available.
    
    Args:
        package_name: Name of the package for error messages
        import_name: Name to import (defaults to package_name)
        warning: Whether to emit a warning if the package is not available
        
    Returns:
        The imported module or None if not available
    """
    import_name = import_name or package_name
    module = try_import(package_name, import_name)
    
    if module is None and warning:
        warnings.warn(
            f"Optional package '{package_name}' is not installed. "
            f"Some functionality may be limited. "
            f"To enable all features, install it with: pip install {package_name}"
        )
    
    return module

def fallback_dependencies() -> Dict[str, Any]:
    """
    Load fallback implementations for critical dependencies if needed.
    
    Returns:
        Dictionary of fallback modules that were loaded
    """
    fallbacks = {}
    
    # Jinja2 fallback for template rendering
    if not is_package_installed('jinja2'):
        fallbacks['jinja2'] = _create_jinja2_fallback()
    
    # Plotly fallback for visualization
    if not is_package_installed('plotly'):
        fallbacks['plotly'] = _create_plotly_fallback()
    
    return fallbacks

def _create_jinja2_fallback():
    """Create and register a fallback implementation for Jinja2."""
    # Add a simple template string replacement function to sys.modules
    class SimpleTemplate:
        def __init__(self, template_string):
            self.template = template_string
            
        def render(self, **kwargs):
            result = self.template
            for key, value in kwargs.items():
                # Handle both {{key}} and {{ key }} patterns with spaces
                placeholders = [
                    '{{' + key + '}}',
                    '{{ ' + key + ' }}',
                    '{%if ' + key + '%}',
                    '{% if ' + key + ' %}',
                    '{%for ' + key + ' in',
                    '{% for ' + key + ' in'
                ]
                for placeholder in placeholders:
                    if placeholder in result:
                        result = result.replace(placeholder, str(value))
            return result
    
    class SimpleMockJinja:
        @staticmethod
        def Template(template_string):
            return SimpleTemplate(template_string)
            
        @staticmethod
        def from_string(template_string):
            return SimpleTemplate(template_string)
    
    # Define exceptions for error handling
    class UndefinedError(Exception):
        pass
        
    class TemplateError(Exception):
        pass
        
    # Mock for undefined variables
    class Undefined:
        def __str__(self):
            return ""
            
    class StrictUndefined(Undefined):
        def __str__(self):
            raise UndefinedError("Variable is undefined")
            
    class ChainableUndefined(Undefined):
        def __getattr__(self, name):
            return self
            
        def __getitem__(self, key):
            return self
    
    # Create a comprehensive mock for jinja2
    class Environment:
        def __init__(self, undefined=None, autoescape=None, **kwargs):
            self.undefined = undefined
            self.autoescape = autoescape
            self.filters = {}
            
        def from_string(self, template_string):
            return SimpleTemplate(template_string)
            
        def get_template(self, template_name):
            # In a real fallback we'd need to read the file here
            return SimpleTemplate("")
    
    # Create a complete Jinja2Mock with all needed components
    class Jinja2Mock:
        def __init__(self):
            self.Template = SimpleMockJinja.Template
            self.from_string = SimpleMockJinja.from_string
            self.Environment = Environment
            self.exceptions = type('exceptions', (), {
                'UndefinedError': UndefinedError,
                'TemplateError': TemplateError
            })
            self.Undefined = Undefined
            self.StrictUndefined = StrictUndefined
            self.ChainableUndefined = ChainableUndefined
    
    # Add to sys.modules to simulate jinja2 being available
    jinja2_mock = Jinja2Mock()
    sys.modules['jinja2'] = jinja2_mock
    logger.warning("⚠️ Using simplified Jinja2 fallback. Install jinja2 for proper template rendering.")
    
    return jinja2_mock

def _create_plotly_fallback():
    """Create and register a fallback implementation for Plotly."""
    # Create a simple Figure class that mimics plotly.graph_objects.Figure
    class Figure:
        def __init__(self, data=None, layout=None):
            self.data = data or []
            self.layout = layout or {}
            
        def update_layout(self, **kwargs):
            self.layout.update(kwargs)
            return self
            
        def add_trace(self, trace):
            self.data.append(trace)
            return self
            
        def to_json(self):
            return json.dumps({'data': self.data, 'layout': self.layout})
            
        def to_html(self):
            return f"<div><p>Plotly visualization (fallback mode)</p></div>"
    
    # Create simple trace classes
    class Scatter:
        def __init__(self, x=None, y=None, mode="lines", name=None, **kwargs):
            self.x = x
            self.y = y
            self.mode = mode
            self.name = name
            self.kwargs = kwargs
    
    class Bar:
        def __init__(self, x=None, y=None, name=None, **kwargs):
            self.x = x
            self.y = y
            self.name = name
            self.kwargs = kwargs
    
    class Box:
        def __init__(self, y=None, name=None, **kwargs):
            self.y = y
            self.name = name
            self.kwargs = kwargs
    
    # Create a mock graph_objects module
    class GraphObjects:
        def __init__(self):
            self.Figure = Figure
            self.Scatter = Scatter
            self.Bar = Bar
            self.Box = Box
    
    # Create a mock express module
    class Express:
        @staticmethod
        def line(data_frame=None, x=None, y=None, **kwargs):
            return Figure()
            
        @staticmethod
        def bar(data_frame=None, x=None, y=None, **kwargs):
            return Figure()
            
        @staticmethod
        def scatter(data_frame=None, x=None, y=None, **kwargs):
            return Figure()
            
        @staticmethod
        def box(data_frame=None, y=None, **kwargs):
            return Figure()
    
    # Create the main Plotly mock
    class PlotlyMock:
        def __init__(self):
            self.graph_objects = GraphObjects()
            self.express = Express()
            self.offline = type('offline', (), {
                'plot': lambda fig, **kwargs: "<div>Plotly Plot (Fallback)</div>",
                'iplot': lambda fig, **kwargs: None
            })
    
    # Add to sys.modules
    plotly_mock = PlotlyMock()
    sys.modules['plotly'] = plotly_mock
    sys.modules['plotly.graph_objects'] = plotly_mock.graph_objects
    sys.modules['plotly.express'] = plotly_mock.express
    sys.modules['plotly.offline'] = plotly_mock.offline
    
    logger.warning("⚠️ Using simplified Plotly fallback. Install plotly for interactive visualizations.")
    
    return plotly_mock

def print_dependency_status():
    """Print the status of dependencies to the console."""
    all_installed, missing_required, missing_optional, version_issues = check_dependencies()
    
    if all_installed and not missing_optional and not version_issues:
        print("✅ All dependencies are installed correctly.")
        return
    
    if not all_installed:
        print("❌ Missing required dependencies:")
        for package in missing_required:
            print(f"   - {package}")
        print("\nInstall required dependencies with:")
        print(f"   {get_install_command(missing_required)}")
    
    if missing_optional:
        prefix = "Additionally, the" if not all_installed else "The"
        print(f"\n⚠️  {prefix} following optional dependencies are missing:")
        for package in missing_optional:
            print(f"   - {package}")
        print("\nInstall optional dependencies with:")
        print(f"   {get_install_command(missing_optional)}")
        
    if version_issues:
        prefix = "Additionally, there are" if (not all_installed or missing_optional) else "There are"
        print(f"\n⚠️  {prefix} version compatibility issues:")
        for issue in version_issues:
            pkg = issue['package']
            installed = issue['installed']
            required = issue['required']
            print(f"   - {pkg}: Installed {installed}, required {required}")
        print("\nUpgrade to resolve version issues:")
        problem_packages = [issue['package'] for issue in version_issues]
        print(f"   {get_install_command(problem_packages, upgrade=True)}")

# Check for fallbacks when this module is imported
fallbacks_loaded = fallback_dependencies()