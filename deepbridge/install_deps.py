"""
DeepBridge - Dependency Installer
This script helps install all dependencies needed for DeepBridge functionality.
"""

import subprocess
import sys
import importlib
from typing import List, Dict

def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed by attempting to import it."""
    try:
        if import_name is None:
            import_name = package_name
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name: str) -> bool:
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main installation function."""
    print("DeepBridge Dependency Installer")
    print("==============================")
    print("This script will check and install required dependencies for DeepBridge.")
    
    # Define required dependencies
    REQUIRED_PACKAGES = {
        'jinja2': 'jinja2',       # For template rendering
        'pandas': 'pandas',       # For data manipulation
        'numpy': 'numpy',         # For numerical operations
        'matplotlib': 'matplotlib',  # For basic plotting
        'plotly': 'plotly',       # For interactive visualizations
    }
    
    # Optional but recommended packages
    OPTIONAL_PACKAGES = {
        'scikit-learn': 'sklearn',  # For machine learning utilities
        'scipy': 'scipy',         # For scientific computing
    }
    
    # Check required packages
    print("\nChecking required packages:")
    missing_required = []
    for package_name, import_name in REQUIRED_PACKAGES.items():
        if check_package(import_name):
            print(f"✅ {package_name}: Installed")
        else:
            print(f"❌ {package_name}: Missing")
            missing_required.append(package_name)
    
    # Check optional packages
    print("\nChecking optional packages:")
    missing_optional = []
    for package_name, import_name in OPTIONAL_PACKAGES.items():
        if check_package(import_name):
            print(f"✅ {package_name}: Installed")
        else:
            print(f"⚠️ {package_name}: Missing")
            missing_optional.append(package_name)
    
    # Install missing packages
    if missing_required:
        print("\nInstalling required packages...")
        for package in missing_required:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
    else:
        print("\n✅ All required packages are already installed.")
    
    # Ask about optional packages
    if missing_optional:
        print("\nSome optional packages are missing. These are recommended but not required.")
        install_optional = input("Would you like to install optional packages? (y/n): ")
        
        if install_optional.lower() == 'y':
            print("\nInstalling optional packages...")
            for package in missing_optional:
                print(f"Installing {package}...")
                if install_package(package):
                    print(f"✅ {package} installed successfully")
                else:
                    print(f"❌ Failed to install {package}")
    
    print("\nDeepBridge dependency installation complete\!")
    print("You can now run the report generation examples and other DeepBridge functionality.")
    print("\nTo test report generation, try running the example:")
    print("python -m deepbridge.examples.report_generation_example")
    
if __name__ == "__main__":
    main()
