#!/usr/bin/env python3
"""
Script to fix chart URLs in existing HTML reports by adding './' prefix.
This ensures charts are correctly referenced relative to the HTML file.
"""

import os
import re
import sys
import glob
import argparse
from pathlib import Path

def fix_chart_urls(html_file):
    """
    Find image URLs in the HTML file that point to chart subdirectories and fix them
    by adding a './' prefix if they don't already have one.
    """
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all img tags with src attributes pointing to chart subdirectories
    # This pattern captures chart URLs like: "uncertainty_charts/chart_name.png"
    pattern = r'src="([^"\.][^"]*?_charts/[^"]+)"'
    
    # Replace them with proper relative URLs
    fixed_content = re.sub(pattern, r'src="./\1"', content)
    
    # If there were changes, save the file
    if fixed_content != content:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Fix chart URLs in HTML reports')
    parser.add_argument('--path', default='.', help='Path to search for HTML files')
    parser.add_argument('--recursive', action='store_true', help='Search recursively')
    args = parser.parse_args()
    
    # Find all HTML files
    if args.recursive:
        search_pattern = os.path.join(args.path, '**', '*.html')
        html_files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(args.path, '*.html')
        html_files = glob.glob(search_pattern)
    
    print(f"Found {len(html_files)} HTML files to check")
    
    # Process each file
    fixed_count = 0
    for html_file in html_files:
        if fix_chart_urls(html_file):
            fixed_count += 1
            print(f"Fixed: {html_file}")
    
    print(f"Fixed {fixed_count}/{len(html_files)} HTML files")

if __name__ == '__main__':
    main()