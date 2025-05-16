"""
Script to update HTML files to use local image files instead of base64 encoded images.
This fixes the chart display issue in the static uncertainty report.

Usage:
    python static_uncertainty_renderer_fix.py [path_to_html_file]
"""

import os
import sys
import base64
import re

def fix_html_file(file_path):
    """
    Modifies the HTML file to use local image files instead of base64 encoded images.
    
    Args:
        file_path: Path to the HTML file to modify
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return False
        
    # Create the charts directory if it doesn't exist
    charts_dir = os.path.join(os.path.dirname(file_path), 'uncertainty_charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # Read the HTML file
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Find all base64 encoded images
    pattern = r'src="data:image/png;base64,([^"]+)"'
    matches = re.findall(pattern, html_content)
    
    if not matches:
        print("No base64 encoded images found in the file.")
        return False
        
    print(f"Found {len(matches)} base64 encoded images.")
    
    # Process each base64 encoded image
    for i, base64_data in enumerate(matches):
        # Create a unique filename
        image_filename = f'chart_{i+1}.png'
        image_path = os.path.join(charts_dir, image_filename)
        
        # Save the image to a file
        try:
            image_data = base64.b64decode(base64_data)
            with open(image_path, 'wb') as f:
                f.write(image_data)
            print(f"Saved image {i+1} to {image_path}")
            
            # Replace the base64 data with the file path in the HTML
            relative_path = f'./uncertainty_charts/{image_filename}'
            html_content = html_content.replace(f'src="data:image/png;base64,{base64_data}"', f'src="{relative_path}"')
        except Exception as e:
            print(f"Error processing image {i+1}: {str(e)}")
    
    # Write the modified HTML back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Updated {file_path} to use local image files.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python static_uncertainty_renderer_fix.py [path_to_html_file]")
        sys.exit(1)
        
    file_path = sys.argv[1]
    fix_html_file(file_path)