"""
Report generation module for experiment results.
"""

import os
import json
import math
import datetime
import base64
from typing import Dict, Any, Optional, Union

try:
    import jinja2
except ImportError:
    raise ImportError(
        "Jinja2 is required for HTML report generation. "
        "Please install it with: pip install jinja2"
    )

class ReportManager:
    """
    Handles the generation of HTML reports from experiment results.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the report manager.
        
        Parameters:
        -----------
        templates_dir : str, optional
            Directory containing report templates. If None, use the default
            templates directory in deepbridge/templates.
        """
        if templates_dir is None:
            # Use default templates directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.templates_dir = os.path.join(base_dir, 'templates')
        else:
            self.templates_dir = templates_dir
        
        # Set up Jinja2 environment with explicit UTF-8 encoding and trim_blocks/lstrip_blocks for better HTML output
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir, encoding='utf-8'),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Import numpy if available for handling numpy types
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.np = None
            
        # Set up paths for favicon and logo
        self.favicon_path = os.path.join(self.templates_dir, 'assets', 'images', 'favicon.png')
        self.logo_path = os.path.join(self.templates_dir, 'assets', 'images', 'logo.png')
        
        # Check if files exist, if not try fallback paths
        if not os.path.exists(self.favicon_path) or not os.path.exists(self.logo_path):
            print("Assets not found in expected path, trying alternative paths...")
            alt_favicon_path = os.path.join(self.templates_dir, 'reports', 'assets', 'images', 'favicon.png')
            alt_logo_path = os.path.join(self.templates_dir, 'reports', 'assets', 'images', 'logo.png')
            
            if os.path.exists(alt_favicon_path):
                self.favicon_path = alt_favicon_path
                print(f"Using alternate favicon path: {self.favicon_path}")
            
            if os.path.exists(alt_logo_path):
                self.logo_path = alt_logo_path
                print(f"Using alternate logo path: {self.logo_path}")
    
    def convert_numpy_types(self, data):
        """
        Safely convert numpy types to Python native types for JSON serialization.
        Compatible with NumPy 1.x and 2.x.
        
        Parameters:
        -----------
        data : Any
            Data that may contain numpy types
            
        Returns:
        --------
        Any : Data with numpy types converted to Python native types
        """
        np = self.np
        if np is not None:
            # Check for integer numpy types
            if hasattr(np, 'integer') and isinstance(data, np.integer):
                return int(data)
            # Check for specific integer types
            elif any(hasattr(np, t) and isinstance(data, getattr(np, t)) 
                     for t in ['int8', 'int16', 'int32', 'int64', 'intc', 'intp']):
                return int(data)
            
            # Check for float numpy types
            if hasattr(np, 'floating') and isinstance(data, np.floating):
                # Handle NaN and Inf values
                if np.isnan(data) or np.isinf(data):
                    return None
                return float(data)
            # Check for specific float types
            elif any(hasattr(np, t) and isinstance(data, getattr(np, t)) 
                     for t in ['float16', 'float32', 'float64']):
                if np.isnan(data) or np.isinf(data):
                    return None
                return float(data)
            
            # Check for numpy array
            elif isinstance(data, np.ndarray):
                # Convert array to list, replacing NaN/Inf with None
                if np.issubdtype(data.dtype, np.floating):
                    result = data.tolist()
                    if isinstance(result, list):
                        return [None if (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x for x in result]
                return data.tolist()
        
        # Handle other types
        if isinstance(data, dict):
            return {k: self.convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_numpy_types(item) for item in data]
        elif isinstance(data, (datetime.datetime, datetime.date)):
            return data.isoformat()
        elif isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
            return None
        else:
            return data
    
    def get_base64_image(self, image_path):
        """
        Convert image to base64 string.
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
            
        Returns:
        --------
        str : Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            
            # Fallback to embedded transparent 1x1 pixel PNG if original image not found
            transparent_1px_png = (
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00'
                b'\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\n'
                b'IDAT\x08\x99c\x00\x00\x00\x02\x00\x01\xe2\xb5\xc7\xb0\x00'
                b'\x00\x00\x00IEND\xaeB`\x82'
            )
            return base64.b64encode(transparent_1px_png).decode('utf-8')

    def generate_robustness_report(self, results: Dict[str, Any], file_path: str, model_name: str = "Model") -> str:
        """
        Generate HTML report for robustness test results using modular templates.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Robustness test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
            
        Returns:
        --------
        str : Path to the generated report
        """
        try:
            print(f"Generating robustness report to: {file_path}")
            print(f"Using templates directory: {self.templates_dir}")
            
            # Check if template exists
            template_path = os.path.join(self.templates_dir, 'report_types/robustness/index.html')
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template file does not exist: {template_path}")
                
            print(f"Template file exists: {template_path}")
            
            # Load the robustness report template with a custom loader to ensure standalone rendering
            loader = jinja2.FileSystemLoader(self.templates_dir, encoding='utf-8')
            env = jinja2.Environment(
                loader=loader,
                autoescape=jinja2.select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Use standalone template without inheritance
            template = env.get_template('report_types/robustness/index.html')
            
            # Create report data
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get base64 encoded favicon and logo
            favicon_base64 = self.get_base64_image(self.favicon_path)
            logo_base64 = self.get_base64_image(self.logo_path)
            
            # Transform results structure for template compatibility
            def transform_robustness_data(results_raw, local_model_name=model_name, local_timestamp=timestamp):
                print("Transforming robustness data structure...")
                
                # Convert results to a compatible format for the template
                if hasattr(results_raw, 'to_dict'):
                    report_data = results_raw.to_dict()
                    print("Used to_dict() method to convert results")
                else:
                    # Create a deep copy to avoid modifying the original
                    import copy
                    report_data = copy.deepcopy(results_raw)
                    print("Used deep copy to convert results")
                
                # Debug log to check for feature_importance in the data
                if 'feature_importance' in report_data:
                    print(f"Found feature_importance at top level with {len(report_data['feature_importance'])} features")
                if 'model_feature_importance' in report_data:
                    print(f"Found model_feature_importance at top level with {len(report_data['model_feature_importance'])} features")
                
                # Initialize empty feature importance dictionaries if not present
                if 'feature_importance' not in report_data:
                    report_data['feature_importance'] = {}
                    print("Initialized empty feature_importance dictionary")
                
                if 'model_feature_importance' not in report_data:
                    report_data['model_feature_importance'] = {}
                    print("Initialized empty model_feature_importance dictionary")
                
                # Handle case where results are nested under 'primary_model' key
                if 'primary_model' in report_data:
                    print("Found 'primary_model' key, extracting data...")
                    primary_data = report_data['primary_model']
                    
                    # Debug log for feature importance in primary_model
                    if 'feature_importance' in primary_data:
                        print(f"Found feature_importance in primary_model with {len(primary_data['feature_importance'])} features")
                    if 'model_feature_importance' in primary_data:
                        print(f"Found model_feature_importance in primary_model with {len(primary_data['model_feature_importance'])} features")
                    
                    # Copy fields from primary_model to the top level
                    for key, value in primary_data.items():
                        if key not in report_data or key == 'raw' or key == 'quantile':
                            report_data[key] = value
                    
                    # Always copy feature importance data to ensure it's available
                    if 'feature_importance' in primary_data:
                        print("Copying feature_importance from primary_model to top level")
                        report_data['feature_importance'] = primary_data['feature_importance']
                    
                    if 'model_feature_importance' in primary_data:
                        print("Copying model_feature_importance from primary_model to top level")
                        report_data['model_feature_importance'] = primary_data['model_feature_importance']
                    
                    # If raw, quantile exists at the top level, don't overwrite
                    if 'raw' not in report_data and 'raw' in primary_data:
                        report_data['raw'] = primary_data['raw']
                    if 'quantile' not in report_data and 'quantile' in primary_data:
                        report_data['quantile'] = primary_data['quantile']
                        
                # Add metadata for display
                report_data['model_name'] = report_data.get('model_name', local_model_name)
                report_data['timestamp'] = report_data.get('timestamp', local_timestamp)
                
                # Set model_type - use direct model_type if available
                if 'model_type' in report_data:
                    # Already has model_type, keep it
                    pass
                # Use model_type from primary_model if available
                elif 'primary_model' in report_data and 'model_type' in report_data['primary_model']:
                    report_data['model_type'] = report_data['primary_model']['model_type']
                # Use type from initial_results if available
                elif 'initial_results' in report_data and 'models' in report_data['initial_results'] and 'primary_model' in report_data['initial_results']['models'] and 'type' in report_data['initial_results']['models']['primary_model']:
                    report_data['model_type'] = report_data['initial_results']['models']['primary_model']['type']
                else:
                    report_data['model_type'] = "Unknown Model"
                
                # Check if we need to get feature importance data from nested structure
                if 'results' in report_data:
                    print("Checking for feature importance in nested results structure")
                    if 'robustness' in report_data['results']:
                        rob_results = report_data['results']['robustness']
                        print(f"Found robustness key with keys: {list(rob_results.keys())}")
                        
                        # Check in direct robustness object
                        if 'feature_importance' in rob_results and rob_results['feature_importance']:
                            print(f"Found feature_importance directly in results.robustness with {len(rob_results['feature_importance'])} features")
                            report_data['feature_importance'] = rob_results['feature_importance']
                        
                        if 'model_feature_importance' in rob_results and rob_results['model_feature_importance']:
                            print(f"Found model_feature_importance directly in results.robustness with {len(rob_results['model_feature_importance'])} features")
                            report_data['model_feature_importance'] = rob_results['model_feature_importance']
                        
                        # Check in nested results
                        if 'results' in rob_results:
                            nested_results = rob_results['results'] 
                            print(f"Found nested results with keys: {list(nested_results.keys())}")
                            
                            if 'primary_model' in nested_results:
                                primary_model = nested_results['primary_model']
                                print("Found primary_model in nested results.robustness.results")
                                
                                if 'feature_importance' in primary_model and primary_model['feature_importance']:
                                    print(f"Found feature_importance in nested results with {len(primary_model['feature_importance'])} features")
                                    report_data['feature_importance'] = primary_model['feature_importance']
                                    
                                if 'model_feature_importance' in primary_model and primary_model['model_feature_importance']:
                                    print(f"Found model_feature_importance in nested results with {len(primary_model['model_feature_importance'])} features")
                                    report_data['model_feature_importance'] = primary_model['model_feature_importance']
                                
                    # Check in experiment results structure (used by newer versions)
                    if 'robustness' in report_data['results'] and 'results' in report_data['results']['robustness']:
                        rob_nested = report_data['results']['robustness']['results']
                        if isinstance(rob_nested, dict) and 'primary_model' in rob_nested:
                            primary_model = rob_nested['primary_model']
                            print("Found primary_model in results.robustness.results")
                            
                            if 'feature_importance' in primary_model and primary_model['feature_importance']:
                                print(f"Found feature_importance in experiment results with {len(primary_model['feature_importance'])} features")
                                report_data['feature_importance'] = primary_model['feature_importance']
                                
                            if 'model_feature_importance' in primary_model and primary_model['model_feature_importance']:
                                print(f"Found model_feature_importance in experiment results with {len(primary_model['model_feature_importance'])} features")
                                report_data['model_feature_importance'] = primary_model['model_feature_importance']
                
                # Ensure we have a proper metrics structure
                if 'metrics' not in report_data:
                    print("Creating metrics structure...")
                    # Use any available metric or score information
                    report_data['metrics'] = {}
                    
                    # If we have 'metric' and 'base_score', use them
                    if 'metric' in report_data and 'base_score' in report_data:
                        metric_name = report_data.get('metric', 'score')
                        report_data['metrics'][metric_name] = report_data.get('base_score', 0)
                
                # Extract metric name if available
                if 'metric' in report_data:
                    report_data['metric'] = report_data['metric']
                else:
                    # Try to determine metric name from metrics dict
                    if 'metrics' in report_data and report_data['metrics']:
                        # Use first metric that's not base_score
                        for key in report_data['metrics']:
                            if key != 'base_score':
                                report_data['metric'] = key
                                break
                        if 'metric' not in report_data:
                            # Fall back to base_score
                            report_data['metric'] = 'base_score'
                    else:
                        report_data['metric'] = 'score'
                
                # Ensure we have base_score
                if 'base_score' not in report_data:
                    # Try to get from metrics
                    if 'metrics' in report_data and 'base_score' in report_data['metrics']:
                        report_data['base_score'] = report_data['metrics']['base_score']
                    elif 'metrics' in report_data and report_data['metrics'] and report_data['metric'] in report_data['metrics']:
                        report_data['base_score'] = report_data['metrics'][report_data['metric']]
                    else:
                        # Default
                        report_data['base_score'] = 0.0
                
                # Ensure robustness_score is calculated correctly
                if 'robustness_score' not in report_data:
                    report_data['robustness_score'] = float(1.0 - report_data.get('avg_overall_impact', 0))
                
                # Set impact values for display
                if 'avg_raw_impact' not in report_data and 'raw' in report_data and 'overall' in report_data['raw']:
                    report_data['avg_raw_impact'] = report_data['raw'].get('overall', {}).get('avg_impact', 0)
                
                if 'avg_quantile_impact' not in report_data and 'quantile' in report_data and 'overall' in report_data['quantile']:
                    report_data['avg_quantile_impact'] = report_data['quantile'].get('overall', {}).get('avg_impact', 0)
                
                # Set display-friendly alias properties
                report_data['raw_impact'] = report_data.get('avg_raw_impact', 0)
                report_data['quantile_impact'] = report_data.get('avg_quantile_impact', 0)
                
                # Handle iterations for test configuration display
                if 'n_iterations' not in report_data:
                    if 'raw' in report_data and 'by_level' in report_data['raw']:
                        for level in report_data['raw']['by_level']:
                            level_data = report_data['raw']['by_level'][level]
                            if 'runs' in level_data and 'all_features' in level_data['runs'] and level_data['runs']['all_features']:
                                run_data = level_data['runs']['all_features'][0]
                                if 'iterations' in run_data and 'n_iterations' in run_data['iterations']:
                                    report_data['n_iterations'] = run_data['iterations']['n_iterations']
                                    break
                    if 'n_iterations' not in report_data:
                        report_data['n_iterations'] = 3  # Default value
                
                # For display in the template
                report_data['iterations'] = report_data.get('n_iterations', 3)
                
                # Feature subset formatting
                if 'feature_subset' in report_data and report_data['feature_subset']:
                    if isinstance(report_data['feature_subset'], list):
                        # Already a list, keep as is
                        pass
                    elif isinstance(report_data['feature_subset'], str):
                        # Convert string to list
                        report_data['feature_subset'] = [report_data['feature_subset']]
                    else:
                        # Unknown format, set to empty list
                        report_data['feature_subset'] = []
                else:
                    report_data['feature_subset'] = []
                
                # Convert feature subset to display string
                if report_data['feature_subset']:
                    report_data['feature_subset_display'] = ", ".join(report_data['feature_subset'])
                else:
                    report_data['feature_subset_display'] = "All Features"
                
                # Process alternative models if present
                if 'alternative_models' in report_data:
                    print("Processing alternative models data...")
                    
                    # Initialize alternative models dict if needed
                    if not isinstance(report_data['alternative_models'], dict):
                        report_data['alternative_models'] = {}
                    
                    # Process each alternative model
                    for model_name, model_data in report_data['alternative_models'].items():
                        print(f"Processing alternative model: {model_name}")
                        
                        # Ensure metrics exist
                        if 'metrics' not in model_data:
                            model_data['metrics'] = {}
                            if 'base_score' in model_data:
                                model_data['metrics']['base_score'] = model_data['base_score']
                        
                        # Process robustness data
                        if 'raw' in model_data and isinstance(model_data['raw'], dict):
                            # Calculate average impact if not present
                            if 'avg_raw_impact' not in model_data and 'overall' in model_data['raw']:
                                model_data['avg_raw_impact'] = model_data['raw'].get('overall', {}).get('avg_impact', 0)
                        
                        if 'quantile' in model_data and isinstance(model_data['quantile'], dict):
                            # Calculate average impact if not present
                            if 'avg_quantile_impact' not in model_data and 'overall' in model_data['quantile']:
                                model_data['avg_quantile_impact'] = model_data['quantile'].get('overall', {}).get('avg_impact', 0)
                                
                        # Ensure robustness_score is calculated correctly
                        if 'robustness_score' not in model_data:
                            model_data['robustness_score'] = float(1.0 - model_data.get('avg_overall_impact', 0))
                            
                        # Update the model data in the report
                        report_data['alternative_models'][model_name] = model_data
                
                return report_data
            
            # Transform the data structure
            report_data = transform_robustness_data(results, model_name, timestamp)
            
            # Debug log to check the final state of feature importance data
            if 'feature_importance' in report_data:
                print(f"After transformation: feature_importance has {len(report_data['feature_importance'])} features")
            else:
                print("WARNING: No feature_importance found after transformation!")
                
            if 'model_feature_importance' in report_data:
                print(f"After transformation: model_feature_importance has {len(report_data['model_feature_importance'])} features")
            else:
                print("WARNING: No model_feature_importance found after transformation!")
            
            # Convert all numpy types to Python native types
            report_data = self.convert_numpy_types(report_data)
            
            # Ensure JSON serialization will work by handling various types
            def json_serializer(obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                # Handle NaN/inf values
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                raise TypeError(f"Type not serializable: {type(obj)}")
                
            # Create template context with structured data for components
            template_context = {
                # Complete report data for template access
                'report_data': report_data,
                # JSON string of report data for JavaScript processing
                'report_data_json': json.dumps(report_data, default=json_serializer),
                
                # Basic metadata
                'model_name': model_name,
                'timestamp': timestamp,
                'current_year': datetime.datetime.now().year,
                'favicon': favicon_base64,
                'logo': logo_base64,
                'block_title': f"Robustness Analysis: {model_name}",
                
                # Main metrics for direct access in templates
                'robustness_score': report_data.get('robustness_score', 0),
                'raw_impact': report_data.get('raw_impact', 0),
                'quantile_impact': report_data.get('quantile_impact', 0),
                'base_score': report_data.get('base_score', 0),
                'metric': report_data.get('metric', 'score'),
                'model_type': report_data.get('model_type', 'Unknown Model'),
                
                # Feature details
                'feature_subset': report_data.get('feature_subset', []),
                'feature_subset_display': report_data.get('feature_subset_display', 'All Features'),
                
                # Configuration details
                'iterations': report_data.get('iterations', 3),
                
                # For charts - we'll make sure these are extracted from the correct place
                'has_feature_importance': bool(report_data.get('feature_importance', {})),
                'has_model_feature_importance': bool(report_data.get('model_feature_importance', {})),
                
                # Pass feature importance data directly - ensure these are always defined
                'feature_importance': report_data.get('feature_importance', {}),
                'model_feature_importance': report_data.get('model_feature_importance', {}),
                
                # For component display logic
                'has_alternative_models': 'alternative_models' in report_data and bool(report_data['alternative_models'])
            }
            
            print("Rendering template...")
            
            # Render the template with detailed context
            rendered_html = template.render(**template_context)
            
            print(f"Template rendered successfully (size: {len(rendered_html)} bytes)")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(os.path.abspath(file_path))
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory created/verified: {output_dir}")
            
            # Write to file with explicit UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(rendered_html)
                
            print(f"Report saved successfully to: {file_path}")
            return file_path
        
        except Exception as e:
            print(f"Error generating robustness report: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error generating robustness report: {str(e)}")
    
    def generate_uncertainty_report(self, results: Dict[str, Any], file_path: str, model_name: str = "Model") -> str:
        """
        Generate HTML report for uncertainty test results using modular templates.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Uncertainty test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
            
        Returns:
        --------
        str : Path to the generated report
        """
        try:
            print(f"Generating uncertainty report to: {file_path}")
            
            # Check if template exists
            template_path = os.path.join(self.templates_dir, 'report_types/uncertainty/index.html')
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template file does not exist: {template_path}")
                
            # Load the template with a custom loader to ensure standalone rendering
            loader = jinja2.FileSystemLoader(self.templates_dir, encoding='utf-8')
            env = jinja2.Environment(
                loader=loader,
                autoescape=jinja2.select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Use standalone template without inheritance
            template = env.get_template('report_types/uncertainty/index.html')
            
            # Create report data
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Define a function to transform uncertainty data for the template
            def transform_uncertainty_data(results_raw, local_model_name=model_name, local_timestamp=timestamp):
                print("Transforming uncertainty data structure...")
                
                # Convert results to compatible format
                if hasattr(results_raw, 'to_dict'):
                    report_data = results_raw.to_dict()
                else:
                    # Create a deep copy to avoid modifying the original
                    import copy
                    report_data = copy.deepcopy(results_raw)
                
                # Handle case where results are nested under 'primary_model' key
                if 'primary_model' in report_data:
                    print("Found 'primary_model' key, extracting data...")
                    primary_data = report_data['primary_model']
                    # Copy fields from primary_model to the top level
                    for key, value in primary_data.items():
                        if key not in report_data or key == 'crqr':
                            report_data[key] = value
                
                # Add metadata for display
                report_data['model_name'] = report_data.get('model_name', local_model_name)
                report_data['timestamp'] = report_data.get('timestamp', local_timestamp)
                
                # Set model_type - use direct model_type if available
                if 'model_type' in report_data:
                    # Already has model_type, keep it
                    pass
                # Use model_type from primary_model if available
                elif 'primary_model' in report_data and 'model_type' in report_data['primary_model']:
                    report_data['model_type'] = report_data['primary_model']['model_type']
                # Use type from initial_results if available
                elif 'initial_results' in report_data and 'models' in report_data['initial_results'] and 'primary_model' in report_data['initial_results']['models'] and 'type' in report_data['initial_results']['models']['primary_model']:
                    report_data['model_type'] = report_data['initial_results']['models']['primary_model']['type']
                else:
                    report_data['model_type'] = "Unknown Model"
                
                # Ensure we have a proper metrics structure
                if 'metrics' not in report_data:
                    report_data['metrics'] = {}
                
                # Ensure metric name is available
                if 'metric' not in report_data:
                    report_data['metric'] = next(iter(report_data.get('metrics', {}).keys()), 'score')
                
                # Set uncertainty score if not present
                if 'uncertainty_score' not in report_data:
                    # Try to calculate from CRQR data
                    if 'crqr' in report_data and 'by_alpha' in report_data['crqr']:
                        # Average coverage quality (actual/expected ratio with penalty for over-coverage)
                        coverage_ratios = []
                        for alpha_key, alpha_data in report_data['crqr']['by_alpha'].items():
                            if 'overall_result' in alpha_data:
                                actual = alpha_data['overall_result'].get('coverage', 0)
                                expected = alpha_data['overall_result'].get('expected_coverage', 0)
                                if expected > 0:
                                    # Penalize over-coverage less than under-coverage
                                    ratio = min(actual / expected, 1.1) if actual > expected else actual / expected
                                    coverage_ratios.append(ratio)
                        
                        if coverage_ratios:
                            report_data['uncertainty_score'] = sum(coverage_ratios) / len(coverage_ratios)
                        else:
                            report_data['uncertainty_score'] = 0.5
                    else:
                        report_data['uncertainty_score'] = 0.5
                
                # Calculate average coverage and width if not present
                if 'avg_coverage' not in report_data and 'crqr' in report_data and 'by_alpha' in report_data['crqr']:
                    coverages = []
                    widths = []
                    
                    for alpha_key, alpha_data in report_data['crqr']['by_alpha'].items():
                        if 'overall_result' in alpha_data:
                            coverages.append(alpha_data['overall_result'].get('coverage', 0))
                            widths.append(alpha_data['overall_result'].get('mean_width', 0))
                    
                    if coverages:
                        report_data['avg_coverage'] = sum(coverages) / len(coverages)
                    else:
                        report_data['avg_coverage'] = 0
                        
                    if widths:
                        report_data['avg_width'] = sum(widths) / len(widths)
                    else:
                        report_data['avg_width'] = 0
                
                # Ensure we have alpha levels
                if 'alpha_levels' not in report_data and 'crqr' in report_data and 'by_alpha' in report_data['crqr']:
                    report_data['alpha_levels'] = list(map(float, report_data['crqr']['by_alpha'].keys()))
                
                # Set method if not present
                if 'method' not in report_data:
                    report_data['method'] = 'crqr'
                
                # Process alternative models if present
                if 'alternative_models' in report_data:
                    print("Processing alternative models data...")
                    
                    # Initialize alternative models dict if needed
                    if not isinstance(report_data['alternative_models'], dict):
                        report_data['alternative_models'] = {}
                    
                    # Process each alternative model
                    for model_name, model_data in report_data['alternative_models'].items():
                        print(f"Processing alternative model: {model_name}")
                        
                        # Ensure metrics exist
                        if 'metrics' not in model_data:
                            model_data['metrics'] = {}
                            
                        # Update the model data in the report
                        report_data['alternative_models'][model_name] = model_data
                
                return report_data
            
            # Transform the data structure
            report_data = transform_uncertainty_data(results, model_name, timestamp)
            
            # Convert numpy types to native Python types
            report_data = self.convert_numpy_types(report_data)
            
            # Ensure JSON serialization will work, handling NaN values
            def json_serializer(obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                # Handle NaN values which are not valid in JSON
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                raise TypeError(f"Type not serializable: {type(obj)}")
            
            # Get base64 encoded favicon and logo
            favicon_base64 = self.get_base64_image(self.favicon_path)
            logo_base64 = self.get_base64_image(self.logo_path)
            
            # Create template context with structured data for components
            template_context = {
                # Complete report data for template access
                'report_data': report_data,
                # JSON string of report data for JavaScript processing
                'report_data_json': json.dumps(report_data, default=json_serializer),
                
                # Basic metadata
                'model_name': model_name,
                'timestamp': timestamp,
                'current_year': datetime.datetime.now().year,
                'favicon': favicon_base64,
                'logo': logo_base64,
                'block_title': f"Uncertainty Analysis: {model_name}",
                
                # Main metrics for direct access in templates
                'uncertainty_score': report_data.get('uncertainty_score', None),
                'coverage_score': report_data.get('avg_coverage', None),
                'calibration_error': report_data.get('calibration_error', None),
                'sharpness': report_data.get('avg_width', None),
                'consistency': report_data.get('consistency', None),
                'avg_coverage': report_data.get('avg_coverage', None),
                'avg_width': report_data.get('avg_width', None),
                'method': report_data.get('method', 'crqr'),
                'metric': report_data.get('metric', 'score'),
                'model_type': report_data.get('model_type', 'Unknown Model'),
                'alpha_levels': report_data.get('alpha_levels', []),
                
                # Configuration details
                'has_alternative_models': 'alternative_models' in report_data and bool(report_data['alternative_models'])
            }
            
            # Render the template with detailed context
            rendered_html = template.render(**template_context)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(os.path.abspath(file_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # Write to file with explicit UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(rendered_html)
                
            print(f"Uncertainty report saved to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error generating uncertainty report: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error generating uncertainty report: {str(e)}")
    
    def generate_resilience_report(self, results: Dict[str, Any], file_path: str, model_name: str = "Model") -> str:
        """
        Generate HTML report for resilience test results using modular templates.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Resilience test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
            
        Returns:
        --------
        str : Path to the generated report
        """
        try:
            print(f"Generating resilience report to: {file_path}")
            
            # Check if template exists
            template_path = os.path.join(self.templates_dir, 'report_types/resilience/index.html')
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template file does not exist: {template_path}")
            
            print(f"Found template: {template_path}")
                
            # Load the template with a custom loader that doesn't look for base templates
            loader = jinja2.FileSystemLoader(self.templates_dir, encoding='utf-8')
            env = jinja2.Environment(
                loader=loader,
                autoescape=jinja2.select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Disable template inheritance to prevent looking for base.html
            template = env.get_template('report_types/resilience/index.html')
            
            # Create report data
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Define a function to transform resilience data for the template
            def transform_resilience_data(results_raw, local_model_name=model_name, local_timestamp=timestamp):
                print("Transforming resilience data structure...")
                
                # Convert results to compatible format
                if hasattr(results_raw, 'to_dict'):
                    report_data = results_raw.to_dict()
                else:
                    # Create a deep copy to avoid modifying the original
                    import copy
                    report_data = copy.deepcopy(results_raw)
                
                # Handle case where results are nested under 'primary_model' key
                if 'primary_model' in report_data:
                    print("Found 'primary_model' key, extracting data...")
                    primary_data = report_data['primary_model']
                    # Copy fields from primary_model to the top level
                    for key, value in primary_data.items():
                        if key not in report_data:
                            report_data[key] = value
                
                # Add metadata for display
                report_data['model_name'] = report_data.get('model_name', local_model_name)
                report_data['timestamp'] = report_data.get('timestamp', local_timestamp)
                
                # Set model_type - use direct model_type if available
                if 'model_type' in report_data:
                    # Already has model_type, keep it
                    pass
                # Use model_type from primary_model if available
                elif 'primary_model' in report_data and 'model_type' in report_data['primary_model']:
                    report_data['model_type'] = report_data['primary_model']['model_type']
                # Use type from initial_results if available
                elif 'initial_results' in report_data and 'models' in report_data['initial_results'] and 'primary_model' in report_data['initial_results']['models'] and 'type' in report_data['initial_results']['models']['primary_model']:
                    report_data['model_type'] = report_data['initial_results']['models']['primary_model']['type']
                else:
                    report_data['model_type'] = "Unknown Model"
                
                # Ensure we have a proper metrics structure
                if 'metrics' not in report_data:
                    report_data['metrics'] = {}
                
                # Ensure metric name is available
                if 'metric' not in report_data:
                    report_data['metric'] = next(iter(report_data.get('metrics', {}).keys()), 'score')
                
                # Make sure we have distribution_shift_results
                if 'distribution_shift_results' not in report_data:
                    # Try to extract from other fields if possible
                    if 'test_results' in report_data and isinstance(report_data['test_results'], list):
                        report_data['distribution_shift_results'] = report_data['test_results']
                    elif 'distribution_shift' in report_data and 'all_results' in report_data['distribution_shift']:
                        # Extract results from the nested structure
                        report_data['distribution_shift_results'] = report_data['distribution_shift']['all_results']
                    else:
                        # Create empty results
                        report_data['distribution_shift_results'] = []
                
                # Ensure we have distance metrics and alphas
                if 'distance_metrics' not in report_data:
                    distance_metrics = set()
                    for result in report_data.get('distribution_shift_results', []):
                        if 'distance_metric' in result:
                            distance_metrics.add(result['distance_metric'])
                    report_data['distance_metrics'] = list(distance_metrics) if distance_metrics else ['PSI', 'KS', 'WD1']
                
                if 'alphas' not in report_data:
                    alphas = set()
                    for result in report_data.get('distribution_shift_results', []):
                        if 'alpha' in result:
                            alphas.add(result['alpha'])
                    report_data['alphas'] = sorted(list(alphas)) if alphas else [0.1, 0.2, 0.3]
                
                # Calculate average performance gap if not present
                if 'avg_performance_gap' not in report_data:
                    performance_gaps = []
                    for result in report_data.get('distribution_shift_results', []):
                        if 'performance_gap' in result:
                            performance_gaps.append(result['performance_gap'])
                    
                    if performance_gaps:
                        report_data['avg_performance_gap'] = sum(performance_gaps) / len(performance_gaps)
                    elif 'resilience_score' in report_data:
                        # If we have resilience score but no average gap, calculate gap from score
                        report_data['avg_performance_gap'] = 1.0 - report_data['resilience_score']
                    else:
                        report_data['avg_performance_gap'] = 0.0
                
                # Process alternative models if present
                if 'alternative_models' in report_data:
                    print("Processing alternative models data...")
                    
                    # Initialize alternative models dict if needed
                    if not isinstance(report_data['alternative_models'], dict):
                        report_data['alternative_models'] = {}
                    
                    # Process each alternative model
                    for model_name, model_data in report_data['alternative_models'].items():
                        print(f"Processing alternative model: {model_name}")
                        
                        # Ensure metrics exist
                        if 'metrics' not in model_data:
                            model_data['metrics'] = {}
                            
                        # Update the model data in the report
                        report_data['alternative_models'][model_name] = model_data
                
                return report_data
            
            # Transform the data structure
            report_data = transform_resilience_data(results, model_name, timestamp)
            
            # Convert numpy types to native Python types
            report_data = self.convert_numpy_types(report_data)
            
            # Ensure JSON serialization will work, handling NaN values
            def json_serializer(obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                # Handle NaN values which are not valid in JSON
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                raise TypeError(f"Type not serializable: {type(obj)}")
            
            # Get base64 encoded favicon and logo
            favicon_base64 = self.get_base64_image(self.favicon_path)
            logo_base64 = self.get_base64_image(self.logo_path)
            
            # Initialize variables without default values
            # When these are not defined, the template will show "Data not available"
            avg_dist_shift = None
            max_gap = None
            most_affected_scenario = None
            
            if 'distribution_shift' in report_data and 'by_distance_metric' in report_data['distribution_shift']:
                # Calculate average distance across all metrics
                dist_values = []
                for dm, dm_data in report_data['distribution_shift']['by_distance_metric'].items():
                    for feature, val in dm_data.get('avg_feature_distances', {}).items():
                        dist_values.append(val)
                if dist_values:
                    avg_dist_shift = sum(dist_values) / len(dist_values)
                # If no values found, leave avg_dist_shift as None to show "Data not available"
            
            # Find the worst scenario (largest performance gap)
            if 'distribution_shift' in report_data and 'all_results' in report_data['distribution_shift']:
                all_results = report_data['distribution_shift']['all_results']
                if all_results:
                    # Find result with max performance gap
                    max_result = max(all_results, key=lambda x: x.get('performance_gap', 0) if isinstance(x.get('performance_gap', 0), (int, float)) else 0)
                    max_gap = max_result.get('performance_gap', 0)
                    # Create a descriptive scenario name
                    scenario_components = []
                    if 'alpha' in max_result:
                        scenario_components.append(f"{int(max_result['alpha'] * 100)}% shift")
                    if 'distance_metric' in max_result:
                        scenario_components.append(f"{max_result['distance_metric']} metric")
                    if scenario_components:
                        most_affected_scenario = " with ".join(scenario_components)
                    else:
                        most_affected_scenario = "Unspecified scenario"
            
            # Calculate outlier sensitivity based on available data
            outlier_sensitivity = None  # No default value - template will show "Data not available"
            if 'distribution_shift' in report_data and 'all_results' in report_data['distribution_shift']:
                sensitivity_values = []
                for result in report_data['distribution_shift']['all_results']:
                    if 'worst_metric' in result and 'remaining_metric' in result and 'alpha' in result:
                        # Sensitivity is how much performance changes per percentage shift
                        sensitivity = abs(result['performance_gap']) / (result['alpha'] * 100)
                        sensitivity_values.append(sensitivity)
                if sensitivity_values:
                    outlier_sensitivity = sum(sensitivity_values) / len(sensitivity_values)
                # If no values found, leave outlier_sensitivity as None to show "Data not available"
            
            # Get baseline and target dataset names only if available in test results
            baseline_dataset = None
            target_dataset = None
            if 'dataset_info' in report_data:
                if 'baseline_name' in report_data['dataset_info']:
                    baseline_dataset = report_data['dataset_info']['baseline_name']
                if 'target_name' in report_data['dataset_info']:
                    target_dataset = report_data['dataset_info']['target_name']
            
            # Extract shift scenarios from test results
            shift_scenarios = []
            if 'distribution_shift' in report_data and 'all_results' in report_data['distribution_shift']:
                for result in report_data['distribution_shift']['all_results']:
                    scenario = {
                        'name': result.get('name', f"Scenario {len(shift_scenarios) + 1}"),
                        'alpha': result.get('alpha', 0),
                        'metric': result.get('metric', 'unknown'),
                        'distance_metric': result.get('distance_metric', 'unknown'),
                        'performance_gap': result.get('performance_gap', 0),
                        'baseline_performance': result.get('baseline_performance', 0),
                        'target_performance': result.get('target_performance', 0),
                        'metrics': result.get('metrics', {})
                    }
                    shift_scenarios.append(scenario)
            
            # Extract sensitive features based on feature distances
            sensitive_features = []
            if 'distribution_shift' in report_data and 'by_distance_metric' in report_data['distribution_shift']:
                all_features = {}
                for dm, dm_data in report_data['distribution_shift']['by_distance_metric'].items():
                    for feature, value in dm_data.get('top_features', {}).items():
                        if feature not in all_features:
                            all_features[feature] = 0
                        all_features[feature] += value
                
                # Get top features across all distance metrics
                sensitive_features = [
                    {'name': feature, 'impact': value}
                    for feature, value in sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:5]
                ]
            
            # Performance gap is the same as avg_performance_gap
            performance_gap = report_data.get('avg_performance_gap', 0)
            
            # Create template context with structured data for components
            template_context = {
                # Complete report data for template access
                'report_data': report_data,
                # JSON string of report data for JavaScript processing
                'report_data_json': json.dumps(report_data, default=json_serializer),
                
                # Basic metadata
                'model_name': model_name,
                'timestamp': timestamp,
                'current_year': datetime.datetime.now().year,
                'favicon': favicon_base64,
                'logo': logo_base64,
                'block_title': f"Resilience Analysis: {model_name}",
                
                # Main metrics for direct access in templates
                'resilience_score': report_data.get('resilience_score', None),
                'avg_performance_gap': report_data.get('avg_performance_gap', None),
                'performance_gap': performance_gap,  # This is already None if not available
                'avg_dist_shift': avg_dist_shift,    # This is already None if not available
                'base_score': report_data.get('base_score', None),
                'base_performance': report_data.get('base_score', None),  # Alias for base_score
                'outlier_sensitivity': outlier_sensitivity,  # This is already None if not available
                'max_gap': max_gap,                  # This is already None if not available
                'most_affected_scenario': most_affected_scenario,  # This is already None if not available
                'distance_metrics': report_data.get('distance_metrics', []),
                'metric': report_data.get('metric', 'score'),
                'model_type': report_data.get('model_type', 'Unknown Model'),
                
                # Feature details
                'feature_subset': report_data.get('feature_subset', []),
                'feature_subset_display': report_data.get('feature_subset_display', 'All Features'),
                
                # Results data
                'distribution_shift_results': report_data.get('distribution_shift_results', []),
                'alphas': report_data.get('alphas', []),
                
                # Dataset information
                'baseline_dataset': baseline_dataset,
                'target_dataset': target_dataset,
                
                # Shift scenarios and sensitive features - note we don't need to double-encode as JSON
                # since these will be assigned directly to a JavaScript variable, not embedded in another JSON string
                'shift_scenarios': shift_scenarios,
                'sensitive_features': sensitive_features,
                
                # Module version information
                'resilience_module_version': report_data.get('module_version', '1.0'),
                
                # Model comparisons
                'has_alternative_models': 'alternative_models' in report_data and bool(report_data['alternative_models'])
            }
            
            # Render the template with detailed context
            rendered_html = template.render(**template_context)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(os.path.abspath(file_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # No need to copy static assets since we've embedded everything in the HTML
            
            # Write to file with explicit UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(rendered_html)
                
            print(f"Resilience report saved to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error generating resilience report: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error generating resilience report: {str(e)}")
    
    def generate_hyperparameter_report(self, results: Dict[str, Any], file_path: str, model_name: str = "Model") -> str:
        """
        Generate HTML report for hyperparameter test results using modular templates.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Hyperparameter test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
            
        Returns:
        --------
        str : Path to the generated report
        """
        try:
            print(f"Generating hyperparameter report to: {file_path}")
            
            # Check if template exists
            template_path = os.path.join(self.templates_dir, 'report_types/hyperparameter/index.html')
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template file does not exist: {template_path}")
                
            # Load the template with a custom loader to ensure standalone rendering
            loader = jinja2.FileSystemLoader(self.templates_dir, encoding='utf-8')
            env = jinja2.Environment(
                loader=loader,
                autoescape=jinja2.select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Use standalone template without inheritance
            template = env.get_template('report_types/hyperparameter/index.html')
            
            # Create report data
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Define a function to transform hyperparameter data for the template
            def transform_hyperparameter_data(results_raw, local_model_name=model_name, local_timestamp=timestamp):
                print("Transforming hyperparameter data structure...")
                
                # Convert results to compatible format
                if hasattr(results_raw, 'to_dict'):
                    report_data = results_raw.to_dict()
                else:
                    # Create a deep copy to avoid modifying the original
                    import copy
                    report_data = copy.deepcopy(results_raw)
                
                # Handle case where results are nested under 'primary_model' key
                if 'primary_model' in report_data:
                    print("Found 'primary_model' key, extracting data...")
                    primary_data = report_data['primary_model']
                    # Copy fields from primary_model to the top level
                    for key, value in primary_data.items():
                        if key not in report_data:
                            report_data[key] = value
                
                # Add metadata for display
                report_data['model_name'] = report_data.get('model_name', local_model_name)
                report_data['timestamp'] = report_data.get('timestamp', local_timestamp)
                
                # Set model_type - use direct model_type if available
                if 'model_type' in report_data:
                    # Already has model_type, keep it
                    pass
                # Use model_type from primary_model if available
                elif 'primary_model' in report_data and 'model_type' in report_data['primary_model']:
                    report_data['model_type'] = report_data['primary_model']['model_type']
                # Use type from initial_results if available
                elif 'initial_results' in report_data and 'models' in report_data['initial_results'] and 'primary_model' in report_data['initial_results']['models'] and 'type' in report_data['initial_results']['models']['primary_model']:
                    report_data['model_type'] = report_data['initial_results']['models']['primary_model']['type']
                else:
                    report_data['model_type'] = "Unknown Model"
                
                # Ensure we have a proper metrics structure
                if 'metrics' not in report_data:
                    report_data['metrics'] = {}
                
                # Make sure we have importance_results
                if 'importance_results' not in report_data:
                    # Try to extract from other fields if possible
                    if 'importance' in report_data and 'all_results' in report_data['importance']:
                        report_data['importance_results'] = report_data['importance']['all_results']
                    else:
                        # Create empty results
                        report_data['importance_results'] = []
                
                # Ensure we have importance_scores at top level
                if 'importance_scores' not in report_data:
                    # Try to get from importance section if it exists
                    if 'importance' in report_data and 'all_results' in report_data['importance'] and report_data['importance']['all_results']:
                        first_result = report_data['importance']['all_results'][0]
                        if 'normalized_importance' in first_result:
                            report_data['importance_scores'] = first_result['normalized_importance']
                        elif 'raw_importance_scores' in first_result:
                            report_data['importance_scores'] = first_result['raw_importance_scores']
                
                # Ensure we have tuning_order
                if 'tuning_order' not in report_data:
                    # Try to extract from results
                    if 'importance' in report_data and 'all_results' in report_data['importance'] and report_data['importance']['all_results']:
                        result = report_data['importance']['all_results'][0]
                        if 'tuning_order' in result:
                            report_data['tuning_order'] = result['tuning_order']
                        elif 'sorted_importance' in result:
                            # Use keys of sorted_importance as tuning_order
                            report_data['tuning_order'] = list(result['sorted_importance'].keys())
                
                return report_data
            
            # Transform the data structure
            report_data = transform_hyperparameter_data(results, model_name, timestamp)
            
            # Convert numpy types to native Python types
            report_data = self.convert_numpy_types(report_data)
            
            # Ensure JSON serialization will work, handling NaN values
            def json_serializer(obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                # Handle NaN values which are not valid in JSON
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                raise TypeError(f"Type not serializable: {type(obj)}")
            
            # Print the structure of report_data for debugging
            print("Report data structure after transformation:")
            for key in report_data:
                print(f"- {key}: {type(report_data[key])}")
            
            # Get base64 encoded favicon and logo
            favicon_base64 = self.get_base64_image(self.favicon_path)
            logo_base64 = self.get_base64_image(self.logo_path)
            
            # Create template context with structured data for components
            template_context = {
                # Complete report data for template access
                'report_data': report_data,
                # JSON string of report data for JavaScript processing
                'report_data_json': json.dumps(report_data, default=json_serializer),
                
                # Basic metadata
                'model_name': model_name,
                'timestamp': timestamp,
                'current_year': datetime.datetime.now().year,
                'favicon': favicon_base64,
                'logo': logo_base64,
                'block_title': f"Hyperparameter Analysis: {model_name}",
                
                # Main metrics for direct access in templates
                'model_type': report_data.get('model_type', 'Unknown Model'),
                'metric': report_data.get('metric', 'score'),
                'base_score': report_data.get('base_score', 0),
                
                # Hyperparameter specific data
                'importance_scores': report_data.get('importance_scores', {}),
                'tuning_order': report_data.get('tuning_order', []),
                'importance_results': report_data.get('importance_results', []),
                'optimization_results': report_data.get('optimization_results', []),
                
                # Feature details if available
                'feature_subset': report_data.get('feature_subset', []),
                'feature_subset_display': report_data.get('feature_subset_display', 'All Features'),
                
                # Model comparisons
                'has_alternative_models': 'alternative_models' in report_data and bool(report_data['alternative_models'])
            }
            
            # Render the template with detailed context
            rendered_html = template.render(**template_context)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(os.path.abspath(file_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # Write to file with explicit UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(rendered_html)
                
            print(f"Hyperparameter report saved to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error generating hyperparameter report: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error generating hyperparameter report: {str(e)}")
    
    def generate_report(self, test_type: str, results: Dict[str, Any], file_path: str, model_name: str = "Model") -> str:
        """
        Generate report for the specified test type.
        
        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', 'resilience', 'hyperparameter')
        results : Dict[str, Any]
            Test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
            
        Returns:
        --------
        str : Path to the generated report
        """
        test_type_lower = test_type.lower()
        if test_type_lower == 'robustness':
            return self.generate_robustness_report(results, file_path, model_name)
        elif test_type_lower == 'uncertainty':
            return self.generate_uncertainty_report(results, file_path, model_name)
        elif test_type_lower == 'resilience':
            return self.generate_resilience_report(results, file_path, model_name)
        elif test_type_lower == 'hyperparameter' or test_type_lower == 'hyperparameters':
            return self.generate_hyperparameter_report(results, file_path, model_name)
        else:
            raise ValueError(f"Unknown test type: {test_type}")