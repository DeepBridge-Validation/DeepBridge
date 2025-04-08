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
            templates directory in deepbridge/templates/reports.
        """
        if templates_dir is None:
            # Use default templates directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.templates_dir = os.path.join(base_dir, 'templates', 'reports')
        else:
            self.templates_dir = templates_dir
        
        # Set up Jinja2 environment with explicit UTF-8 encoding
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir, encoding='utf-8'),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Import numpy if available for handling numpy types
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.np = None
            
        # Set up paths for favicon and logo - use absolute paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.favicon_path = os.path.join(project_root, 'docs', 'assets', 'images', 'favicon.png')
        self.logo_path = os.path.join(project_root, 'docs', 'assets', 'images', 'logo.png')
    
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
            return ""

    def generate_robustness_report(self, results: Dict[str, Any], file_path: str, model_name: str = "Model") -> str:
        """
        Generate HTML report for robustness test results.
        
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
            template_path = os.path.join(self.templates_dir, 'robustness/report.html')
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template file does not exist: {template_path}")
                
            print(f"Template file exists: {template_path}")
            
            # Try reading the template directly to check encoding
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                    print(f"Successfully read template file (size: {len(template_content)} bytes)")
            except UnicodeDecodeError as e:
                print(f"Unicode decode error when reading template: {str(e)}")
                # Try to fix the encoding
                try:
                    import chardet
                    with open(template_path, 'rb') as f:
                        raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    print(f"Detected encoding: {detected}")
                    
                    # Convert to UTF-8
                    with open(template_path, 'w', encoding='utf-8') as f:
                        f.write(raw_data.decode(detected['encoding']))
                    print("Converted template to UTF-8")
                except ImportError:
                    print("chardet library not available for automatic encoding detection")
                except Exception as e:
                    print(f"Failed to fix encoding: {str(e)}")
            
            # Load the robustness report template
            template = self.jinja_env.get_template('robustness/report.html')
            
            # Create report data
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get base64 encoded favicon and logo
            favicon_base64 = self.get_base64_image(self.favicon_path)
            logo_base64 = self.get_base64_image(self.logo_path)
            
            # No need to import numpy here as we're using the class method
            
            # Transform results structure for template compatibility
            def transform_robustness_data(results_raw, local_model_name=model_name, local_timestamp=timestamp):
                print("Transforming robustness data structure...")
                
                # Debug input data
                print("Raw structure keys:", [k for k in results_raw.keys() if isinstance(results_raw, dict)])
                if isinstance(results_raw, dict) and 'raw' in results_raw:
                    print("Raw section exists with keys:", list(results_raw['raw'].keys()))
                    if 'by_level' in results_raw['raw']:
                        print("Raw by_level exists with levels:", list(results_raw['raw']['by_level'].keys()))
                
                # Convert results to a compatible format for the template
                if hasattr(results_raw, 'to_dict'):
                    report_data = results_raw.to_dict()
                    print("Used to_dict() method to convert results")
                else:
                    # Create a deep copy to avoid modifying the original
                    import copy
                    report_data = copy.deepcopy(results_raw)
                    print("Used deep copy to convert results")
                
                # Handle case where results are nested under 'primary_model' key
                if 'primary_model' in report_data:
                    print("Found 'primary_model' key, extracting data...")
                    primary_data = report_data['primary_model']
                    # Copy fields from primary_model to the top level
                    for key, value in primary_data.items():
                        if key not in report_data or key == 'raw' or key == 'quantile' or key == 'feature_importance':
                            report_data[key] = value
                    
                    # If raw, quantile, or feature_importance exists at the top level, don't overwrite
                    if 'raw' not in report_data and 'raw' in primary_data:
                        report_data['raw'] = primary_data['raw']
                    if 'quantile' not in report_data and 'quantile' in primary_data:
                        report_data['quantile'] = primary_data['quantile']
                    if 'feature_importance' not in report_data and 'feature_importance' in primary_data:
                        report_data['feature_importance'] = primary_data['feature_importance']
                        
                # Add metadata for display
                report_data['model_name'] = report_data.get('model_name', local_model_name)
                report_data['timestamp'] = report_data.get('timestamp', local_timestamp)
                report_data['model_type'] = report_data.get('model_type', "Unknown Model")
                
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
            
            # Convert all numpy types to Python native types
            report_data = self.convert_numpy_types(report_data)
            
            # Ensure JSON serialization will work by handling various types
            def json_serializer(obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                raise TypeError(f"Type not serializable: {type(obj)}")
                
            # Print the structure of report_data for debugging
            print("Report data structure after transformation:")
            for key in report_data:
                print(f"- {key}: {type(report_data[key])}")
                
            # Debug the JSON serialization
            try:
                json_data = json.dumps(report_data, default=json_serializer)
                print(f"JSON data serialized successfully (size: {len(json_data)} bytes)")
            except Exception as e:
                print(f"Error serializing to JSON: {str(e)}")
                # Try to find problematic keys
                for key in report_data:
                    try:
                        json.dumps({key: report_data[key]}, default=json_serializer)
                    except Exception as e:
                        print(f"  Problem serializing key '{key}': {str(e)}")
                        # If key is a dict, go deeper
                        if isinstance(report_data[key], dict):
                            for subkey in report_data[key]:
                                try:
                                    json.dumps({subkey: report_data[key][subkey]}, default=json_serializer)
                                except Exception as e:
                                    print(f"    Problem with subkey '{subkey}': {str(e)}")
                
            print("Rendering template...")
            
            # Render the template with favicon and logo base64 data
            rendered_html = template.render(
                report_data=json.dumps(report_data, default=json_serializer),
                model_name=model_name,
                timestamp=timestamp,
                current_year=datetime.datetime.now().year,
                favicon=favicon_base64,
                logo=logo_base64
            )
            
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
        Generate HTML report for uncertainty test results.
        
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
            template_path = os.path.join(self.templates_dir, 'uncertainty/report.html')
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template file does not exist: {template_path}")
                
            # Load the template
            template = self.jinja_env.get_template('uncertainty/report.html')
            
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
                report_data['model_type'] = report_data.get('model_type', "Unknown Model")
                
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
            
            # Print the structure of report_data for debugging
            print("Report data structure after transformation:")
            for key in report_data:
                print(f"- {key}: {type(report_data[key])}")
            
            # Get base64 encoded favicon and logo
            favicon_base64 = self.get_base64_image(self.favicon_path)
            logo_base64 = self.get_base64_image(self.logo_path)
            
            # Render the template with favicon and logo base64 data
            rendered_html = template.render(
                report_data=json.dumps(report_data, default=json_serializer),
                model_name=model_name,
                timestamp=timestamp,
                current_year=datetime.datetime.now().year,
                favicon=favicon_base64,
                logo=logo_base64
            )
            
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
        Generate HTML report for resilience test results.
        
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
            template_path = os.path.join(self.templates_dir, 'resilience/report.html')
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template file does not exist: {template_path}")
                
            # Load the template
            template = self.jinja_env.get_template('resilience/report.html')
            
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
                report_data['model_type'] = report_data.get('model_type', "Unknown Model")
                
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
            
            # Print the structure of report_data for debugging
            print("Report data structure after transformation:")
            for key in report_data:
                print(f"- {key}: {type(report_data[key])}")
            
            # Get base64 encoded favicon and logo
            favicon_base64 = self.get_base64_image(self.favicon_path)
            logo_base64 = self.get_base64_image(self.logo_path)
            
            # Render the template with favicon and logo base64 data
            rendered_html = template.render(
                report_data=json.dumps(report_data, default=json_serializer),
                model_name=model_name,
                timestamp=timestamp,
                current_year=datetime.datetime.now().year,
                favicon=favicon_base64,
                logo=logo_base64
            )
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(os.path.abspath(file_path))
            os.makedirs(output_dir, exist_ok=True)
            
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
        Generate HTML report for hyperparameter test results.
        
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
            template_path = os.path.join(self.templates_dir, 'hyperparameter/report.html')
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template file does not exist: {template_path}")
                
            # Load the template
            template = self.jinja_env.get_template('hyperparameter/report.html')
            
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
                report_data['model_type'] = report_data.get('model_type', "Unknown Model")
                
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
            
            # Render the template with favicon and logo base64 data
            rendered_html = template.render(
                report_data=json.dumps(report_data, default=json_serializer),
                model_name=model_name,
                timestamp=timestamp,
                current_year=datetime.datetime.now().year,
                favicon=favicon_base64,
                logo=logo_base64
            )
            
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