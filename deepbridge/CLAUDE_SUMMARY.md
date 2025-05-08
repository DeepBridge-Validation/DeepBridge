# DeepBridge Framework Analysis

## Architecture Overview

DeepBridge is a comprehensive framework for evaluating, testing, and enhancing machine learning models with specific focus on:

1. **Robustness Testing**: Assessing model performance under perturbations
2. **Uncertainty Quantification**: Evaluating prediction confidence
3. **Resilience Testing**: Testing performance on shifted data
4. **Hyperparameter Optimization**: Finding optimal parameters
5. **Synthetic Data Generation**: Creating realistic synthetic datasets

## Core Components

### Experiment System
- `core/experiment/experiment.py`: Central experiment manager
- Test runners handle the execution of various test types
- Manager Factory pattern creates appropriate test managers
- Standardized results structure enables comparative reporting

### Report Generation
- `core/experiment/report/report_manager.py`: Coordinates report generation
- Template-based system using Jinja2 templates
- Specialized renderers for different test types (robustness, uncertainty, etc.)
- JSON data transformation for JavaScript visualization

### Robustness Testing
- Tests model sensitivity to input perturbations
- Evaluates performance degradation at multiple perturbation levels
- Identifies critical features that affect model robustness
- Compares alternative models for relative robustness assessment

### Report Visualization
- Interactive HTML reports with charts and tables
- Plotly.js for visualization
- Tabs for different analysis views (Overview, Details, Feature Importance)
- Client-side JavaScript for dynamic data visualization

## Report Generation Flow

1. Test execution in `experiment.py`
2. Results collection and standardization
3. `ReportManager` decides which renderer to use
4. Renderer transforms data into chart-compatible format
5. Template filled with sanitized JSON data
6. JavaScript in the template creates visualizations

## Key Insights

1. **Modular Design**: Clear separation between test execution and reporting
2. **Extensible Architecture**: Easy to add new test types and visualization components
3. **JavaScript Integration**: Complex data visualization handled by client-side JavaScript
4. **Data Transformation**: Raw test results transformed for visualization
5. **Error Handling**: Multiple levels of error detection and recovery

The HTML reporting system uses embedded JavaScript to create interactive visualizations, with fallback mechanisms to handle potential syntax errors or missing data.

# Synthetic Data Removal in Robustness Visualization

## Overview
This summary documents the changes made to remove synthetic data generation from the robustness visualization JavaScript and HTML files. The goal was to ensure that visualizations only display real data from actual robustness tests, showing clear error messages when data is not available rather than generating synthetic demonstrations.

## Files Modified

### JavaScript Files

#### 1. `templates/report_types/robustness/js/fix_boxplot.js`
- Removed the `generateSyntheticScores` function
- Removed code that generated synthetic data for models with insufficient scores
- Added clear error messages when no real data is available
- Improved error handling with informative UI elements

#### 2. `templates/report_types/robustness/js/model_chart_fix.js`
- Removed synthetic data generation for alternative models
- Added validation to only include models with real perturbation test data
- Implemented appropriate error messaging when models lack test data

#### 3. `templates/report_types/robustness/js/feature_importance_handler.js`
- Removed demo data generation for feature importance visualization
- Added null checks and clear error messages
- Updated the error message styling to match other visualizations

#### 4. `templates/report_types/robustness/js/charts/boxplot.js`
- Completely rewrote file to remove all synthetic data generation
- Removed code that created synthetic scores for models with insufficient data
- Removed code that generated synthetic alternative models
- Added clear error messaging directing users to run tests with iterations > 1

#### 5. `templates/report_types/robustness/js/charts/details.js`
- Removed the `createSampleData` function that generated synthetic perturbation data
- Removed code that created synthetic iteration data based on mean and standard deviation
- Updated error handling to return null instead of sample data
- Added console warnings to indicate when synthetic data will not be generated

#### 6. `templates/report_types/robustness/js/charts/modelComparison.js`
- Removed `generatePerturbationScores` function that created synthetic model performance data
- Updated chart rendering to handle null values appropriately
- Improved error messaging to guide users on how to generate real comparison data

#### 7. `templates/report_types/robustness/js/controllers/overview.js`
- Removed synthetic data generation for feature subset scores in the tables
- Removed synthetic data generation for worst subset scores in charts
- Removed synthetic data generation for model comparison
- Added appropriate error messages to guide users on getting real data

### HTML Files

#### 8. `templates/report_types/robustness/partials/boxplot.html`
- Removed demo model creation code that generated synthetic data
- Replaced synthetic data generation with error messages
- Updated to display a proper error message when real data is not available
- Removed the `generateScores` function that created synthetic data

#### 9. `templates/report_types/robustness/partials/features.html`
- Removed static example features data from the HTML
- Replaced with a loading message that will be replaced by real data or an error
- Updated feature count placeholders to use "-" instead of synthetic counts

#### 10. `templates/report_types/robustness/index.html`
- Removed all demo data definitions across multiple chart initializations:
  - Perturbation chart initialization
  - Worst-score chart initialization
  - Model comparison chart initialization
  - Model level details chart initialization
- Updated the `extractPerturbationData` function to return null instead of synthetic data
- Updated the `extractModelComparisonData` function to remove synthetic model generation
- Updated the `populateModelComparisonTable` function to show error message instead of synthetic data
- Updated the `populateRawPerturbationTable` function to remove synthetic data generation
- Updated the `extractFeatureImportanceData` function to return null instead of synthetic data
- Added clear error messages in Portuguese to guide users on how to generate real data

## Key Changes

1. **Removed All Synthetic Data Generation**
   - Eliminated all functions and code that generated fake/demo data
   - Removed fallbacks that created data when none was available

2. **Added Clear Error Messages**
   - Implemented user-friendly error messages in the UI
   - Added console warnings for debugging
   - Provided guidance on how to generate real data (e.g., run tests with iterations > 1)

3. **Improved Error Handling**
   - Updated functions to return null instead of synthetic data
   - Added null checks throughout the code
   - Ensured visualizations gracefully handle missing data

4. **Standardized Error Message Style**
   - Used consistent styling for error messages across all visualizations
   - Made error messages visually distinct to clearly indicate missing data
   - Added specific instructions on how to generate the required data

## Impact

These changes ensure that:

1. **Transparency**: Visualizations only show real data from actual tests
2. **Accuracy**: No misleading synthetic data is presented as if it were real
3. **Guidance**: Users receive clear instructions on how to generate the data they need
4. **Consistency**: All error messages follow a similar format and style
5. **Reliability**: The code gracefully handles missing or incomplete data
6. **Localization**: Error messages are in Portuguese to match the user's language preferences
7. **User Experience**: Clear error states help users understand when data is missing

These improvements make the robustness visualization more honest and transparent, encouraging users to run the appropriate tests to generate real data rather than relying on synthetic demonstrations. When real data is not available, users now see an informative error message that provides specific guidance on how to generate the required test data (e.g., running tests with n_iterations > 1), rather than seeing potentially misleading synthetic data.

## Common Error Message Pattern

The standard error message pattern used across all visualizations includes:

```html
<div style="padding: 40px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
    <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
    <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #d32f2f;">Dados não disponíveis</h3>
    <p style="color: #333; font-size: 16px; line-height: 1.4;">Não há dados disponíveis para visualização.</p>
    <p style="color: #333; margin-top: 20px; font-size: 14px;">Execute testes de robustez com iterações (n_iterations > 1) para visualizar dados reais.</p>
</div>
```

This standardized format ensures consistency across the entire report, providing a better user experience and clear, actionable guidance.