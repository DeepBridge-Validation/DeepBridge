# DeepBridge Report Generation

## Overview

DeepBridge's report generation system creates comprehensive, professional HTML reports for model validation results. The system supports both interactive reports with dynamic visualizations and static reports suitable for printing or PDF conversion.

## Report Architecture

### System Components

```
Report Generation System
├── Report Manager (Orchestrator)
├── Data Transformers (Test-specific)
├── Asset Management
│   ├── CSS Processor
│   ├── JavaScript Processor
│   └── Image Encoder
├── Template Engine (Jinja2)
└── Renderers
    ├── Interactive Renderers
    └── Static Renderers
```

### Key Features

- **Self-contained HTML**: All assets embedded inline
- **No external dependencies**: Works offline
- **Professional design**: Clean, modern interface
- **Interactive visualizations**: Plotly.js charts
- **Static alternatives**: Matplotlib/Seaborn charts
- **Responsive layout**: Mobile-friendly
- **Print optimization**: Clean printing styles

## Report Types

### 1. Robustness Reports

Visualizes model performance under input perturbations.

**Sections:**
- **Overview**: Summary metrics and key findings
- **Perturbation Analysis**: Performance by perturbation level
- **Feature Impact**: Individual feature sensitivities
- **Model Comparison**: Multi-model analysis

**Key Visualizations:**
- Line charts showing performance degradation
- Feature importance bar charts
- Box plots of performance distributions
- Heatmaps of feature interactions

### 2. Uncertainty Reports

Displays prediction interval calibration and coverage.

**Sections:**
- **Calibration Summary**: Coverage vs confidence levels
- **Interval Analysis**: Width vs coverage trade-offs
- **Feature Uncertainty**: Per-feature uncertainty contributions
- **Distribution Plots**: Prediction interval distributions

**Key Visualizations:**
- Coverage calibration curves
- Interval width distributions
- Feature uncertainty rankings
- Confidence band visualizations

### 3. Resilience Reports

Shows model behavior under distribution shifts.

**Sections:**
- **Drift Analysis**: Performance under different drift types
- **Feature Stability**: Feature-level drift impacts
- **Critical Points**: Drift thresholds and breaking points
- **Recovery Analysis**: Model adaptation capabilities

**Key Visualizations:**
- Performance gap charts
- Feature distance heatmaps
- Drift detection timelines
- Resilience score comparisons

### 4. Hyperparameter Reports

Presents parameter importance and tuning insights.

**Sections:**
- **Importance Ranking**: Parameter impact scores
- **Sensitivity Analysis**: Performance sensitivity curves
- **Interaction Effects**: Parameter interactions
- **Tuning Recommendations**: Optimization priorities

**Key Visualizations:**
- Importance score bar charts
- Sensitivity curves
- Interaction heatmaps
- Parallel coordinates plots

## Basic Usage

### Generate a Report

```python
from deepbridge.core.experiment import Experiment

# Create experiment
experiment = Experiment(
    name='my_analysis',
    dataset=dataset,
    models={'model_v1': model}
)

# Run tests
results = experiment.run_test('robustness', config='medium')

# Generate report
experiment.generate_report(
    test_type='robustness',
    output_dir='./reports',
    format='interactive'  # or 'static'
)
```

### Direct Report Generation

```python
from deepbridge.core.experiment.report import ReportManager

# Create report manager
manager = ReportManager(output_dir='./reports')

# Generate report from results
manager.generate_report(
    test_type='robustness',
    results=test_results,
    experiment_name='my_experiment',
    format='interactive'
)
```

## Advanced Features

### Custom Styling

Override default styles:

```python
# Custom CSS
custom_css = """
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
"""

manager.generate_report(
    test_type='robustness',
    results=results,
    custom_css=custom_css
)
```

### Additional Visualizations

Add custom charts to reports:

```python
import plotly.graph_objects as go

# Create custom chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data['x'],
    y=data['y'],
    mode='markers',
    name='Custom Analysis'
))

# Add to report
results['custom_charts'] = {
    'my_analysis': fig.to_json()
}

manager.generate_report(
    test_type='robustness',
    results=results,
    include_custom_charts=True
)
```

### Static Report Options

Configure static report generation:

```python
# Generate static report with saved charts
experiment.generate_report(
    test_type='uncertainty',
    output_dir='./reports',
    format='static',
    static_options={
        'save_charts': True,       # Save PNGs separately
        'chart_format': 'png',     # or 'svg', 'pdf'
        'dpi': 300,               # High resolution
        'figure_size': (10, 6),   # Width, height in inches
        'style': 'seaborn'        # Matplotlib style
    }
)
```

### Multi-Model Reports

Compare multiple models in a single report:

```python
# Run tests on multiple models
experiment = Experiment(
    name='model_comparison',
    dataset=dataset,
    models={
        'baseline': baseline_model,
        'improved_v1': model_v1,
        'improved_v2': model_v2
    }
)

results = experiment.run_test('robustness')

# Generate comparative report
experiment.generate_report(
    test_type='robustness',
    output_dir='./reports',
    comparison_mode=True
)
```

## Report Customization

### Template Structure

Reports use a hierarchical template structure:

```
templates/
├── base.html              # Base template
├── common/                # Shared components
│   ├── header.html
│   ├── footer.html
│   └── navigation.html
└── report_types/         # Test-specific templates
    ├── robustness/
    │   ├── index.html    # Main template
    │   ├── partials/     # Section templates
    │   ├── css/          # Styles
    │   └── js/           # Scripts
    └── [other types...]
```

### Creating Custom Templates

Extend existing templates:

```html
<!-- custom_robustness.html -->
{% raw %}
{% extends "report_types/robustness/index.html" %}

{% block custom_section %}
<div class="custom-analysis">
    <h2>Custom Analysis</h2>
    {{ custom_data | safe }}
</div>
{% endblock %}
{% endraw %}
```

Use custom template:

```python
manager.generate_report(
    test_type='robustness',
    results=results,
    template_override='custom_robustness.html'
)
```

### Custom Renderers

Create specialized renderers:

```python
from deepbridge.core.experiment.report.renderers import BaseRenderer

class CustomRenderer(BaseRenderer):
    def __init__(self, template_name='custom_template.html'):
        super().__init__(template_name)
    
    def transform_data(self, results):
        # Custom data transformation
        transformed = super().transform_data(results)
        transformed['custom_metric'] = self.calculate_custom_metric(results)
        return transformed
    
    def calculate_custom_metric(self, results):
        # Custom calculation
        return results['base_score'] * results['avg_impact']

# Register and use
from deepbridge.core.experiment.report import ReportManager

manager = ReportManager(output_dir='./reports')
manager.register_renderer('custom', CustomRenderer())
manager.generate_report(
    test_type='custom',
    results=results
)
```

## Report Components

### 1. Header Section

```html
<!-- Standard header components -->
- Logo and branding
- Report title and timestamp
- Experiment metadata
- Navigation menu
```

### 2. Summary Cards

Key metrics displayed as cards:

```python
summary_metrics = {
    'Overall Score': 0.85,
    'Critical Features': 3,
    'Test Coverage': '95%',
    'Confidence Level': 'High'
}
```

### 3. Interactive Charts

Plotly.js visualizations:

```javascript
// Chart configuration
const chartConfig = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d']
};

// Initialize charts
Plotly.newPlot('chart-div', data, layout, chartConfig);
```

### 4. Data Tables

Sortable, filterable tables:

```html
<table class="data-table sortable">
    <thead>
        <tr>
            <th data-sort="string">Feature</th>
            <th data-sort="float">Importance</th>
            <th data-sort="float">Impact</th>
        </tr>
    </thead>
    <tbody>
        <!-- Dynamic content -->
    </tbody>
</table>
```

### 5. Tab Navigation

Organized content sections:

```javascript
// Tab switching logic
document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', (e) => {
        const tabId = e.target.dataset.tab;
        showTab(tabId);
    });
});
```

## Best Practices

### 1. Performance Optimization

**Large Datasets**
```python
# Enable data sampling for reports
results_summary = {
    'full_results': results,  # Complete data
    'plot_data': sample_data(results, n=1000)  # Sampled for plotting
}
```

**Chart Optimization**
```python
# Limit data points in visualizations
chart_options = {
    'max_points': 1000,
    'aggregation': 'mean',  # or 'median', 'sample'
    'decimation': True
}
```

### 2. Accessibility

Ensure reports are accessible:

```python
# Enable accessibility features
report_options = {
    'high_contrast': True,
    'aria_labels': True,
    'keyboard_navigation': True,
    'screen_reader_friendly': True
}
```

### 3. Export Options

Support multiple export formats:

```python
# Generate multiple formats
formats = ['interactive', 'static', 'pdf', 'markdown']

for fmt in formats:
    experiment.generate_report(
        test_type='robustness',
        output_dir=f'./reports/{fmt}',
        format=fmt
    )
```

### 4. Report Archiving

Version and archive reports:

```python
import datetime

# Timestamped reports
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'./reports/archive/{timestamp}/'

experiment.generate_report(
    test_type='all',
    output_dir=output_path,
    include_metadata=True,
    compress=True  # Create ZIP archive
)
```

## Troubleshooting

### Common Issues

1. **Large HTML files**
   - Use static format with separate chart files
   - Enable data sampling
   - Compress output with gzip

2. **Chart rendering issues**
   - Check browser console for errors
   - Verify data format
   - Update Plotly.js version

3. **Template errors**
   - Validate Jinja2 syntax
   - Check variable names
   - Enable debug mode

### Debug Mode

```python
# Enable debug reporting
import logging
logging.basicConfig(level=logging.DEBUG)

manager = ReportManager(
    output_dir='./reports',
    debug=True
)

# Generate with debug info
manager.generate_report(
    test_type='robustness',
    results=results,
    include_debug_info=True,
    save_intermediate_data=True
)
```

### Report Validation

```python
from deepbridge.core.experiment.report.validators import ReportValidator

validator = ReportValidator()

# Validate report structure
issues = validator.validate_report('./reports/report.html')

if issues:
    print("Report issues found:")
    for issue in issues:
        print(f"- {issue}")
```

## Integration Examples

### With CI/CD

```yaml
# .github/workflows/validation.yml
name: Model Validation

on: [push]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run validation
        run: |
          python -m deepbridge validate \
            --dataset data.csv \
            --model model.pkl \
            --tests all \
            --config medium
      
      - name: Generate reports
        run: |
          python -m deepbridge report \
            --results ./results \
            --output ./reports \
            --format interactive
      
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: validation-reports
          path: ./reports
```

### With Documentation

Include reports in project documentation:

```markdown
# Model Validation Results

The latest validation reports are available:

- [Robustness Report](./reports/robustness/index.html)
- [Uncertainty Report](./reports/uncertainty/index.html)
- [Resilience Report](./reports/resilience/index.html)

## Key Findings

{% raw %}{{ include_summary('./reports/summary.json') }}{% endraw %}
```

### Email Integration

Send reports via email:

```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase

def send_report(report_path, recipient):
    msg = MIMEMultipart()
    msg['Subject'] = 'Model Validation Report'
    
    # Attach HTML report
    with open(report_path, 'r') as f:
        attachment = MIMEText(f.read(), 'html')
    
    msg.attach(attachment)
    
    # Send email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.send_message(msg)
```

## Future Features

### Planned Enhancements

1. **Interactive Dashboards**: Real-time monitoring dashboards
2. **Report Sharing**: Cloud-based report sharing
3. **Custom Branding**: Full white-label support
4. **Export Formats**: Word, PowerPoint, LaTeX
5. **Report Automation**: Scheduled report generation
6. **Collaborative Features**: Comments and annotations