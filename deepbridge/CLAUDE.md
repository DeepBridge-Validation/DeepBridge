# DeepBridge Report Generation Documentation

## Overview
In DeepBridge, when running experiments with `experiment.run_tests("quick")`, the result is an `ExperimentResult` object with a `save_report` method that can generate HTML reports from test results.

## Usage

### Generate Reports from an ExperimentResult Object

```python
# Run the experiment and get results
results = experiment.run_tests("quick")

# Generate a robustness report
results.save_report("robustness", "robustness_report.html")

# Generate an uncertainty report
results.save_report("uncertainty", "uncertainty_report.html")
```

### Generate Reports from a Dictionary

If you have a dictionary with test results (e.g., from saved data):

```python
from deepbridge.utils.report_from_results import generate_report_from_results

# Generate a standard report
generate_report_from_results(results_dict, "report.html", "My Experiment")

# Generate a specialized report
generate_report_from_results(results_dict, "robustness_report.html", "My Experiment", report_type="robustness")
```

## Troubleshooting
- If specialized reports fail to generate, check that the results dictionary has the correct structure.
- For robustness reports, ensure the 'robustness' key contains a properly structured results dictionary.
- For uncertainty reports, ensure the 'uncertainty' key contains a properly structured results dictionary.