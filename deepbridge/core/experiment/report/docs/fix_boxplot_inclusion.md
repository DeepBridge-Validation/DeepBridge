# Fix Boxplot.js Inclusion

## Background

The `fix_boxplot.js` script is a critical component for robustness reports that ensures boxplot charts display the correct data. 

Previously, this script was included in the template via:

```html
{% include 'report_types/' + report_type + '/js/fix_boxplot.js' %}
```

However, this approach had two major issues:
1. Changes to the `fix_boxplot.js` file were not being reflected when generating new reports
2. The asset manager was not aware of this file being directly included, which could lead to inconsistencies

## Solution

We've implemented a solution that ensures `fix_boxplot.js` is properly included in the combined JavaScript content for robustness reports:

1. Modified `asset_processor.py` to directly include the contents of `fix_boxplot.js` in the combined JS content for robustness reports
2. Removed the template inclusion line from `report_types/robustness/index.html` 
3. Added documentation in `file_discovery.py` to explain the special handling
4. Added a unit test to verify that `fix_boxplot.js` is properly included

## How It Works

1. When the asset manager compiles JavaScript content for a robustness report:
   - It loads the generic JS content
   - It loads the test-specific JS content
   - **Special handling**: For robustness reports specifically, it directly looks for `fix_boxplot.js` and includes its contents
   - It combines all JS content and applies syntax fixes

2. The rendered report now gets `fix_boxplot.js` content as part of the combined JavaScript, ensuring:
   - Any changes to `fix_boxplot.js` are immediately reflected in new reports
   - The code is properly processed with syntax fixes
   - There's no duplicate inclusion of the same file

## Testing

The inclusion is verified with a unit test in `tests/test_fix_boxplot_inclusion.py` that:
1. Creates a temporary test environment with a sample `fix_boxplot.js`
2. Verifies that the script content is included in robustness reports
3. Verifies that the script content is NOT included in other report types

## Future Considerations

If we need similar handling for other specific JavaScript files, we should follow the same pattern - update the `get_combined_js_content` method to include them directly rather than using template inclusion.