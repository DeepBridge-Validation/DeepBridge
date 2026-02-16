/**
 * Patches for report bug fixes
 * Contains fixes for JavaScript errors in generated reports
 * Updated: May 7, 2024 - Added fix for "Details by Level" chart
 */

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log("Loading JavaScript patches for robustness report");
    
    // Fix variable scope issues with model chart initialization
    window.fixModelChartScope = function() {
        // If we have the ModelComparisonChartManager function that's causing issues
        if (typeof ModelComparisonChartManager !== 'undefined') {
            console.log("Fixing model comparison chart scope");
            
            // Override the problematic function to fix the scope issue
            const originalExtract = ModelComparisonChartManager.extractModelComparisonData;
            ModelComparisonChartManager.extractModelComparisonData = function() {
                try {
                    // Call the original function but wrap in try/catch
                    return originalExtract.call(this);
                } catch (error) {
                    console.error("Error in extractModelComparisonData, using fallback:", error);
                    // Return minimal valid data structure to prevent JS errors
                    return {
                        levels: [],
                        modelScores: {},
                        modelNames: {},
                        metricName: "unknown"
                    };
                }
            };
        }
    };
    
    // Fix feature importance data issues
    window.fixFeatureImportanceData = function() {
        // Add safeguards for feature importance data
        if (window.reportData && window.reportData.feature_importance) {
            console.log("Normalizing feature importance data");
            
            // Ensure all feature keys use consistent format
            const normalizedFeatures = {};
            const normalizedModelFeatures = {};
            
            // Helper function to normalize keys
            function normalizeKey(key) {
                return String(key).trim();
            }
            
            // Normalize feature importance data
            Object.keys(window.reportData.feature_importance).forEach(function(key) {
                const normalizedKey = normalizeKey(key);
                normalizedFeatures[normalizedKey] = window.reportData.feature_importance[key];
            });
            
            // Normalize model feature importance if available
            if (window.reportData.model_feature_importance) {
                Object.keys(window.reportData.model_feature_importance).forEach(function(key) {
                    const normalizedKey = normalizeKey(key);
                    normalizedModelFeatures[normalizedKey] = window.reportData.model_feature_importance[key];
                });
            }
            
            // Replace with normalized data
            window.reportData.feature_importance = normalizedFeatures;
            if (Object.keys(normalizedModelFeatures).length > 0) {
                window.reportData.model_feature_importance = normalizedModelFeatures;
            }
            
            console.log("Feature importance data normalized");
        }
    };
    
    // Fix for "Model Comparison > Details by Level" chart
    window.fixModelLevelDetailsChart = function() {
        console.log("Applying fix for 'Details by Level' chart");
        
        // Ensure ChartManager has the best extract function loaded
        if (window.ChartManager && typeof window.SafeModelComparisonChartManager !== 'undefined' && 
            typeof window.SafeModelComparisonChartManager.extractModelLevelDetailsData === 'function') {
            
            console.log("Replacing ChartManager.extractModelLevelDetailsData with safe version");
            // Use the improved safe version from model_chart_fix.js
            window.ChartManager.extractModelLevelDetailsData = window.SafeModelComparisonChartManager.extractModelLevelDetailsData;
            
            // Add event listener to reinitialize chart when "Details by Level" tab is clicked
            const detailsSelector = document.querySelector('#model_comparison_selector [data-chart-type="details"]');
            if (detailsSelector) {
                console.log("Adding click listener to 'Details by Level' selector");
                detailsSelector.addEventListener('click', function() {
                    // Reinitialize chart with a small delay to ensure DOM updates
                    setTimeout(function() {
                        const chartElement = document.getElementById('model-level-details-chart-plot');
                        if (chartElement && chartElement.innerHTML.includes('No Data Available')) {
                            console.log("'No Data Available' message detected, reinitializing chart");
                            if (typeof ChartManager.initializeModelLevelDetailsChart === 'function') {
                                ChartManager.initializeModelLevelDetailsChart('model-level-details-chart-plot');
                            }
                        }
                    }, 300);
                });
            }
            
            // Check if overview tab is active, and if so, initialize model details chart
            setTimeout(function() {
                const overviewTab = document.getElementById('overview');
                if (overviewTab && overviewTab.classList.contains('active')) {
                    const modelDetailChart = document.getElementById('model-level-details-chart-plot');
                    if (modelDetailChart && modelDetailChart.innerHTML.includes('No Data Available')) {
                        console.log("Overview tab active, initializing model details chart");
                        if (typeof ChartManager.initializeModelLevelDetailsChart === 'function') {
                            console.log("Forced initialization of model level details chart");
                            ChartManager.initializeModelLevelDetailsChart('model-level-details-chart-plot');
                        }
                    }
                }
            }, 1000); // Longer delay for initial page load
        } else {
            console.warn("Unable to fix model level details chart - required components not available");
        }
    };
    
    // Fix for initializing the details tab
    window.fixDetailsTabInitialization = function() {
        console.log("Setting up details tab initialization");
        
        // Listen for tab changes to initialize details tab content
        document.addEventListener('tabChange', function(e) {
            if (e.detail && e.detail.tabId === 'details') {
                console.log("Details tab activated");
                
                // Initialize the OverviewController if not already initialized
                if (typeof window.OverviewController !== 'undefined' && 
                    typeof window.OverviewController.init === 'function') {
                    console.log("Initializing OverviewController for details tab");
                    window.OverviewController.init();
                }
                
                // Initialize the radar chart
                if (typeof window.OverviewChartsManager !== 'undefined' && 
                    typeof window.OverviewChartsManager.initializeOverviewCharts === 'function') {
                    console.log("Initializing OverviewChartsManager radar chart");
                    window.OverviewChartsManager.initializeOverviewCharts();
                }
            }
        });
        
        // Add click listener to the details tab link
        const detailsTab = document.querySelector('.main-nav a[data-tab="details"]');
        if (detailsTab) {
            detailsTab.addEventListener('click', function() {
                console.log("Details tab clicked");
                // Handle any additional initialization for details tab
            });
        }
    };
    
    // Apply all patches
    try {
        window.fixModelChartScope();
        window.fixFeatureImportanceData();
        window.fixModelLevelDetailsChart();
        window.fixDetailsTabInitialization();
        console.log("All patches applied successfully");
    } catch (error) {
        console.error("Error applying patches:", error);
    }
});