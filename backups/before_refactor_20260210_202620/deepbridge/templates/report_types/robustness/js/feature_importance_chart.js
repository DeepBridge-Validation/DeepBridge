/**
 * Standalone feature importance chart handler
 * This is a self-contained version that doesn't depend on other code
 */

// Define a standalone handler that works regardless of other code
window.StandaloneFeatureImportanceChart = {
    /**
     * Initialize the feature importance chart
     * @param {string} containerId - ID of container element for the chart
     */
    initialize: function(containerId) {
        console.log("Initializing standalone feature importance chart:", containerId);
        
        const container = document.getElementById(containerId);
        if (!container) {
            console.error("Chart container not found:", containerId);
            return;
        }
        
        // Extract data separately from other code
        const chartData = this.extractChartData();
        if (!chartData || !chartData.features || chartData.features.length === 0) {
            this.showNoDataMessage(container, "No feature importance data available");
            return;
        }
        
        // Create and render chart
        this.renderChart(container, chartData);
    },
    
    /**
     * Extract data for the feature importance chart
     * @returns {Object} Chart data object
     */
    extractChartData: function() {
        try {
            // Try to get feature importance data from various sources
            let featureImportance = {};
            let modelFeatureImportance = {};
            
            if (window.reportConfig && window.reportConfig.feature_importance) {
                featureImportance = window.reportConfig.feature_importance || {};
                modelFeatureImportance = window.reportConfig.model_feature_importance || {};
                console.log("Using feature importance from reportConfig");
            } 
            else if (window.reportData) {
                if (window.reportData.feature_importance) {
                    featureImportance = window.reportData.feature_importance;
                    modelFeatureImportance = window.reportData.model_feature_importance || {};
                    console.log("Using feature importance from reportData");
                }
            }
            
            if (Object.keys(featureImportance).length === 0) {
                console.warn("No feature importance data found");
                return null;
            }
            
            // Convert to arrays for plotting
            const featureArray = Object.keys(featureImportance).map(feature => {
                return {
                    name: feature,
                    importance: featureImportance[feature],
                    modelImportance: modelFeatureImportance[feature] || 0
                };
            });
            
            // Sort by absolute importance
            featureArray.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
            
            // Get top 15 features
            const topFeatures = featureArray.slice(0, 15);
            
            return {
                features: topFeatures.map(f => f.name),
                robustnessValues: topFeatures.map(f => f.importance),
                modelValues: topFeatures.map(f => f.modelImportance)
            };
        } catch (error) {
            console.error("Error extracting feature importance chart data:", error);
            return null;
        }
    },
    
    /**
     * Render the feature importance chart
     * @param {HTMLElement} container - Chart container element
     * @param {Object} chartData - Chart data object
     */
    renderChart: function(container, chartData) {
        try {
            // Verify Plotly is available
            if (typeof Plotly === 'undefined') {
                this.showErrorMessage(container, "Plotly library not available");
                return;
            }
            
            // Clear container
            container.innerHTML = '';
            
            // Create traces for the chart
            const traces = [
                {
                    x: chartData.robustnessValues,
                    y: chartData.features,
                    name: 'Robustness Impact',
                    type: 'bar',
                    orientation: 'h',
                    marker: {
                        color: '#8884d8'
                    }
                }
            ];
            
            // Add model importance if available
            if (chartData.modelValues && chartData.modelValues.length > 0) {
                traces.push({
                    x: chartData.modelValues,
                    y: chartData.features,
                    name: 'Model Importance',
                    type: 'bar',
                    orientation: 'h',
                    marker: {
                        color: '#82ca9d'
                    }
                });
            }
            
            // Chart layout
            const layout = {
                title: 'Feature Importance Comparison',
                xaxis: {
                    title: 'Importance Score'
                },
                yaxis: {
                    title: 'Feature',
                    automargin: true
                },
                barmode: 'group',
                margin: {
                    l: 150,
                    r: 20,
                    t: 40,
                    b: 40
                }
            };
            
            // Create the plot
            Plotly.newPlot(container, traces, layout, {
                responsive: true,
                displayModeBar: false
            });
            
            console.log("Feature importance chart rendered successfully");
        } catch (error) {
            console.error("Error rendering feature importance chart:", error);
            this.showErrorMessage(container, error.message);
        }
    },
    
    /**
     * Show no data message in chart container
     * @param {HTMLElement} container - Chart container element
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(container, message) {
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #f8f9fa; border-radius: 8px; margin: 20px auto; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px;">No Data Available</h3>
                <p style="color: #666; font-size: 16px; line-height: 1.4;">
                    ${message}
                </p>
            </div>`;
    },
    
    /**
     * Show error message in chart container
     * @param {HTMLElement} container - Chart container element
     * @param {string} errorMessage - Error message to display
     */
    showErrorMessage: function(container, errorMessage) {
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border: 1px solid #ffcccc; border-radius: 8px; margin: 20px auto; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #cc0000;">Chart Error</h3>
                <p style="color: #666; font-size: 16px; line-height: 1.4;">${errorMessage}</p>
                <div style="margin-top: 20px; padding: 15px; background-color: #f8f8f8; border-radius: 5px; text-align: left;">
                    <p style="font-weight: bold; margin-bottom: 10px;">Troubleshooting:</p>
                    <ul style="list-style-type: disc; padding-left: 20px; margin-bottom: 0;">
                        <li>Check if the Plotly library is loaded correctly</li>
                        <li>Verify that feature importance data is available in the report</li>
                        <li>Try reloading the page</li>
                    </ul>
                </div>
            </div>`;
    }
};

// Initialize charts when document is ready
document.addEventListener('DOMContentLoaded', function() {
    // NOTA: Não inicializamos mais o gráfico de comparação aqui, apenas o de feature_impact
    // para evitar a sobreposição com ImportanceComparisonHandler.js
    
    // Initialize feature importance chart if the container exists
    const featureContainer = document.getElementById('feature-importance-chart');
    if (featureContainer) {
        console.log("Found feature importance chart container - initializing");
        setTimeout(function() {
            window.StandaloneFeatureImportanceChart.initialize('feature-importance-chart');
        }, 500);
    }
    
    // Listen for tab changes to initialize charts when tabs become visible
    document.querySelectorAll('.tab-btn').forEach(button => {
        button.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            if (tabId === 'feature_impact') {
                console.log("Feature impact tab activated - initializing chart");
                setTimeout(function() {
                    if (document.getElementById('feature-importance-chart')) {
                        window.StandaloneFeatureImportanceChart.initialize('feature-importance-chart');
                    }
                }, 100);
            }
            // Removido o inicializador para importance_comparison para evitar conflito
        });
    });
});