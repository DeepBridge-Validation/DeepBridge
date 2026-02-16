// Feature Importance Chart Manager
window.FeatureImportanceChartManager = {
    /**
     * Initialize feature importance chart
     * @param {string} elementId - Chart container ID
     */
    initializeFeatureImportanceChart: function(elementId) {
        console.log("Initializing feature importance chart");
        const chartElement = document.getElementById(elementId);
        if (!chartElement) {
            console.error("Chart element not found:", elementId);
            return;
        }
        
        try {
            // Extract data for chart
            const chartData = this.extractFeatureImportanceData();
            
            if (!chartData || !chartData.features || chartData.features.length === 0) {
                this.showNoDataMessage(chartElement, "No feature importance data available");
                return;
            }
            
            // Create plot data for robustness impact
            const robustnessTrace = {
                x: chartData.robustnessValues,
                y: chartData.features,
                type: 'bar',
                orientation: 'h',
                name: 'Robustness Impact',
                marker: {
                    color: 'rgb(136, 132, 216)'
                }
            };
            
            // Create plot data for model importance (if available)
            const plotData = [robustnessTrace];
            
            if (chartData.modelValues && chartData.modelValues.length > 0) {
                const modelTrace = {
                    x: chartData.modelValues,
                    y: chartData.features,
                    type: 'bar',
                    orientation: 'h',
                    name: 'Model Importance',
                    marker: {
                        color: 'rgb(130, 202, 157)'
                    }
                };
                plotData.push(modelTrace);
            }
            
            // Layout for the chart
            const layout = {
                title: 'Top Features by Importance',
                xaxis: {
                    title: 'Importance Score'
                },
                yaxis: {
                    title: 'Feature',
                    automargin: true
                },
                barmode: 'group',
                legend: {
                    orientation: 'h',
                    y: -0.2
                },
                hovermode: 'closest',
                margin: {
                    l: 150,
                    r: 20,
                    t: 60,
                    b: 100
                }
            };
            
            // Create the plot
            Plotly.newPlot(chartElement, plotData, layout, {responsive: true});
            
        } catch (error) {
            console.error("Error creating feature importance chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Extract data for feature importance chart from report data
     */
    extractFeatureImportanceData: function() {
        let features = [];
        let robustnessValues = [];
        let modelValues = [];
        
        // Extract data from report data
        if (window.reportData) {
            const featureImportance = window.reportData.feature_importance || {};
            const modelFeatureImportance = window.reportData.model_feature_importance || {};
            
            // Convert to array and sort by robustness impact (absolute value)
            const featureArray = Object.keys(featureImportance).map(feature => ({
                name: feature,
                robustnessImportance: featureImportance[feature],
                modelImportance: modelFeatureImportance[feature] || 0
            }));
            
            // Sort by absolute robustness importance value
            featureArray.sort((a, b) => Math.abs(b.robustnessImportance) - Math.abs(a.robustnessImportance));
            
            // Get top 10 features
            const topFeatures = featureArray.slice(0, 10);
            
            // Extract arrays for plotting
            features = topFeatures.map(f => f.name);
            robustnessValues = topFeatures.map(f => f.robustnessImportance);
            modelValues = topFeatures.map(f => f.modelImportance);
        }
        
        return {
            features,
            robustnessValues,
            modelValues
        };
    },
    
    /**
     * Initialize model vs robustness importance chart
     * @param {string} elementId - Chart container ID
     */
    initializeImportanceComparisonChart: function(elementId) {
        const chartElement = document.getElementById(elementId);
        if (!chartElement) {
            console.error("Chart element not found:", elementId);
            return;
        }
        
        try {
            // Extract data for chart
            const chartData = this.extractFeatureImportanceData();
            
            if (!chartData || !chartData.features || chartData.features.length === 0 || 
                !chartData.modelValues || chartData.modelValues.length === 0) {
                this.showNoDataMessage(chartElement, "No feature importance comparison data available");
                return;
            }
            
            // Create scatter plot data
            const importanceData = chartData.features.map((feature, i) => ({
                feature: feature,
                robustness: chartData.robustnessValues[i],
                model: chartData.modelValues[i]
            }));
            
            // Create scatter plot
            const scatterTrace = {
                x: importanceData.map(d => d.model),
                y: importanceData.map(d => d.robustness),
                mode: 'markers+text',
                type: 'scatter',
                text: importanceData.map(d => d.feature),
                textposition: 'top',
                marker: {
                    size: 12,
                    color: 'rgb(44, 160, 101)',
                    opacity: 0.7
                },
                hovertemplate: '<b>%{text}</b><br>Model Importance: %{x:.4f}<br>Robustness Impact: %{y:.4f}<extra></extra>'
            };
            
            // Layout for the chart
            const layout = {
                title: 'Model vs. Robustness Importance',
                xaxis: {
                    title: 'Model Importance'
                },
                yaxis: {
                    title: 'Robustness Impact'
                },
                hovermode: 'closest',
                margin: {
                    l: 60,
                    r: 20,
                    t: 60,
                    b: 60
                }
            };
            
            // Create the plot
            Plotly.newPlot(chartElement, [scatterTrace], layout, {responsive: true});
            
        } catch (error) {
            console.error("Error creating importance comparison chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Show no data message in chart container
     * @param {HTMLElement} element - Chart container element
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(element, message) {
        element.innerHTML = `
            <div class="data-unavailable">
                <div class="data-message">
                    <span class="message-icon">📊</span>
                    <h3>No Data Available</h3>
                    <p>${message}</p>
                </div>
            </div>`;
    },
    
    /**
     * Show error message in chart container
     * @param {HTMLElement} element - Chart container element
     * @param {string} errorMessage - Error message to display
     */
    showErrorMessage: function(element, errorMessage) {
        element.innerHTML = `
            <div style='padding: 20px; color: red;'>
                Error creating chart: ${errorMessage}
            </div>`;
    }
};