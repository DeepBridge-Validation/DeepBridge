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
            // Clear any previous content to avoid double rendering
            chartElement.innerHTML = '';
            
            // Extract data for chart with improved validation
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
                    color: chartData.robustnessValues.map(val => val >= 0 ? 'rgb(136, 132, 216)' : 'rgb(216, 132, 132)')
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
            
            // Layout for the chart with improved readability
            const layout = {
                title: 'Top Features by Importance',
                xaxis: {
                    title: 'Importance Score',
                    zeroline: true,
                    zerolinecolor: '#888',
                    zerolinewidth: 1
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
                },
                plot_bgcolor: '#fafafa',
                paper_bgcolor: '#fff'
            };
            
            // Create the plot with improved error handling
            try {
                Plotly.newPlot(chartElement, plotData, layout, {
                    responsive: true,
                    displayModeBar: false,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                    displaylogo: false,
                    staticPlot: false,
                });
            } catch (plotlyError) {
                console.error("Plotly rendering error:", plotlyError);
                this.showErrorMessage(chartElement, "Chart rendering failed: " + plotlyError.message);
                return;
            }
            
            // Add resize event listener to properly redraw the chart when tab becomes visible
            window.addEventListener('resize', () => {
                if (chartElement.closest('.tab-content.active')) {
                    Plotly.relayout(chartElement, {
                        'autosize': true
                    });
                }
            });
            
        } catch (error) {
            console.error("Error creating feature importance chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Extract data for feature importance chart from report data with improved validation
     */
    extractFeatureImportanceData: function() {
        let features = [];
        let robustnessValues = [];
        let modelValues = [];
        
        // First validate we have the required data
        let featureImportance = {};
        let modelFeatureImportance = {};
        
        // Try multiple sources for feature importance data
        if (window.chartData && window.chartData.feature_importance) {
            // Try chart data first (preferred source)
            featureImportance = window.chartData.feature_importance;
            modelFeatureImportance = window.chartData.model_feature_importance || {};
            console.log("Using feature importance from chartData");
        } else if (window.reportConfig && window.reportConfig.feature_importance) {
            // Try reportConfig 
            featureImportance = window.reportConfig.feature_importance;
            modelFeatureImportance = window.reportConfig.model_feature_importance || {};
            console.log("Using feature importance from reportConfig");
        } else if (window.reportData) {
            if (window.reportData.feature_importance) {
                // Try direct reportData
                featureImportance = window.reportData.feature_importance;
                modelFeatureImportance = window.reportData.model_feature_importance || {};
                console.log("Using feature importance from reportData");
            } else if (window.reportData.chart_data_json) {
                // Try parsing chart_data_json
                try {
                    const chartData = JSON.parse(window.reportData.chart_data_json);
                    featureImportance = chartData.feature_importance || {};
                    modelFeatureImportance = chartData.model_feature_importance || {};
                    console.log("Using feature importance from parsed chart_data_json");
                } catch (e) {
                    console.error("Error parsing chart_data_json:", e);
                }
            }
        }
        
        // Check if we have any feature importance data
        if (Object.keys(featureImportance).length === 0) {
            console.warn("No feature importance data found in any data source");
            return null;
        }
        
        console.log("Found feature importance data for ", Object.keys(featureImportance).length, "features");
        
        try {
            // Convert to array and sort by robustness impact (absolute value)
            const featureArray = Object.keys(featureImportance).map(feature => ({
                name: feature,
                robustnessImportance: featureImportance[feature],
                modelImportance: modelFeatureImportance[feature] || 0
            }));
            
            // Sort by absolute robustness importance value
            featureArray.sort((a, b) => Math.abs(b.robustnessImportance) - Math.abs(a.robustnessImportance));
            
            // Get top 15 features (increased from 10)
            const topFeatures = featureArray.slice(0, 15);
            
            // Extract arrays for plotting
            features = topFeatures.map(f => f.name);
            robustnessValues = topFeatures.map(f => f.robustnessImportance);
            modelValues = topFeatures.map(f => f.modelImportance);
            
            // Log what we found
            console.log("Extracted top", features.length, "features for chart");
        } catch (error) {
            console.error("Error processing feature importance data:", error);
            return null;
        }
        
        return {
            features,
            robustnessValues,
            modelValues
        };
    },
    
    /**
     * Initialize model vs robustness importance chart with improved error handling
     * @param {string} elementId - Chart container ID
     */
    initializeImportanceComparisonChart: function(elementId) {
        const chartElement = document.getElementById(elementId);
        if (!chartElement) {
            console.error("Chart element not found:", elementId);
            return;
        }
        
        try {
            // Clear any previous content
            chartElement.innerHTML = '';
            
            // Extract data with validation
            const chartData = this.extractFeatureImportanceData();
            
            if (!chartData || !chartData.features || chartData.features.length === 0 || 
                !chartData.modelValues || chartData.modelValues.length === 0) {
                this.showNoDataMessage(chartElement, "No feature importance comparison data available");
                return;
            }
            
            // Create scatter plot data with color coding based on importance
            const importanceData = chartData.features.map((feature, i) => ({
                feature: feature,
                robustness: chartData.robustnessValues[i],
                model: chartData.modelValues[i],
                impact: Math.abs(chartData.robustnessValues[i]) // For sizing markers
            }));
            
            // Create scatter plot with color coding
            const scatterTrace = {
                x: importanceData.map(d => d.model),
                y: importanceData.map(d => d.robustness),
                mode: 'markers+text',
                type: 'scatter',
                text: importanceData.map(d => d.feature),
                textposition: 'top',
                marker: {
                    size: importanceData.map(d => Math.min(Math.max(d.impact * 500, 8), 20)), // Size by importance
                    color: importanceData.map(d => d.robustness >= 0 ? 'rgb(44, 160, 101)' : 'rgb(215, 48, 39)'),
                    opacity: 0.8,
                    line: {
                        width: 1,
                        color: '#333'
                    }
                },
                hovertemplate: '<b>%{text}</b><br>Model Importance: %{x:.4f}<br>Robustness Impact: %{y:.4f}<extra></extra>'
            };
            
            // Layout for the chart
            const layout = {
                title: 'Model vs. Robustness Importance',
                xaxis: {
                    title: 'Model Importance',
                    zeroline: true
                },
                yaxis: {
                    title: 'Robustness Impact',
                    zeroline: true
                },
                hovermode: 'closest',
                margin: {
                    l: 60,
                    r: 20,
                    t: 60,
                    b: 60
                },
                // Add reference lines at x=0 and y=0
                shapes: [
                    {
                        type: 'line',
                        x0: 0,
                        y0: 0,
                        x1: 0,
                        y1: 1,
                        yref: 'paper',
                        line: {
                            color: 'grey',
                            width: 1,
                            dash: 'dot'
                        }
                    },
                    {
                        type: 'line',
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 0,
                        xref: 'paper',
                        line: {
                            color: 'grey',
                            width: 1,
                            dash: 'dot'
                        }
                    }
                ],
                annotations: [
                    {
                        xref: 'paper',
                        yref: 'paper',
                        x: 0.01,
                        y: 0.99,
                        text: 'Negative impact on model robustness',
                        showarrow: false,
                        font: {
                            size: 11,
                            color: 'gray'
                        }
                    },
                    {
                        xref: 'paper',
                        yref: 'paper',
                        x: 0.99,
                        y: 0.99,
                        text: 'Positive impact on model robustness',
                        showarrow: false,
                        font: {
                            size: 11,
                            color: 'gray'
                        },
                        align: 'right'
                    }
                ]
            };
            
            // Create the plot with error handling
            try {
                Plotly.newPlot(chartElement, [scatterTrace], layout, {
                    responsive: true,
                    displayModeBar: false
                });
            } catch (plotlyError) {
                console.error("Plotly rendering error:", plotlyError);
                this.showErrorMessage(chartElement, "Chart rendering failed: " + plotlyError.message);
                return;
            }
            
            // Add resize event listener
            window.addEventListener('resize', () => {
                if (chartElement.closest('.tab-content.active')) {
                    Plotly.relayout(chartElement, {
                        'autosize': true
                    });
                }
            });
            
        } catch (error) {
            console.error("Error creating importance comparison chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Show no data message in chart container with improved styling
     * @param {HTMLElement} element - Chart container element
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(element, message) {
        element.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #f8f9fa; border-radius: 8px; margin: 20px auto; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px;">No Data Available</h3>
                <p style="color: #666; font-size: 16px; line-height: 1.4;">
                    ${message}
                </p>
            </div>`;
    },
    
    /**
     * Show error message in chart container with improved styling
     * @param {HTMLElement} element - Chart container element
     * @param {string} errorMessage - Error message to display
     */
    showErrorMessage: function(element, errorMessage) {
        element.innerHTML = `
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