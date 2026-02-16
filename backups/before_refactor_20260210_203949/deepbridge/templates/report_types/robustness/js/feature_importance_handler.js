// Feature Importance Handler
// This is a standalone script that handles the feature importance visualization

(function() {
    // Create the handler
    window.FeatureImportanceHandler = {
        /**
         * Initialize the feature importance visualization
         */
        initialize: function() {
            console.log("Initializing feature importance handler");
            
            try {
                // Initialize the feature importance chart
                this.initializeFeatureImportanceChart();
                
                // Initialize the feature importance table
                this.initializeFeatureImportanceTable();
                
                // Initialize event listeners
                this.initializeEventListeners();
                
                console.log("Feature importance handler initialized");
            } catch (error) {
                console.error("Error initializing feature importance handler:", error);
            }
        },
        
        /**
         * Initialize the feature importance chart using Plotly
         */
        initializeFeatureImportanceChart: function() {
            try {
                const chartContainer = document.getElementById('feature-importance-chart');
                if (!chartContainer) return;
                
                // Get feature importance data - no synthetic data will be generated
                const chartData = this.extractFeatureImportanceData();
                if (!chartData || !chartData.features || chartData.features.length === 0) {
                    this.showNoDataMessage(chartContainer, "Dados de importância de características não disponíveis. Execute testes com cálculo de importância de características habilitado.");
                    return;
                }
                
                // Create chart
                this.renderFeatureImportanceChart(chartContainer, chartData);
            } catch (error) {
                console.error("Error initializing feature importance chart:", error);
                const container = document.getElementById('feature-importance-chart');
                if (container) {
                    this.showErrorMessage(container, "Error initializing chart: " + error.message);
                }
            }
        },
        
        /**
         * Extract feature importance data from report data
         * @returns {Object} Feature importance data
         */
        extractFeatureImportanceData: function() {
            try {
                // Try to get data from various sources
                let featureImportance = {};
                let modelFeatureImportance = {};
                
                if (window.reportConfig && window.reportConfig.feature_importance) {
                    featureImportance = window.reportConfig.feature_importance || {};
                    modelFeatureImportance = window.reportConfig.model_feature_importance || {};
                } 
                else if (window.reportData) {
                    if (window.reportData.feature_importance) {
                        featureImportance = window.reportData.feature_importance;
                        modelFeatureImportance = window.reportData.model_feature_importance || {};
                    }
                }
                
                // No synthetic data generation - return null if no data available
                if (Object.keys(featureImportance).length === 0) {
                    console.warn("No feature importance data available - returning null");
                    return null;
                }
                
                // Convert to arrays for plotting
                const featureArray = [];
                for (const feature in featureImportance) {
                    featureArray.push({
                        name: feature,
                        importance: featureImportance[feature],
                        modelImportance: modelFeatureImportance[feature] || 0
                    });
                }
                
                // Sort by absolute importance value
                featureArray.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
                
                // Get top features
                const topFeatures = featureArray.slice(0, 15);
                
                return {
                    features: topFeatures.map(f => f.name),
                    robustnessValues: topFeatures.map(f => f.importance),
                    modelValues: topFeatures.map(f => f.modelImportance)
                };
            } catch (error) {
                console.error("Error extracting feature importance data:", error);
                return null;
            }
        },
        
        /**
         * Render feature importance chart using Plotly
         * @param {HTMLElement} container - Chart container element
         * @param {Object} chartData - Chart data object
         */
        renderFeatureImportanceChart: function(container, chartData) {
            try {
                // Verify Plotly is available
                if (typeof Plotly === 'undefined') {
                    this.showErrorMessage(container, "Plotly library not available");
                    return;
                }
                
                // Clear container
                container.innerHTML = '';
                
                // Create traces for chart
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
                if (chartData.modelValues && chartData.modelValues.some(v => v !== 0)) {
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
                
                // Layout
                const layout = {
                    title: 'Feature Importance',
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
                
                // Create plot
                Plotly.newPlot(container, traces, layout, {
                    responsive: true,
                    displayModeBar: false
                });
                
                console.log("Feature importance chart rendered successfully");
            } catch (error) {
                console.error("Error rendering feature importance chart:", error);
                this.showErrorMessage(container, "Error rendering chart: " + error.message);
            }
        },
        
        /**
         * Initialize the feature importance table
         */
        initializeFeatureImportanceTable: function() {
            try {
                // Get the table body
                const tableBody = document.getElementById('feature-impact-data');
                if (!tableBody) return;
                
                // Get the feature subset if available
                let featureSubset = [];
                if (window.reportConfig && window.reportConfig.feature_subset) {
                    featureSubset = window.reportConfig.feature_subset;
                } else if (window.reportData && window.reportData.feature_subset) {
                    featureSubset = window.reportData.feature_subset;
                }
                
                // Get feature importance data
                let featureImportance = {};
                let modelFeatureImportance = {};
                
                if (window.reportConfig && window.reportConfig.feature_importance) {
                    featureImportance = window.reportConfig.feature_importance || {};
                    modelFeatureImportance = window.reportConfig.model_feature_importance || {};
                } 
                else if (window.reportData) {
                    if (window.reportData.feature_importance) {
                        featureImportance = window.reportData.feature_importance;
                        modelFeatureImportance = window.reportData.model_feature_importance || {};
                    }
                }
                
                // If we have real data, use it to populate the table
                if (Object.keys(featureImportance).length > 0) {
                    // Convert feature data to array for sorting
                    const featureArray = [];
                    for (const feature in featureImportance) {
                        featureArray.push({
                            name: feature,
                            impact: featureImportance[feature],
                            importance: modelFeatureImportance[feature] || 0,
                            inSubset: featureSubset.includes(feature)
                        });
                    }
                    
                    // Sort by absolute importance value (default sort)
                    featureArray.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
                    
                    // Clear existing rows (keep them if there's no real data)
                    tableBody.innerHTML = '';
                    
                    // Add rows for each feature
                    featureArray.forEach(feature => {
                        const row = document.createElement('tr');
                        row.className = feature.inSubset ? 'feature-subset-row' : '';
                        
                        // Feature name
                        const nameCell = document.createElement('td');
                        nameCell.textContent = feature.name;
                        row.appendChild(nameCell);
                        
                        // Robustness impact
                        const impactCell = document.createElement('td');
                        impactCell.textContent = feature.impact.toFixed(4);
                        row.appendChild(impactCell);
                        
                        // Model importance
                        const importanceCell = document.createElement('td');
                        importanceCell.textContent = feature.importance.toFixed(4);
                        row.appendChild(importanceCell);
                        
                        // Feature subset status
                        const subsetCell = document.createElement('td');
                        const subsetBadge = document.createElement('span');
                        subsetBadge.className = `subset-badge ${feature.inSubset ? 'included' : 'excluded'}`;
                        subsetBadge.textContent = feature.inSubset ? 'Included' : 'Excluded';
                        subsetCell.appendChild(subsetBadge);
                        row.appendChild(subsetCell);
                        
                        tableBody.appendChild(row);
                    });
                    
                    // Update feature counts
                    const totalFeaturesCount = document.getElementById('total-features-count');
                    if (totalFeaturesCount) {
                        totalFeaturesCount.textContent = featureArray.length;
                    }
                    
                    const subsetFeaturesCount = document.getElementById('subset-features-count');
                    if (subsetFeaturesCount) {
                        subsetFeaturesCount.textContent = featureArray.filter(f => f.inSubset).length;
                    }
                }
                
                console.log("Feature importance table initialized");
            } catch (error) {
                console.error("Error initializing feature importance table:", error);
            }
        },
        
        /**
         * Initialize event listeners for the feature importance UI
         */
        initializeEventListeners: function() {
            try {
                // Search input
                const searchInput = document.getElementById('feature-search');
                if (searchInput) {
                    searchInput.addEventListener('input', () => {
                        this.filterFeatures(searchInput.value);
                    });
                }
                
                // Show subset only toggle
                const subsetToggle = document.getElementById('show-subset-only');
                if (subsetToggle) {
                    subsetToggle.addEventListener('change', () => {
                        this.toggleSubsetOnly(subsetToggle.checked);
                    });
                }
                
                // Sortable column headers
                const sortableHeaders = document.querySelectorAll('.feature-importance-table th.sortable');
                sortableHeaders.forEach(header => {
                    header.addEventListener('click', () => {
                        this.sortFeatureTable(header.dataset.sort);
                    });
                });
                
                console.log("Feature importance event listeners initialized");
            } catch (error) {
                console.error("Error initializing feature importance event listeners:", error);
            }
        },
        
        /**
         * Filter features based on search text
         * @param {string} searchText - Search text
         */
        filterFeatures: function(searchText) {
            try {
                const rows = document.querySelectorAll('#feature-impact-data tr');
                const searchLower = searchText.toLowerCase();
                
                rows.forEach(row => {
                    const featureName = row.querySelector('td').textContent.toLowerCase();
                    
                    if (featureName.includes(searchLower)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
            } catch (error) {
                console.error("Error filtering features:", error);
            }
        },
        
        /**
         * Toggle showing only subset features
         * @param {boolean} showSubsetOnly - Whether to show only subset features
         */
        toggleSubsetOnly: function(showSubsetOnly) {
            try {
                const rows = document.querySelectorAll('#feature-impact-data tr');
                
                rows.forEach(row => {
                    if (showSubsetOnly && !row.classList.contains('feature-subset-row')) {
                        row.style.display = 'none';
                    } else {
                        row.style.display = '';
                    }
                });
            } catch (error) {
                console.error("Error toggling subset only:", error);
            }
        },
        
        /**
         * Sort feature table by column
         * @param {string} sortBy - Column to sort by
         */
        sortFeatureTable: function(sortBy) {
            try {
                const table = document.querySelector('.feature-importance-table');
                const tableBody = document.getElementById('feature-impact-data');
                if (!table || !tableBody) return;
                
                // Update sort indicators
                const headers = table.querySelectorAll('th.sortable');
                headers.forEach(header => {
                    const indicator = header.querySelector('.sort-indicator');
                    if (header.dataset.sort === sortBy) {
                        if (indicator.textContent === '▼') {
                            indicator.textContent = '▲';
                        } else {
                            indicator.textContent = '▼';
                        }
                    } else {
                        indicator.textContent = '';
                    }
                });
                
                // Get current sort direction
                const sortHeader = table.querySelector(`th[data-sort="${sortBy}"]`);
                const sortDirection = sortHeader.querySelector('.sort-indicator').textContent === '▼' ? 'desc' : 'asc';
                
                // Get all rows as array for sorting
                const rows = Array.from(tableBody.querySelectorAll('tr'));
                
                // Sort rows based on column and direction
                rows.sort((rowA, rowB) => {
                    const cellA = rowA.querySelector(`td:nth-child(${getColumnIndex(sortBy)})`);
                    const cellB = rowB.querySelector(`td:nth-child(${getColumnIndex(sortBy)})`);
                    
                    let valueA, valueB;
                    
                    if (sortBy === 'name') {
                        valueA = cellA.textContent.toLowerCase();
                        valueB = cellB.textContent.toLowerCase();
                        return sortDirection === 'desc' 
                            ? valueA.localeCompare(valueB)
                            : valueB.localeCompare(valueA);
                    } else {
                        valueA = parseFloat(cellA.textContent);
                        valueB = parseFloat(cellB.textContent);
                        
                        if (sortBy === 'impact') {
                            // Sort by absolute value for impact
                            valueA = Math.abs(valueA);
                            valueB = Math.abs(valueB);
                        }
                        
                        return sortDirection === 'desc' 
                            ? valueB - valueA 
                            : valueA - valueB;
                    }
                });
                
                // Function to get column index based on sort key
                function getColumnIndex(key) {
                    switch (key) {
                        case 'name': return 1;
                        case 'impact': return 2;
                        case 'importance': return 3;
                        default: return 1;
                    }
                }
                
                // Reappend sorted rows
                rows.forEach(row => tableBody.appendChild(row));
            } catch (error) {
                console.error("Error sorting feature table:", error);
            }
        },
        
        /**
         * Show no data message in container
         * @param {HTMLElement} container - Container element
         * @param {string} message - Message to display
         */
        showNoDataMessage: function(container, message) {
            container.innerHTML = `
                <div style="padding: 40px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto; max-width: 800px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                    <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #d32f2f;">Dados não disponíveis</h3>
                    <p style="color: #333; font-size: 16px; line-height: 1.4;">
                        ${message}
                    </p>
                    <p style="color: #333; margin-top: 20px; font-size: 14px;">
                        Não serão gerados dados sintéticos ou demonstrativos. Apenas dados reais serão exibidos.
                    </p>
                </div>`;
        },
        
        /**
         * Show error message in container
         * @param {HTMLElement} container - Container element
         * @param {string} errorMessage - Error message to display
         */
        showErrorMessage: function(container, errorMessage) {
            container.innerHTML = `
                <div style="padding: 40px; text-align: center; background-color: #fff0f0; border: 1px solid #ffcccc; border-radius: 8px; margin: 20px auto; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                    <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #cc0000;">Chart Error</h3>
                    <p style="color: #666; font-size: 16px; line-height: 1.4;">${errorMessage}</p>
                </div>`;
        }
    };
    
    // Initialize when document is ready
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize after a small delay to ensure everything is loaded
        setTimeout(function() {
            if (document.getElementById('feature-importance-chart')) {
                window.FeatureImportanceHandler.initialize();
            }
        }, 500);
    });
})();