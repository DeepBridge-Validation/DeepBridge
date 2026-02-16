// Feature Importance Table Manager
window.FeatureImportanceTableManager = {
    /**
     * Extract feature importance data from report data
     * @returns {Array} Array of feature data objects
     */
    extractFeatureData: function() {
        let featureData = [];
        
        try {
            // Try all possible sources to find feature importance data
            let featureImportance = {};
            let modelFeatureImportance = {};
            let featureSubset = [];
            
            // Check multiple sources in order of preference
            if (window.reportConfig && window.reportConfig.feature_importance) {
                // First priority - directly from reportConfig
                featureImportance = window.reportConfig.feature_importance || {};
                modelFeatureImportance = window.reportConfig.model_feature_importance || {};
                featureSubset = window.reportConfig.feature_subset || [];
                console.log("Using feature data from reportConfig");
            } 
            else if (window.chartData && window.chartData.feature_importance) {
                // Second priority - from parsed chart data
                featureImportance = window.chartData.feature_importance || {};
                modelFeatureImportance = window.chartData.model_feature_importance || {};
                featureSubset = window.chartData.feature_subset || [];
                console.log("Using feature data from chartData");
            }
            else if (window.reportData) {
                // If we have reportData, check several possible locations
                if (window.reportData.feature_importance) {
                    // Direct property in reportData
                    featureImportance = window.reportData.feature_importance;
                    modelFeatureImportance = window.reportData.model_feature_importance || {};
                    featureSubset = window.reportData.feature_subset || [];
                    console.log("Using feature data from reportData");
                }
                else if (window.reportData.chart_data_json && typeof window.reportData.chart_data_json === 'string') {
                    // Try to parse the chart_data_json string
                    try {
                        const chartData = JSON.parse(window.reportData.chart_data_json);
                        if (chartData && chartData.feature_importance) {
                            featureImportance = chartData.feature_importance;
                            modelFeatureImportance = chartData.model_feature_importance || {};
                            featureSubset = chartData.feature_subset || [];
                            console.log("Using feature data from parsed chart_data_json");
                        }
                    } catch (e) {
                        console.error("Error parsing chart_data_json:", e);
                    }
                }
            }
            
            // Verificar se featureImportance é um objeto válido
            if (typeof featureImportance !== 'object' || featureImportance === null) {
                console.error("Feature importance data is not a valid object:", featureImportance);
                return [];
            }
            
            // Verificar se modelFeatureImportance é um objeto válido
            if (typeof modelFeatureImportance !== 'object' || modelFeatureImportance === null) {
                console.error("Model feature importance is not a valid object");
                modelFeatureImportance = {};
            }
            
            // Verificar se featureSubset é um array válido
            if (!Array.isArray(featureSubset)) {
                console.error("Feature subset is not a valid array");
                featureSubset = [];
            }
            
            // Convert to array format with calculated properties
            featureData = Object.entries(featureImportance).map(([name, value]) => ({
                name,
                robustness: value,
                robustnessAbs: Math.abs(value),
                impact: value,
                importance: modelFeatureImportance[name] || 0,
                impactType: value >= 0 ? 'positive' : 'negative',
                inSubset: featureSubset.includes(name)
            }));
            
            console.log(`Loaded ${featureData.length} features from data`);
        } catch (error) {
            console.error("Error extracting feature data:", error);
        }
        
        return featureData;
    },
    
    /**
     * Sort feature data by specified key and direction
     * @param {Array} data - Array of feature data objects
     * @param {string} key - Key to sort by
     * @param {string} direction - Sort direction ('asc' or 'desc')
     * @returns {Array} Sorted array
     */
    sortData: function(data, key, direction) {
        // Map sort keys to actual data properties
        const propertyMap = {
            'name': 'name',
            'impact': 'robustnessAbs', // Sort by absolute value for impact
            'importance': 'importance'
        };
        
        const sortKey = propertyMap[key] || 'robustnessAbs';
        
        return [...data].sort((a, b) => {
            let valueA = a[sortKey];
            let valueB = b[sortKey];
            
            // Special case for name: use string comparison
            if (sortKey === 'name') {
                return direction === 'asc' 
                    ? valueA.localeCompare(valueB) 
                    : valueB.localeCompare(valueA);
            }
            
            // For numeric values
            return direction === 'asc' 
                ? valueA - valueB 
                : valueB - valueA;
        });
    },
    
    /**
     * Filter feature data by search term and subset option
     * @param {Array} data - Array of feature data objects
     * @param {string} searchTerm - Search term
     * @param {boolean} showOnlySubset - Whether to show only subset features
     * @returns {Array} Filtered array
     */
    filterData: function(data, searchTerm, showOnlySubset) {
        return data.filter(item => {
            const matchesSearch = !searchTerm || item.name.toLowerCase().includes(searchTerm.toLowerCase());
            const matchesSubset = !showOnlySubset || item.inSubset;
            return matchesSearch && matchesSubset;
        });
    },
    
    /**
     * Format numeric value for display
     * @param {number} value - Value to format
     * @param {number} decimals - Number of decimal places
     * @returns {string} Formatted value
     */
    formatValue: function(value, decimals = 4) {
        return value.toFixed(decimals);
    },
    
    /**
     * Calculate visual bar width for importance values
     * @param {number} value - Importance value
     * @param {boolean} isModelImportance - Whether this is model importance
     * @returns {string} CSS width value as percentage
     */
    getBarWidth: function(value, isModelImportance = false) {
        if (isModelImportance) {
            // Model importance is typically larger, scale differently
            return `${Math.min(Math.abs(value) * 100 * 5, 100)}%`;
        }
        // For robustness, scale absolute value
        return `${Math.min(Math.abs(value) * 100 * 10, 100)}%`;
    },
    
    /**
     * Generate table rows HTML from feature data
     * @param {Array} data - Array of feature data objects
     * @param {string} hoveredRow - ID of currently hovered row
     * @returns {string} HTML for table rows
     */
    generateTableRows: function(data, hoveredRow = null) {
        let html = '';
        
        data.forEach(item => {
            const rowClasses = [
                hoveredRow === item.name ? 'hovered-row' : '',
                item.inSubset ? 'feature-subset-row' : ''
            ].filter(Boolean).join(' ');
            
            html += `
            <tr class="${rowClasses}" data-feature="${item.name}">
                <td><span class="feature-name">${item.name}</span></td>
                <td>
                    <div class="value-with-bar">
                        <span class="value-text">${this.formatValue(item.robustness)}</span>
                        <div class="progress-container">
                            <div class="progress-bar ${item.impactType}" 
                                 style="width: ${this.getBarWidth(item.robustness)}"></div>
                        </div>
                    </div>
                </td>
                <td>
                    <div class="value-with-bar">
                        <span class="value-text">${this.formatValue(item.importance)}</span>
                        <div class="progress-container">
                            <div class="progress-bar model" 
                                 style="width: ${this.getBarWidth(item.importance, true)}"></div>
                        </div>
                    </div>
                </td>
                <td>
                    <span class="subset-badge ${item.inSubset ? 'included' : 'excluded'}">
                        ${item.inSubset ? 'Included' : 'Excluded'}
                    </span>
                </td>
            </tr>`;
        });
        
        return html || this.generateNoDataMessage();
    },
    
    /**
     * Generate a "no data" message
     * @returns {string} HTML for no data message
     */
    generateNoDataMessage: function() {
        return `
        <tr>
            <td colspan="4" class="no-data-message">
                <div style="padding: 40px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                    <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #d32f2f;">Dados não disponíveis</h3>
                    <p style="color: #333; font-size: 16px; line-height: 1.4;">Não há dados de importância de características disponíveis.</p>
                    <p style="color: #333; margin-top: 20px; font-size: 14px;">
                        Execute testes de robustez com análise de importância de características habilitada.
                    </p>
                </div>
            </td>
        </tr>`;
    },
    
    /**
     * Generate an error message
     * @param {string} errorMessage - Error message to display
     * @returns {string} HTML for error message
     */
    generateErrorMessage: function(errorMessage) {
        return `
        <tr>
            <td colspan="4" class="error-message">
                <div style="padding: 40px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                    <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #d32f2f;">Erro ao carregar dados</h3>
                    <p style="color: #333; font-size: 16px; line-height: 1.4;">${errorMessage}</p>
                    <p style="color: #333; margin-top: 20px; font-size: 14px;">
                        Não serão exibidos dados sintéticos. Apenas dados reais são aceitos.
                    </p>
                </div>
            </td>
        </tr>`;
    },
    
    /**
     * Generate feature count statistics
     * @param {Array} data - Feature data array
     * @returns {Object} Count statistics
     */
    getFeatureCounts: function(data) {
        return {
            total: data.length,
            inSubset: data.filter(f => f.inSubset).length
        };
    }
};