// Feature Importance Table Manager
window.FeatureImportanceTableManager = {
    /**
     * Extract feature importance data from report data
     * @returns {Array} Array of feature data objects
     */
    extractFeatureData: function() {
        let featureData = [];
        
        try {
            // Try both reportData and reportConfig sources (reportConfig is preferred)
            let featureImportance = {};
            let modelFeatureImportance = {};
            let featureSubset = [];
            
            if (window.reportConfig) {
                featureImportance = window.reportConfig.feature_importance || {};
                modelFeatureImportance = window.reportConfig.model_feature_importance || {};
                featureSubset = window.reportConfig.feature_subset || [];
            } else if (window.reportData) {
                featureImportance = window.reportData.feature_importance || {};
                modelFeatureImportance = window.reportData.model_feature_importance || {};
                featureSubset = window.reportData.feature_subset || [];
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
                <div class="message-container">
                    <span class="message-icon">📊</span>
                    <h3>No Feature Data Available</h3>
                    <p>No feature importance data was found in the report.</p>
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
                <div class="message-container">
                    <span class="message-icon">⚠️</span>
                    <h3>Error Loading Data</h3>
                    <p>${errorMessage}</p>
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