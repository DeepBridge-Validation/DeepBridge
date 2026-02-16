/**
 * Chart Utilities for DeepBridge Reports
 *
 * This module provides common utilities for chart rendering across all report types.
 * Extracted from embedded JavaScript in old renderer files.
 */

(function() {
    'use strict';

    // Only define ChartUtils if it doesn't already exist
    if (typeof window.ChartUtils !== 'undefined') {
        console.log("ChartUtils already exists, skipping initialization");
        return;
    }

    /**
     * Global ChartUtils object
     */
    window.ChartUtils = {
        /**
         * Get color for a model based on name or index
         * @param {string} modelName - Name of the model
         * @param {number} index - Index for fallback color generation
         * @returns {string} RGBA color string
         */
        getModelColor: function(modelName, index) {
            // Predefined colors for known models
            const modelColors = {
                'Primary Model': 'rgba(31, 119, 180, 0.7)',
                'primary_model': 'rgba(31, 119, 180, 0.7)',
                'GLM_CLASSIFIER': 'rgba(255, 127, 14, 0.7)',
                'GAM_CLASSIFIER': 'rgba(44, 160, 44, 0.7)',
                'GBM': 'rgba(214, 39, 40, 0.7)',
                'XGB': 'rgba(148, 103, 189, 0.7)',
                'RANDOM_FOREST': 'rgba(140, 86, 75, 0.7)',
                'SVM': 'rgba(227, 119, 194, 0.7)',
                'NEURAL_NETWORK': 'rgba(127, 127, 127, 0.7)'
            };

            // Return predefined color if available
            if (modelColors[modelName]) {
                return modelColors[modelName];
            }

            // Generate deterministic color based on model name or index
            const hash = index || Array.from(modelName).reduce((hash, char) => {
                return ((hash << 5) - hash) + char.charCodeAt(0);
            }, 0);

            const r = Math.abs(hash) % 200 + 55;
            const g = Math.abs(hash * 31) % 200 + 55;
            const b = Math.abs(hash * 17) % 200 + 55;

            return `rgba(${r}, ${g}, ${b}, 0.7)`;
        },

        /**
         * Format number for display
         * @param {number} value - Number to format
         * @param {number} decimals - Number of decimal places
         * @returns {string} Formatted number
         */
        formatNumber: function(value, decimals = 4) {
            if (value === null || value === undefined || isNaN(value)) {
                return 'N/A';
            }
            return Number(value).toFixed(decimals);
        },

        /**
         * Format percentage for display
         * @param {number} value - Number to format (0-1 or 0-100)
         * @param {number} decimals - Number of decimal places
         * @returns {string} Formatted percentage
         */
        formatPercentage: function(value, decimals = 1) {
            if (value === null || value === undefined || isNaN(value)) {
                return 'N/A';
            }
            // Assume value is 0-1 if less than 1.5, otherwise 0-100
            const percentage = value < 1.5 ? value * 100 : value;
            return `${Number(percentage).toFixed(decimals)}%`;
        },

        /**
         * Get report data safely from window
         * @returns {object} Report data
         */
        getReportData: function() {
            return window.reportData || {};
        },

        /**
         * Get chart data safely from window
         * @returns {object} Chart data
         */
        getChartData: function() {
            return window.chartData || {};
        },

        /**
         * Create default Chart.js options
         * @param {string} title - Chart title
         * @param {string} xLabel - X-axis label
         * @param {string} yLabel - Y-axis label
         * @returns {object} Chart.js options
         */
        getDefaultChartOptions: function(title, xLabel, yLabel) {
            return {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: !!title,
                        text: title,
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += ChartUtils.formatNumber(context.parsed.y, 4);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: !!xLabel,
                            text: xLabel
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: !!yLabel,
                            text: yLabel
                        }
                    }
                }
            };
        },

        /**
         * Show error message in chart container
         * @param {string} containerId - ID of container element
         * @param {string} message - Error message to display
         */
        showError: function(containerId, message) {
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        <strong>Error:</strong> ${message}
                    </div>
                `;
            }
        },

        /**
         * Show loading indicator in container
         * @param {string} containerId - ID of container element
         */
        showLoading: function(containerId) {
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = `
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="sr-only">Loading...</span>
                        </div>
                    </div>
                `;
            }
        }
    };

    // Global error handler for chart-related errors
    window.addEventListener('error', function(event) {
        if (event.message && (
            event.message.includes('Chart') ||
            event.message.includes('canvas')
        )) {
            console.error('Chart error caught:', event.message);
            // Don't prevent default handling
            return false;
        }
    });

    console.log('ChartUtils initialized');
})();
