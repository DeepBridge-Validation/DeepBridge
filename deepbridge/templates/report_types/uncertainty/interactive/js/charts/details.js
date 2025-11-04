// CRQRDataManager.js - Manages CRQR (Conformal Quantile Regression) data extraction and processing
const CRQRDataManager = {
    /**
     * Extract and process CRQR data from window.reportData
     * @returns {Object} Processed CRQR data structure
     */
    extractCRQRData: function() {
        try {
            console.log("Extracting CRQR data from window.reportData");

            // Check if window.reportData exists
            if (!window.reportData) {
                console.warn("window.reportData not found");
                return this.getEmptyDataStructure();
            }

            // Check if CRQR data exists in primary_model
            if (!window.reportData.primary_model ||
                !window.reportData.primary_model.crqr ||
                !window.reportData.primary_model.crqr.by_alpha) {
                console.warn("CRQR data not found in window.reportData.primary_model");
                return this.getEmptyDataStructure();
            }

            const crqrData = window.reportData.primary_model.crqr.by_alpha;
            const alphas = window.reportData.alphas || Object.keys(crqrData).map(parseFloat).sort((a, b) => a - b);

            console.log(`Found CRQR data for ${alphas.length} alpha values:`, alphas);

            // Process data for each alpha value
            const processedData = {
                alphas: alphas,
                results: {},
                summary: {
                    model_name: window.reportData.model_name || 'Model',
                    model_type: window.reportData.model_type || 'Unknown',
                    uncertainty_score: window.reportData.uncertainty_score || 0,
                    metrics: window.reportData.metrics || {}
                },
                alternative_models: window.reportData.alternative_models || {}
            };

            // Process each alpha level
            alphas.forEach(alpha => {
                const alphaKey = alpha.toString();
                if (crqrData[alphaKey] && crqrData[alphaKey].overall_result) {
                    const result = crqrData[alphaKey].overall_result;

                    processedData.results[alphaKey] = {
                        alpha: result.alpha,
                        coverage: result.coverage,
                        expected_coverage: result.expected_coverage,
                        mean_width: result.mean_width,
                        median_width: result.median_width,
                        coverage_gap: result.coverage - result.expected_coverage,
                        is_well_calibrated: Math.abs(result.coverage - result.expected_coverage) < 0.05,
                        widths: result.widths || [],
                        lower_bounds: result.lower_bounds || [],
                        upper_bounds: result.upper_bounds || [],
                        test_predictions: result.test_predictions || [],
                        split_sizes: result.split_sizes || {}
                    };
                }
            });

            console.log("CRQR data extraction complete:", processedData);
            return processedData;

        } catch (error) {
            console.error("Error extracting CRQR data:", error);
            return this.getEmptyDataStructure();
        }
    },

    /**
     * Get empty data structure for when no data is available
     */
    getEmptyDataStructure: function() {
        return {
            alphas: [],
            results: {},
            summary: {
                model_name: 'Model',
                model_type: 'Unknown',
                uncertainty_score: 0,
                metrics: {}
            },
            alternative_models: {}
        };
    },

    /**
     * Check if CRQR data is available
     */
    hasData: function() {
        return window.reportData &&
               window.reportData.primary_model &&
               window.reportData.primary_model.crqr &&
               Object.keys(window.reportData.primary_model.crqr.by_alpha || {}).length > 0;
    },

    /**
     * Format number for display
     */
    formatNumber: function(value, decimals = 4) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        return Number(value).toFixed(decimals);
    },

    /**
     * Get color class based on calibration quality
     */
    getCalibrationColorClass: function(coverageGap) {
        const absGap = Math.abs(coverageGap);
        if (absGap < 0.02) return 'text-green-600';
        if (absGap < 0.05) return 'text-yellow-600';
        return 'text-red-600';
    },

    /**
     * Get background color class based on calibration quality
     */
    getCalibrationBgColorClass: function(coverageGap) {
        const absGap = Math.abs(coverageGap);
        if (absGap < 0.02) return 'bg-green-50';
        if (absGap < 0.05) return 'bg-yellow-50';
        return 'bg-red-50';
    },

    /**
     * Calculate statistics for interval widths
     */
    calculateWidthStats: function(widths) {
        if (!widths || widths.length === 0) {
            return {
                min: 0,
                max: 0,
                mean: 0,
                median: 0,
                q1: 0,
                q3: 0
            };
        }

        const sorted = [...widths].sort((a, b) => a - b);
        const n = sorted.length;

        return {
            min: sorted[0],
            max: sorted[n - 1],
            mean: widths.reduce((sum, w) => sum + w, 0) / n,
            median: n % 2 === 0 ? (sorted[n/2 - 1] + sorted[n/2]) / 2 : sorted[Math.floor(n/2)],
            q1: sorted[Math.floor(n * 0.25)],
            q3: sorted[Math.floor(n * 0.75)]
        };
    },

    /**
     * Prepare data for coverage chart
     */
    prepareCoverageChartData: function(crqrData) {
        const chartData = {
            labels: [],
            expected: [],
            actual: []
        };

        crqrData.alphas.forEach(alpha => {
            const alphaKey = alpha.toString();
            if (crqrData.results[alphaKey]) {
                const result = crqrData.results[alphaKey];
                chartData.labels.push(`α=${this.formatNumber(alpha, 2)}`);
                chartData.expected.push(result.expected_coverage);
                chartData.actual.push(result.coverage);
            }
        });

        return chartData;
    },

    /**
     * Prepare data for interval width boxplot
     */
    prepareWidthBoxplotData: function(crqrData) {
        const boxplotData = [];

        crqrData.alphas.forEach(alpha => {
            const alphaKey = alpha.toString();
            if (crqrData.results[alphaKey]) {
                const result = crqrData.results[alphaKey];
                const stats = this.calculateWidthStats(result.widths);

                boxplotData.push({
                    label: `α=${this.formatNumber(alpha, 2)}`,
                    min: stats.min,
                    q1: stats.q1,
                    median: stats.median,
                    q3: stats.q3,
                    max: stats.max,
                    mean: stats.mean
                });
            }
        });

        return boxplotData;
    }
};

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.CRQRDataManager = CRQRDataManager;
}
