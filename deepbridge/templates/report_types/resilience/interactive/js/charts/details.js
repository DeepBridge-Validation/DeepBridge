// ResilienceDataManager.js - Manages resilience/distribution shift data extraction and processing
const ResilienceDataManager = {
    /**
     * Extract and process resilience data from window.reportData
     * @returns {Object} Processed resilience data structure
     */
    extractResilienceData: function() {
        try {
            console.log("Extracting resilience data from window.reportData");

            // Check if window.reportData exists
            if (!window.reportData) {
                console.warn("window.reportData not found");
                return this.getEmptyDataStructure();
            }

            const data = {
                model_name: window.reportData.model_name || 'Model',
                model_type: window.reportData.model_type || 'Unknown',
                resilience_score: window.reportData.resilience_score || 0,
                avg_performance_gap: window.reportData.avg_performance_gap,
                avg_dist_shift: window.reportData.avg_dist_shift || 0,
                distance_metrics: window.reportData.distance_metrics || [],
                alphas: window.reportData.alphas || [],
                shift_scenarios: window.reportData.shift_scenarios || [],
                sensitive_features: window.reportData.sensitive_features || [],
                baseline_dataset: window.reportData.baseline_dataset || 'Baseline',
                target_dataset: window.reportData.target_dataset || 'Target'
            };

            console.log(`Found ${data.shift_scenarios.length} shift scenarios`);
            console.log(`Sensitive features: ${data.sensitive_features.length}`);

            return data;

        } catch (error) {
            console.error("Error extracting resilience data:", error);
            return this.getEmptyDataStructure();
        }
    },

    /**
     * Get empty data structure for when no data is available
     */
    getEmptyDataStructure: function() {
        return {
            model_name: 'Model',
            model_type: 'Unknown',
            resilience_score: 0,
            avg_performance_gap: null,
            avg_dist_shift: 0,
            distance_metrics: [],
            alphas: [],
            shift_scenarios: [],
            sensitive_features: [],
            baseline_dataset: 'Baseline',
            target_dataset: 'Target'
        };
    },

    /**
     * Check if resilience data is available
     */
    hasData: function() {
        return window.reportData &&
               window.reportData.shift_scenarios &&
               window.reportData.shift_scenarios.length > 0;
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
     * Get color class based on performance gap
     */
    getPerformanceGapColorClass: function(gap) {
        if (gap === null || gap === undefined) return 'text-gray-500';
        const absGap = Math.abs(gap);
        if (absGap < 0.02) return 'text-green-600';
        if (absGap < 0.05) return 'text-yellow-600';
        return 'text-red-600';
    },

    /**
     * Get background color class based on performance gap
     */
    getPerformanceGapBgColorClass: function(gap) {
        if (gap === null || gap === undefined) return 'bg-gray-50';
        const absGap = Math.abs(gap);
        if (absGap < 0.02) return 'bg-green-50';
        if (absGap < 0.05) return 'bg-yellow-50';
        return 'bg-red-50';
    },

    /**
     * Group scenarios by distance metric
     */
    groupScenariosByMetric: function(scenarios) {
        const grouped = {};
        scenarios.forEach(scenario => {
            const metric = scenario.distance_metric || 'Unknown';
            if (!grouped[metric]) {
                grouped[metric] = [];
            }
            grouped[metric].push(scenario);
        });
        return grouped;
    },

    /**
     * Group scenarios by alpha
     */
    groupScenariosByAlpha: function(scenarios) {
        const grouped = {};
        scenarios.forEach(scenario => {
            const alpha = scenario.alpha || 0;
            const key = alpha.toString();
            if (!grouped[key]) {
                grouped[key] = [];
            }
            grouped[key].push(scenario);
        });
        return grouped;
    },

    /**
     * Calculate statistics for scenarios
     */
    calculateScenarioStats: function(scenarios) {
        const gaps = scenarios
            .map(s => s.performance_gap)
            .filter(g => g !== null && g !== undefined && !isNaN(g));

        if (gaps.length === 0) {
            return {
                mean: 0,
                min: 0,
                max: 0,
                count: 0,
                valid_count: 0
            };
        }

        return {
            mean: gaps.reduce((sum, g) => sum + g, 0) / gaps.length,
            min: Math.min(...gaps),
            max: Math.max(...gaps),
            count: scenarios.length,
            valid_count: gaps.length
        };
    },

    /**
     * Prepare data for shift comparison chart
     */
    prepareShiftComparisonData: function(resilienceData) {
        const grouped = this.groupScenariosByMetric(resilienceData.shift_scenarios);

        const chartData = {
            labels: [],
            datasets: []
        };

        Object.keys(grouped).forEach(metric => {
            const scenarios = grouped[metric];
            const gaps = scenarios.map(s => s.performance_gap);
            const alphas = scenarios.map(s => `α=${this.formatNumber(s.alpha, 2)}`);

            chartData.labels = alphas;
            chartData.datasets.push({
                label: metric,
                data: gaps
            });
        });

        return chartData;
    }
};

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.ResilienceDataManager = ResilienceDataManager;
}
