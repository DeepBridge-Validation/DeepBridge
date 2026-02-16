/**
 * Robustness Chart Rendering Module
 *
 * This module handles all chart rendering for robustness reports.
 * Replaces embedded JavaScript from robustness_renderer.py
 */

(function() {
    'use strict';

    /**
     * RobustnessCharts class for rendering robustness-specific charts
     */
    class RobustnessCharts {
        constructor(data) {
            this.data = data || ChartUtils.getReportData();
            this.charts = {};
        }

        /**
         * Render all robustness charts
         */
        renderAll() {
            try {
                this.renderPerturbationChart();
                this.renderFeatureImportanceChart();
                this.renderMetricsOverview();
                console.log('All robustness charts rendered successfully');
            } catch (error) {
                console.error('Error rendering charts:', error);
                ChartUtils.showError('charts-container', error.message);
            }
        }

        /**
         * Render perturbation impact chart
         */
        renderPerturbationChart() {
            const canvasId = 'perturbation-chart';
            const canvas = document.getElementById(canvasId);

            if (!canvas) {
                console.warn(`Canvas #${canvasId} not found`);
                return;
            }

            const ctx = canvas.getContext('2d');

            // Prepare data for chart
            const rawResults = this.data.perturbation_results_raw || [];
            const quantileResults = this.data.perturbation_results_quantile || [];

            const labels = rawResults.map(r => r.level);

            const datasets = [];

            // Raw perturbations
            if (rawResults.length > 0) {
                datasets.push({
                    label: 'Raw Perturbations',
                    data: rawResults.map(r => r.mean_score),
                    borderColor: 'rgba(31, 119, 180, 1)',
                    backgroundColor: 'rgba(31, 119, 180, 0.2)',
                    borderWidth: 2,
                    tension: 0.1
                });

                // Worst scores
                if (rawResults[0].worst_score !== null) {
                    datasets.push({
                        label: 'Raw Perturbations (Worst)',
                        data: rawResults.map(r => r.worst_score),
                        borderColor: 'rgba(214, 39, 40, 1)',
                        backgroundColor: 'rgba(214, 39, 40, 0.2)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        tension: 0.1
                    });
                }
            }

            // Quantile perturbations
            if (quantileResults.length > 0) {
                datasets.push({
                    label: 'Quantile Perturbations',
                    data: quantileResults.map(r => r.mean_score),
                    borderColor: 'rgba(44, 160, 44, 1)',
                    backgroundColor: 'rgba(44, 160, 44, 0.2)',
                    borderWidth: 2,
                    tension: 0.1
                });
            }

            // Add baseline if available
            if (this.data.metrics && this.data.metrics.base_score) {
                const baselineData = new Array(labels.length).fill(this.data.metrics.base_score);
                datasets.push({
                    label: 'Baseline',
                    data: baselineData,
                    borderColor: 'rgba(127, 127, 127, 1)',
                    backgroundColor: 'rgba(127, 127, 127, 0.1)',
                    borderWidth: 2,
                    borderDash: [10, 5],
                    pointRadius: 0
                });
            }

            // Create chart
            this.charts.perturbation = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: datasets
                },
                options: ChartUtils.getDefaultChartOptions(
                    'Impact of Perturbations on Model Performance',
                    'Perturbation Level',
                    this.data.metric_name || 'Score'
                )
            });
        }

        /**
         * Render feature importance chart
         */
        renderFeatureImportanceChart() {
            const canvasId = 'feature-importance-chart';
            const canvas = document.getElementById(canvasId);

            if (!canvas) {
                console.warn(`Canvas #${canvasId} not found`);
                return;
            }

            const featureImportance = this.data.feature_importance || [];

            if (featureImportance.length === 0) {
                console.warn('No feature importance data available');
                return;
            }

            // Take top 10 features
            const topFeatures = featureImportance.slice(0, 10);

            const ctx = canvas.getContext('2d');

            this.charts.featureImportance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: topFeatures.map(f => f.feature_name),
                    datasets: [{
                        label: 'Feature Importance',
                        data: topFeatures.map(f => f.importance),
                        backgroundColor: 'rgba(31, 119, 180, 0.7)',
                        borderColor: 'rgba(31, 119, 180, 1)',
                        borderWidth: 1
                    }]
                },
                options: Object.assign(
                    ChartUtils.getDefaultChartOptions(
                        'Top 10 Feature Importance',
                        'Features',
                        'Importance'
                    ),
                    {
                        indexAxis: 'y', // Horizontal bars
                        scales: {
                            x: {
                                beginAtZero: true
                            }
                        }
                    }
                )
            });
        }

        /**
         * Render metrics overview (summary stats)
         */
        renderMetricsOverview() {
            const containerId = 'metrics-overview';
            const container = document.getElementById(containerId);

            if (!container) {
                console.warn(`Container #${containerId} not found`);
                return;
            }

            const metrics = this.data.metrics || {};

            const html = `
                <div class="row">
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Robustness Score</h5>
                            <p class="metric-value">${ChartUtils.formatNumber(metrics.robustness_score, 3)}</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Base Score</h5>
                            <p class="metric-value">${ChartUtils.formatNumber(metrics.base_score, 3)}</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Raw Impact</h5>
                            <p class="metric-value">${ChartUtils.formatPercentage(metrics.avg_raw_impact, 2)}</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Quantile Impact</h5>
                            <p class="metric-value">${ChartUtils.formatPercentage(metrics.avg_quantile_impact, 2)}</p>
                        </div>
                    </div>
                </div>
            `;

            container.innerHTML = html;
        }

        /**
         * Destroy all charts (cleanup)
         */
        destroyAll() {
            Object.values(this.charts).forEach(chart => {
                if (chart) {
                    chart.destroy();
                }
            });
            this.charts = {};
        }
    }

    // Export to global scope
    window.RobustnessCharts = RobustnessCharts;

    // Auto-initialize when DOM is ready if data is available
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            if (window.reportData) {
                const charts = new RobustnessCharts(window.reportData);
                charts.renderAll();
            }
        });
    } else {
        // DOM already loaded
        if (window.reportData) {
            const charts = new RobustnessCharts(window.reportData);
            charts.renderAll();
        }
    }

    console.log('RobustnessCharts module loaded');
})();
