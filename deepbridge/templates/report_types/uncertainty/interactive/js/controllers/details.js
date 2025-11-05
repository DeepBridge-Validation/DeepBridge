// CRQRDetailsController.js - Controller for CRQR details view
const CRQRDetailsController = {
    data: null,
    selectedAlpha: null,

    /**
     * Initialize the controller
     */
    init: function() {
        console.log("Initializing CRQRDetailsController");

        try {
            // Check if CRQRDataManager is available
            if (typeof CRQRDataManager === 'undefined') {
                console.error("CRQRDataManager not found. Make sure it's loaded before this controller.");
                this.showNoDataMessage("Data manager not available");
                return;
            }

            // Extract data
            this.data = CRQRDataManager.extractCRQRData();

            // Check if we have data
            if (!this.data || this.data.alphas.length === 0) {
                console.warn("No CRQR data available");
                this.showNoDataMessage("No uncertainty quantification data available");
                return;
            }

            console.log("CRQR data loaded successfully:", this.data);

            // Set default selected alpha (first one)
            this.selectedAlpha = this.data.alphas[0].toString();

            // Initialize the view
            this.render();

        } catch (error) {
            console.error("Error initializing CRQRDetailsController:", error);
            this.showNoDataMessage("Error loading uncertainty data");
        }
    },

    /**
     * Render the complete details view
     */
    render: function() {
        // Render alpha selector
        this.renderAlphaSelector();

        // Render coverage summary
        this.renderCoverageSummary();

        // Render interval width analysis
        this.renderWidthAnalysis();

        // Render calibration quality
        this.renderCalibrationQuality();

        // Render sample predictions table
        this.renderSamplePredictions();

        // Render coverage chart
        this.renderCoverageChart();
    },

    /**
     * Render alpha level selector
     */
    renderAlphaSelector: function() {
        const container = document.getElementById('alpha-selector');
        if (!container) {
            console.warn("Alpha selector container not found");
            return;
        }

        let html = '<div class="mb-4"><label class="block text-sm font-medium text-gray-700 mb-2">Select Significance Level (α):</label>';
        html += '<select id="alpha-select" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md">';

        this.data.alphas.forEach(alpha => {
            const alphaKey = alpha.toString();
            const result = this.data.results[alphaKey];
            const selected = alphaKey === this.selectedAlpha ? 'selected' : '';
            html += `<option value="${alphaKey}" ${selected}>α = ${CRQRDataManager.formatNumber(alpha, 2)} (Expected Coverage: ${CRQRDataManager.formatNumber(result.expected_coverage * 100, 1)}%)</option>`;
        });

        html += '</select></div>';
        container.innerHTML = html;

        // Add event listener
        const select = document.getElementById('alpha-select');
        if (select) {
            select.addEventListener('change', (e) => {
                this.selectedAlpha = e.target.value;
                this.render();
            });
        }
    },

    /**
     * Render coverage summary for selected alpha
     */
    renderCoverageSummary: function() {
        const container = document.getElementById('coverage-summary');
        if (!container) return;

        const result = this.data.results[this.selectedAlpha];
        if (!result) {
            container.innerHTML = '<p class="text-gray-500">No data available for selected alpha</p>';
            return;
        }

        const calibrationClass = CRQRDataManager.getCalibrationBgColorClass(result.coverage_gap);
        const colorClass = CRQRDataManager.getCalibrationColorClass(result.coverage_gap);

        const html = `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-white p-4 rounded-lg shadow">
                    <h4 class="text-sm font-medium text-gray-500 mb-2">Expected Coverage</h4>
                    <p class="text-2xl font-bold text-blue-600">${CRQRDataManager.formatNumber(result.expected_coverage * 100, 2)}%</p>
                    <p class="text-xs text-gray-500 mt-1">Target: ${CRQRDataManager.formatNumber((1 - result.alpha) * 100, 1)}%</p>
                </div>
                <div class="bg-white p-4 rounded-lg shadow ${calibrationClass}">
                    <h4 class="text-sm font-medium text-gray-500 mb-2">Actual Coverage</h4>
                    <p class="text-2xl font-bold ${colorClass}">${CRQRDataManager.formatNumber(result.coverage * 100, 2)}%</p>
                    <p class="text-xs text-gray-500 mt-1">Gap: ${CRQRDataManager.formatNumber(result.coverage_gap * 100, 2)}%</p>
                </div>
                <div class="bg-white p-4 rounded-lg shadow">
                    <h4 class="text-sm font-medium text-gray-500 mb-2">Calibration Quality</h4>
                    <p class="text-2xl font-bold ${colorClass}">
                        ${result.is_well_calibrated ? '✓ Good' : '⚠ Fair'}
                    </p>
                    <p class="text-xs text-gray-500 mt-1">
                        ${result.is_well_calibrated ? 'Well calibrated' : 'Needs improvement'}
                    </p>
                </div>
            </div>
        `;

        container.innerHTML = html;
    },

    /**
     * Render interval width analysis
     */
    renderWidthAnalysis: function() {
        const container = document.getElementById('width-analysis');
        if (!container) return;

        const result = this.data.results[this.selectedAlpha];
        if (!result || !result.widths || result.widths.length === 0) {
            container.innerHTML = '<p class="text-gray-500">No interval width data available</p>';
            return;
        }

        const stats = CRQRDataManager.calculateWidthStats(result.widths);

        const html = `
            <div class="bg-white p-4 rounded-lg shadow">
                <h4 class="text-lg font-semibold mb-4">Prediction Interval Width Statistics</h4>
                <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div>
                        <p class="text-sm text-gray-500">Mean Width</p>
                        <p class="text-xl font-bold text-gray-900">${CRQRDataManager.formatNumber(stats.mean, 3)}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Median Width</p>
                        <p class="text-xl font-bold text-gray-900">${CRQRDataManager.formatNumber(stats.median, 3)}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Min Width</p>
                        <p class="text-xl font-bold text-green-600">${CRQRDataManager.formatNumber(stats.min, 3)}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Max Width</p>
                        <p class="text-xl font-bold text-red-600">${CRQRDataManager.formatNumber(stats.max, 3)}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Q1 (25th)</p>
                        <p class="text-xl font-bold text-gray-700">${CRQRDataManager.formatNumber(stats.q1, 3)}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Q3 (75th)</p>
                        <p class="text-xl font-bold text-gray-700">${CRQRDataManager.formatNumber(stats.q3, 3)}</p>
                    </div>
                </div>
                <div class="mt-4 p-3 bg-blue-50 rounded">
                    <p class="text-sm text-gray-700">
                        <strong>Interpretation:</strong> Narrower intervals indicate higher confidence in predictions,
                        while wider intervals reflect greater uncertainty.
                    </p>
                </div>
            </div>
        `;

        container.innerHTML = html;
    },

    /**
     * Render calibration quality assessment
     */
    renderCalibrationQuality: function() {
        const container = document.getElementById('calibration-quality');
        if (!container) return;

        const result = this.data.results[this.selectedAlpha];
        if (!result) {
            container.innerHTML = '<p class="text-gray-500">No calibration data available</p>';
            return;
        }

        const absGap = Math.abs(result.coverage_gap);
        let qualityText = '';
        let qualityColor = '';
        let recommendation = '';

        if (absGap < 0.02) {
            qualityText = 'Excellent';
            qualityColor = 'text-green-600';
            recommendation = 'The model is very well calibrated. The actual coverage closely matches the expected coverage.';
        } else if (absGap < 0.05) {
            qualityText = 'Good';
            qualityColor = 'text-yellow-600';
            recommendation = 'The model shows acceptable calibration. Consider recalibrating if higher precision is needed.';
        } else {
            qualityText = 'Needs Improvement';
            qualityColor = 'text-red-600';
            recommendation = 'The model calibration should be improved. Consider increasing the calibration set size or adjusting the conformal prediction method.';
        }

        const html = `
            <div class="bg-white p-4 rounded-lg shadow">
                <h4 class="text-lg font-semibold mb-4">Calibration Assessment</h4>
                <div class="mb-4">
                    <p class="text-sm text-gray-500">Calibration Quality</p>
                    <p class="text-2xl font-bold ${qualityColor}">${qualityText}</p>
                </div>
                <div class="mb-4">
                    <p class="text-sm text-gray-500">Coverage Gap</p>
                    <p class="text-xl font-semibold text-gray-900">${CRQRDataManager.formatNumber(absGap * 100, 2)}%</p>
                    <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                        <div class="h-2 rounded-full ${absGap < 0.02 ? 'bg-green-600' : absGap < 0.05 ? 'bg-yellow-600' : 'bg-red-600'}"
                             style="width: ${Math.min(absGap * 1000, 100)}%"></div>
                    </div>
                </div>
                <div class="p-3 bg-gray-50 rounded">
                    <p class="text-sm text-gray-700">${recommendation}</p>
                </div>
            </div>
        `;

        container.innerHTML = html;
    },

    /**
     * Render sample predictions with intervals
     */
    renderSamplePredictions: function() {
        const container = document.getElementById('sample-predictions');
        if (!container) return;

        const result = this.data.results[this.selectedAlpha];
        if (!result || !result.test_predictions || result.test_predictions.length === 0) {
            container.innerHTML = '<p class="text-gray-500">No prediction data available</p>';
            return;
        }

        // Show first 10 predictions
        const sampleSize = Math.min(10, result.test_predictions.length);
        let html = `
            <div class="bg-white p-4 rounded-lg shadow overflow-x-auto">
                <h4 class="text-lg font-semibold mb-4">Sample Predictions with Intervals (First ${sampleSize} of ${result.test_predictions.length})</h4>
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Index</th>
                            <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Prediction</th>
                            <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Lower Bound</th>
                            <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Upper Bound</th>
                            <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Width</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
        `;

        for (let i = 0; i < sampleSize; i++) {
            const pred = result.test_predictions[i];
            const predValue = pred[1]; // Probability of class 1
            const lower = result.lower_bounds[i];
            const upper = result.upper_bounds[i];
            const width = result.widths[i];

            html += `
                <tr class="${i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}">
                    <td class="px-3 py-2 text-sm text-gray-900">${i + 1}</td>
                    <td class="px-3 py-2 text-sm font-medium text-blue-600">${CRQRDataManager.formatNumber(predValue, 4)}</td>
                    <td class="px-3 py-2 text-sm text-gray-700">${CRQRDataManager.formatNumber(lower, 4)}</td>
                    <td class="px-3 py-2 text-sm text-gray-700">${CRQRDataManager.formatNumber(upper, 4)}</td>
                    <td class="px-3 py-2 text-sm text-gray-600">${CRQRDataManager.formatNumber(width, 4)}</td>
                </tr>
            `;
        }

        html += `
                    </tbody>
                </table>
            </div>
        `;

        container.innerHTML = html;
    },

    /**
     * Render coverage comparison chart
     */
    renderCoverageChart: function() {
        const container = document.getElementById('coverage-chart');
        if (!container) return;

        const chartData = CRQRDataManager.prepareCoverageChartData(this.data);

        // Create simple text-based visualization if no charting library
        let html = `
            <div class="bg-white p-4 rounded-lg shadow">
                <h4 class="text-lg font-semibold mb-4">Coverage: Expected vs Actual</h4>
                <div class="space-y-3">
        `;

        chartData.labels.forEach((label, i) => {
            const expected = chartData.expected[i] * 100;
            const actual = chartData.actual[i] * 100;
            const diff = actual - expected;
            const diffColor = Math.abs(diff) < 2 ? 'text-green-600' : Math.abs(diff) < 5 ? 'text-yellow-600' : 'text-red-600';

            html += `
                <div class="border-l-4 ${Math.abs(diff) < 2 ? 'border-green-500' : Math.abs(diff) < 5 ? 'border-yellow-500' : 'border-red-500'} pl-4">
                    <div class="flex justify-between items-center mb-1">
                        <span class="text-sm font-medium text-gray-700">${label}</span>
                        <span class="text-sm ${diffColor} font-semibold">${diff > 0 ? '+' : ''}${diff.toFixed(2)}%</span>
                    </div>
                    <div class="flex items-center space-x-2 text-xs text-gray-600">
                        <span>Expected: ${expected.toFixed(1)}%</span>
                        <span>|</span>
                        <span>Actual: ${actual.toFixed(1)}%</span>
                    </div>
                    <div class="mt-2 flex space-x-1">
                        <div class="flex-1 bg-blue-100 rounded-full h-2" style="width: ${expected}%"></div>
                        <div class="flex-1 bg-blue-600 rounded-full h-2" style="width: ${actual}%"></div>
                    </div>
                </div>
            `;
        });

        html += `
                </div>
                <div class="mt-4 flex items-center space-x-4 text-xs text-gray-600">
                    <div class="flex items-center">
                        <div class="w-4 h-2 bg-blue-100 rounded mr-1"></div>
                        <span>Expected</span>
                    </div>
                    <div class="flex items-center">
                        <div class="w-4 h-2 bg-blue-600 rounded mr-1"></div>
                        <span>Actual</span>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = html;
    },

    /**
     * Show "no data" message
     */
    showNoDataMessage: function(message) {
        const containers = [
            'alpha-selector',
            'coverage-summary',
            'width-analysis',
            'calibration-quality',
            'sample-predictions',
            'coverage-chart'
        ];

        containers.forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                container.innerHTML = `
                    <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4">
                        <div class="flex">
                            <div class="flex-shrink-0">
                                <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
                                </svg>
                            </div>
                            <div class="ml-3">
                                <p class="text-sm text-yellow-700">${message}</p>
                            </div>
                        </div>
                    </div>
                `;
            }
        });
    }
};

// Export and auto-initialize when DOM is ready
if (typeof window !== 'undefined') {
    window.CRQRDetailsController = CRQRDetailsController;
    window.DetailsController = CRQRDetailsController; // Alias for compatibility
}
