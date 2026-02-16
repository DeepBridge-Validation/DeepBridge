// CRQROverviewChartManager.js - Chart Manager for Uncertainty Overview Section
const CRQROverviewChartManager = {
    /**
     * Initialize CRQR overview chart
     * @param {string} elementId - Chart container ID
     */
    initializeCRQRChart: function(elementId) {
        console.log("Initializing CRQR overview chart");
        const chartElement = document.getElementById(elementId);
        if (!chartElement) {
            console.error("Chart element not found:", elementId);
            return;
        }

        try {
            // Check if CRQRDataManager is available
            if (typeof CRQRDataManager === 'undefined') {
                console.error("CRQRDataManager not available");
                this.showNoDataMessage(chartElement, "Data manager not available");
                return;
            }

            // Extract CRQR data
            const crqrData = CRQRDataManager.extractCRQRData();

            if (!crqrData || crqrData.alphas.length === 0) {
                this.showNoDataMessage(chartElement, "No uncertainty quantification data available");
                return;
            }

            console.log("Creating CRQR overview visualization with data:", crqrData);

            // Create visualization
            this.renderCRQROverview(chartElement, crqrData);

        } catch (error) {
            console.error("Error initializing CRQR chart:", error);
            this.showNoDataMessage(chartElement, "Error loading uncertainty data");
        }
    },

    /**
     * Render CRQR overview visualization
     */
    renderCRQROverview: function(container, crqrData) {
        // Prepare chart data
        const chartData = CRQRDataManager.prepareCoverageChartData(crqrData);

        // Create HTML visualization
        let html = `
            <div class="bg-white p-6 rounded-lg shadow-lg">
                <h3 class="text-xl font-bold mb-6 text-gray-800">Uncertainty Quantification Overview</h3>

                <!-- Summary Cards -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div class="bg-blue-50 p-4 rounded-lg border border-blue-200">
                        <div class="text-sm text-blue-600 font-medium mb-1">Model</div>
                        <div class="text-lg font-bold text-blue-900">${crqrData.summary.model_name}</div>
                        <div class="text-xs text-blue-600 mt-1">${crqrData.summary.model_type}</div>
                    </div>
                    <div class="bg-purple-50 p-4 rounded-lg border border-purple-200">
                        <div class="text-sm text-purple-600 font-medium mb-1">Uncertainty Score</div>
                        <div class="text-lg font-bold text-purple-900">${CRQRDataManager.formatNumber(crqrData.summary.uncertainty_score, 4)}</div>
                        <div class="text-xs text-purple-600 mt-1">Overall assessment</div>
                    </div>
                    <div class="bg-green-50 p-4 rounded-lg border border-green-200">
                        <div class="text-sm text-green-600 font-medium mb-1">Alpha Levels Tested</div>
                        <div class="text-lg font-bold text-green-900">${crqrData.alphas.length}</div>
                        <div class="text-xs text-green-600 mt-1">Significance levels</div>
                    </div>
                </div>

                <!-- Coverage Chart -->
                <div class="mb-6">
                    <h4 class="text-lg font-semibold mb-4 text-gray-700">Coverage Analysis by Alpha Level</h4>
                    <div class="space-y-4">
        `;

        // Add bars for each alpha level
        chartData.labels.forEach((label, i) => {
            const expected = chartData.expected[i] * 100;
            const actual = chartData.actual[i] * 100;
            const diff = actual - expected;
            const absGap = Math.abs(diff);

            const borderColor = absGap < 2 ? 'border-green-500' : absGap < 5 ? 'border-yellow-500' : 'border-red-500';
            const textColor = absGap < 2 ? 'text-green-600' : absGap < 5 ? 'text-yellow-600' : 'text-red-600';
            const bgColor = absGap < 2 ? 'bg-green-50' : absGap < 5 ? 'bg-yellow-50' : 'bg-red-50';

            html += `
                <div class="border-l-4 ${borderColor} ${bgColor} p-4 rounded-r-lg">
                    <div class="flex justify-between items-center mb-2">
                        <div>
                            <span class="text-base font-semibold text-gray-800">${label}</span>
                            <span class="text-sm text-gray-500 ml-2">(Expected: ${expected.toFixed(1)}%)</span>
                        </div>
                        <div class="text-right">
                            <span class="text-lg font-bold ${textColor}">${actual.toFixed(2)}%</span>
                            <span class="text-sm ${textColor} ml-2">${diff > 0 ? '+' : ''}${diff.toFixed(2)}%</span>
                        </div>
                    </div>
                    <div class="relative h-8 bg-gray-200 rounded-full overflow-hidden">
                        <div class="absolute h-full bg-blue-200" style="width: ${expected}%"></div>
                        <div class="absolute h-full bg-blue-600 opacity-75" style="width: ${actual}%"></div>
                    </div>
                    <div class="flex justify-between text-xs text-gray-500 mt-1">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                    </div>
                </div>
            `;
        });

        html += `
                    </div>
                    <div class="mt-4 flex items-center justify-end space-x-6 text-sm">
                        <div class="flex items-center">
                            <div class="w-6 h-3 bg-blue-200 rounded mr-2"></div>
                            <span class="text-gray-600">Expected Coverage</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-6 h-3 bg-blue-600 rounded mr-2"></div>
                            <span class="text-gray-600">Actual Coverage</span>
                        </div>
                    </div>
                </div>

                <!-- Interval Width Summary -->
                <div class="mb-6">
                    <h4 class="text-lg font-semibold mb-4 text-gray-700">Prediction Interval Width by Alpha</h4>
                    <div class="grid grid-cols-1 md:grid-cols-${Math.min(crqrData.alphas.length, 3)} gap-4">
        `;

        // Add width cards for each alpha
        crqrData.alphas.forEach(alpha => {
            const alphaKey = alpha.toString();
            const result = crqrData.results[alphaKey];
            if (result) {
                html += `
                    <div class="bg-white border border-gray-200 p-4 rounded-lg hover:shadow-md transition-shadow">
                        <div class="text-sm text-gray-500 mb-2">α = ${CRQRDataManager.formatNumber(alpha, 2)}</div>
                        <div class="text-2xl font-bold text-gray-900 mb-1">${CRQRDataManager.formatNumber(result.mean_width, 3)}</div>
                        <div class="text-xs text-gray-600">Mean Width</div>
                        <div class="mt-2 pt-2 border-t border-gray-100">
                            <div class="flex justify-between text-xs text-gray-500">
                                <span>Median:</span>
                                <span class="font-medium">${CRQRDataManager.formatNumber(result.median_width, 3)}</span>
                            </div>
                        </div>
                    </div>
                `;
            }
        });

        html += `
                    </div>
                </div>

                <!-- Model Metrics -->
                <div>
                    <h4 class="text-lg font-semibold mb-4 text-gray-700">Model Performance Metrics</h4>
                    <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
        `;

        // Add metrics
        const metrics = crqrData.summary.metrics || {};
        const metricLabels = {
            'accuracy': 'Accuracy',
            'roc_auc': 'ROC AUC',
            'f1': 'F1 Score',
            'precision': 'Precision',
            'recall': 'Recall'
        };

        Object.keys(metricLabels).forEach(key => {
            if (metrics[key] !== undefined) {
                const value = metrics[key];
                const percentage = (value * 100).toFixed(1);
                html += `
                    <div class="bg-gray-50 p-3 rounded-lg text-center">
                        <div class="text-xs text-gray-500 mb-1">${metricLabels[key]}</div>
                        <div class="text-lg font-bold text-gray-900">${percentage}%</div>
                    </div>
                `;
            }
        });

        html += `
                    </div>
                </div>

                <!-- Info Box -->
                <div class="mt-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded-r">
                    <p class="text-sm text-gray-700">
                        <strong>About CRQR:</strong> Conformal Quantile Regression provides prediction intervals with
                        guaranteed coverage. The actual coverage should closely match the expected coverage (1 - α) for
                        well-calibrated models.
                    </p>
                </div>
            </div>
        `;

        container.innerHTML = html;
    },

    /**
     * Show "no data" message in container
     */
    showNoDataMessage: function(container, message) {
        container.innerHTML = `
            <div class="bg-white p-6 rounded-lg shadow">
                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4">
                    <div class="flex">
                        <div class="flex-shrink-0">
                            <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
                            </svg>
                        </div>
                        <div class="ml-3">
                            <h3 class="text-sm font-medium text-yellow-800">No Data Available</h3>
                            <div class="mt-2 text-sm text-yellow-700">
                                <p>${message}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
};

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.CRQROverviewChartManager = CRQROverviewChartManager;
    window.ChartManager = CRQROverviewChartManager; // Alias for compatibility
}
