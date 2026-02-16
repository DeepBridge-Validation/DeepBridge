// CRQROverviewController.js - Controller for CRQR Overview Section
const CRQROverviewController = {
    /**
     * Initialize the overview controller
     */
    init: function() {
        console.log("Initializing CRQROverviewController");

        try {
            // Check if dependencies are available
            if (typeof CRQRDataManager === 'undefined') {
                console.error("CRQRDataManager not available");
                this.showError("Data manager not available");
                return;
            }

            if (typeof CRQROverviewChartManager === 'undefined') {
                console.error("CRQROverviewChartManager not available");
                this.showError("Chart manager not available");
                return;
            }

            // Check if we have data
            if (!CRQRDataManager.hasData()) {
                console.warn("No CRQR data available");
                this.showNoDataMessage();
                return;
            }

            console.log("Dependencies loaded, initializing overview");

            // Initialize overview chart
            this.initializeOverviewChart();

            // Initialize summary cards
            this.initializeSummary();

        } catch (error) {
            console.error("Error initializing CRQROverviewController:", error);
            this.showError("Error loading overview");
        }
    },

    /**
     * Initialize the overview chart
     */
    initializeOverviewChart: function() {
        const chartContainer = document.getElementById('overview-chart');
        if (!chartContainer) {
            console.warn("Overview chart container not found");
            return;
        }

        try {
            CRQROverviewChartManager.initializeCRQRChart('overview-chart');
            console.log("Overview chart initialized successfully");
        } catch (error) {
            console.error("Error initializing overview chart:", error);
            this.showChartError(chartContainer, "Error loading chart");
        }
    },

    /**
     * Initialize summary cards
     */
    initializeSummary: function() {
        const summaryContainer = document.getElementById('overview-summary');
        if (!summaryContainer) {
            console.warn("Overview summary container not found");
            return;
        }

        try {
            const crqrData = CRQRDataManager.extractCRQRData();

            if (!crqrData || crqrData.alphas.length === 0) {
                summaryContainer.innerHTML = '<p class="text-gray-500">No summary data available</p>';
                return;
            }

            // Calculate overall calibration quality
            let wellCalibratedCount = 0;
            crqrData.alphas.forEach(alpha => {
                const result = crqrData.results[alpha.toString()];
                if (result && result.is_well_calibrated) {
                    wellCalibratedCount++;
                }
            });

            const calibrationPercentage = (wellCalibratedCount / crqrData.alphas.length) * 100;
            const calibrationClass = calibrationPercentage >= 80 ? 'text-green-600' : calibrationPercentage >= 50 ? 'text-yellow-600' : 'text-red-600';

            const html = `
                <div class="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-lg shadow-md border border-blue-200">
                    <h3 class="text-2xl font-bold mb-4 text-gray-800">Uncertainty Quantification Summary</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <h4 class="text-lg font-semibold text-gray-700 mb-3">Model Information</h4>
                            <div class="space-y-2 text-sm">
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Model Name:</span>
                                    <span class="font-medium text-gray-900">${crqrData.summary.model_name}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Model Type:</span>
                                    <span class="font-medium text-gray-900">${crqrData.summary.model_type}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Uncertainty Score:</span>
                                    <span class="font-medium text-blue-600">${CRQRDataManager.formatNumber(crqrData.summary.uncertainty_score, 4)}</span>
                                </div>
                            </div>
                        </div>
                        <div>
                            <h4 class="text-lg font-semibold text-gray-700 mb-3">Calibration Assessment</h4>
                            <div class="space-y-2 text-sm">
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Alpha Levels Tested:</span>
                                    <span class="font-medium text-gray-900">${crqrData.alphas.length}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Well Calibrated:</span>
                                    <span class="font-medium ${calibrationClass}">${wellCalibratedCount} / ${crqrData.alphas.length}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Calibration Quality:</span>
                                    <span class="font-medium ${calibrationClass}">${calibrationPercentage.toFixed(0)}%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="mt-4 p-3 bg-white bg-opacity-60 rounded">
                        <p class="text-xs text-gray-700">
                            <strong>Note:</strong> CRQR (Conformal Quantile Regression) provides prediction intervals with theoretical guarantees.
                            A well-calibrated model has actual coverage close to expected coverage.
                        </p>
                    </div>
                </div>
            `;

            summaryContainer.innerHTML = html;

        } catch (error) {
            console.error("Error initializing summary:", error);
            summaryContainer.innerHTML = '<p class="text-red-500">Error loading summary</p>';
        }
    },

    /**
     * Show "no data" message
     */
    showNoDataMessage: function() {
        const containers = ['overview-chart', 'overview-summary'];

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
                                <p class="text-sm text-yellow-700">
                                    No uncertainty quantification data available. Please run the CRQR test first.
                                </p>
                            </div>
                        </div>
                    </div>
                `;
            }
        });
    },

    /**
     * Show general error message
     */
    showError: function(message) {
        const containers = ['overview-chart', 'overview-summary'];

        containers.forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                container.innerHTML = `
                    <div class="bg-red-50 border-l-4 border-red-400 p-4">
                        <div class="flex">
                            <div class="flex-shrink-0">
                                <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                                </svg>
                            </div>
                            <div class="ml-3">
                                <p class="text-sm text-red-700">${message}</p>
                            </div>
                        </div>
                    </div>
                `;
            }
        });
    },

    /**
     * Show chart-specific error
     */
    showChartError: function(container, message) {
        container.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded p-4">
                <p class="text-sm text-red-700">${message}</p>
            </div>
        `;
    }
};

// Export and provide alias for compatibility
if (typeof window !== 'undefined') {
    window.CRQROverviewController = CRQROverviewController;
    window.OverviewController = CRQROverviewController; // Alias for compatibility
}
