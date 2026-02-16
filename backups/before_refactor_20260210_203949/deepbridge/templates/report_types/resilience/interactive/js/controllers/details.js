// ResilienceDetailsController.js - Controller for resilience details view
const ResilienceDetailsController = {
    data: null,

    /**
     * Initialize the controller
     */
    init: function() {
        console.log("Initializing ResilienceDetailsController");

        try {
            // Check if ResilienceDataManager is available
            if (typeof ResilienceDataManager === 'undefined') {
                console.error("ResilienceDataManager not found");
                this.showNoDataMessage("Data manager not available");
                return;
            }

            // Extract data
            this.data = ResilienceDataManager.extractResilienceData();

            // Check if we have data
            if (!this.data || this.data.shift_scenarios.length === 0) {
                console.warn("No resilience data available");
                this.showNoDataMessage("No distribution shift data available");
                return;
            }

            console.log("Resilience data loaded successfully:", this.data);

            // Initialize the view
            this.render();

        } catch (error) {
            console.error("Error initializing ResilienceDetailsController:", error);
            this.showNoDataMessage("Error loading resilience data");
        }
    },

    /**
     * Render the complete details view
     */
    render: function() {
        // Render shift scenarios summary
        this.renderShiftScenarios();

        // Render sensitive features
        this.renderSensitiveFeatures();

        // Render distance metrics comparison
        this.renderDistanceMetricsComparison();
    },

    /**
     * Render shift scenarios summary
     */
    renderShiftScenarios: function() {
        const container = document.getElementById('shift-scenarios');
        if (!container) {
            console.warn("Shift scenarios container not found");
            return;
        }

        const grouped = ResilienceDataManager.groupScenariosByMetric(this.data.shift_scenarios);

        let html = `
            <div class="section">
                <h4 class="subsection-title">Distribution Shift Scenarios</h4>
                <p>Total scenarios: ${this.data.shift_scenarios.length}</p>
        `;

        // Show summary by distance metric
        Object.keys(grouped).forEach(metric => {
            const scenarios = grouped[metric];
            const stats = ResilienceDataManager.calculateScenarioStats(scenarios);

            html += `
                <div class="metric-card">
                    <h5 class="metric-name">${metric}</h5>
                    <div class="metric-grid">
                        <div class="metric-item">
                            <span class="label">Scenarios:</span>
                            <span class="value">${stats.count}</span>
                        </div>
                        <div class="metric-item">
                            <span class="label">Valid gaps:</span>
                            <span class="value">${stats.valid_count}</span>
                        </div>
                        <div class="metric-item">
                            <span class="label">Avg gap:</span>
                            <span class="value">${ResilienceDataManager.formatNumber(stats.mean, 4)}</span>
                        </div>
                        <div class="metric-item">
                            <span class="label">Range:</span>
                            <span class="value">[${ResilienceDataManager.formatNumber(stats.min, 3)}, ${ResilienceDataManager.formatNumber(stats.max, 3)}]</span>
                        </div>
                    </div>
                </div>
            `;
        });

        html += `</div>`;
        container.innerHTML = html;
    },

    /**
     * Render sensitive features
     */
    renderSensitiveFeatures: function() {
        const container = document.getElementById('sensitive-features');
        if (!container) {
            console.warn("Sensitive features container not found");
            return;
        }

        if (this.data.sensitive_features.length === 0) {
            container.innerHTML = '<p class="text-muted">No sensitive features identified</p>';
            return;
        }

        let html = `
            <div class="section">
                <h4 class="subsection-title">Most Sensitive Features</h4>
                <p>Features most affected by distribution shifts</p>
                <div class="features-list">
        `;

        this.data.sensitive_features.slice(0, 10).forEach((feature, index) => {
            const impactPercent = Math.min((feature.impact || 0) * 10, 100);
            html += `
                <div class="feature-item">
                    <div class="feature-header">
                        <span class="feature-name">${index + 1}. ${feature.name}</span>
                        <span class="feature-impact">Impact: ${ResilienceDataManager.formatNumber(feature.impact, 2)}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${impactPercent}%"></div>
                    </div>
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;

        container.innerHTML = html;
    },

    /**
     * Render distance metrics comparison
     */
    renderDistanceMetricsComparison: function() {
        const container = document.getElementById('distance-metrics');
        if (!container) {
            console.warn("Distance metrics container not found");
            return;
        }

        const grouped = ResilienceDataManager.groupScenariosByMetric(this.data.shift_scenarios);

        let html = `
            <div class="section">
                <h4 class="subsection-title">Distance Metrics Comparison</h4>
                <div class="row">
        `;

        Object.keys(grouped).forEach(metric => {
            const scenarios = grouped[metric];
            const stats = ResilienceDataManager.calculateScenarioStats(scenarios);

            html += `
                <div class="col-4">
                    <div class="metric-card highlight">
                        <h5 class="metric-name">${metric}</h5>
                        <div class="metric-list">
                            <div class="metric-row">
                                <span>Total Scenarios:</span>
                                <strong>${stats.count}</strong>
                            </div>
                            <div class="metric-row">
                                <span>Valid Gaps:</span>
                                <strong>${stats.valid_count}</strong>
                            </div>
                            <div class="metric-row">
                                <span>Mean Gap:</span>
                                <strong class="primary">${ResilienceDataManager.formatNumber(stats.mean, 4)}</strong>
                            </div>
                            <div class="metric-row">
                                <span>Min Gap:</span>
                                <strong class="success">${ResilienceDataManager.formatNumber(stats.min, 4)}</strong>
                            </div>
                            <div class="metric-row">
                                <span>Max Gap:</span>
                                <strong class="error">${ResilienceDataManager.formatNumber(stats.max, 4)}</strong>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        html += `
                </div>
                <div class="info-box">
                    <p><strong>Note:</strong> Performance gap indicates how much the model's performance
                    degrades when the data distribution shifts. Lower values indicate better resilience.</p>
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
            'shift-scenarios',
            'sensitive-features',
            'distance-metrics'
        ];

        containers.forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                container.innerHTML = `
                    <div class="warning-box">
                        <p>${message}</p>
                    </div>
                `;
            }
        });
    }
};

// Export and auto-initialize when DOM is ready
if (typeof window !== 'undefined') {
    window.ResilienceDetailsController = ResilienceDetailsController;
    window.DetailsController = ResilienceDetailsController; // Alias for compatibility
}
