// ResilienceOverviewChartManager.js - Simple overview for resilience
const ResilienceOverviewChartManager = {
    initializeResilienceChart: function(elementId) {
        console.log("Initializing resilience overview chart");
        const chartElement = document.getElementById(elementId);
        if (!chartElement) return;

        try {
            if (typeof ResilienceDataManager === 'undefined') {
                this.showNoDataMessage(chartElement, "Data manager not available");
                return;
            }

            const data = ResilienceDataManager.extractResilienceData();
            if (!data || data.shift_scenarios.length === 0) {
                this.showNoDataMessage(chartElement, "No distribution shift data available");
                return;
            }

            this.renderOverview(chartElement, data);
        } catch (error) {
            console.error("Error initializing resilience chart:", error);
            this.showNoDataMessage(chartElement, "Error loading data");
        }
    },

    renderOverview: function(container, data) {
        console.log("renderOverview called with data:", data);
        const grouped = ResilienceDataManager.groupScenariosByMetric(data.shift_scenarios);
        console.log("Grouped scenarios:", grouped);

        let html = `
                <div class="row">
                    <div class="col-4">
                        <div class="metric-card">
                            <div class="metric-label">Model</div>
                            <div class="metric-value">${data.model_name}</div>
                            <div class="metric-subtitle">${data.model_type}</div>
                        </div>
                    </div>
                    <div class="col-4">
                        <div class="metric-card">
                            <div class="metric-label">Resilience Score</div>
                            <div class="metric-value">${ResilienceDataManager.formatNumber(data.resilience_score, 4)}</div>
                            <div class="metric-subtitle">Overall assessment</div>
                        </div>
                    </div>
                    <div class="col-4">
                        <div class="metric-card">
                            <div class="metric-label">Scenarios Tested</div>
                            <div class="metric-value">${data.shift_scenarios.length}</div>
                            <div class="metric-subtitle">Distribution shifts</div>
                        </div>
                    </div>
                </div>

                <h4 class="subsection-title">Performance by Distance Metric</h4>
                <div class="metrics-list">
        `;

        Object.keys(grouped).forEach(metric => {
            const stats = ResilienceDataManager.calculateScenarioStats(grouped[metric]);

            html += `
                <div class="metric-item">
                    <div class="metric-header">
                        <span class="metric-name">${metric}</span>
                        <span class="metric-count">${stats.valid_count}/${stats.count} scenarios</span>
                    </div>
                    <div class="metric-details">
                        <span>Avg Gap: <strong>${ResilienceDataManager.formatNumber(stats.mean, 4)}</strong></span>
                        <span>Min: <strong class="success">${ResilienceDataManager.formatNumber(stats.min, 4)}</strong></span>
                        <span>Max: <strong class="error">${ResilienceDataManager.formatNumber(stats.max, 4)}</strong></span>
                    </div>
                </div>
            `;
        });

        html += `
                </div>

                <div class="info-box">
                    <p><strong>About Resilience:</strong> Tests how well the model maintains performance
                    when data distributions shift. Lower performance gaps indicate better resilience.</p>
                </div>
        `;

        console.log("Setting innerHTML with generated HTML");
        container.innerHTML = html;
        console.log("Overview rendered successfully");
    },

    showNoDataMessage: function(container, message) {
        container.innerHTML = `
            <div class="warning-box">
                <p>${message}</p>
            </div>
        `;
    }
};

if (typeof window !== 'undefined') {
    window.ResilienceOverviewChartManager = ResilienceOverviewChartManager;
    window.ChartManager = ResilienceOverviewChartManager;
}
