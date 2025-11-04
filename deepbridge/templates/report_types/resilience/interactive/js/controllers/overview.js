// ResilienceOverviewController.js - Simple controller for resilience overview
const ResilienceOverviewController = {
    init: function() {
        console.log("Initializing ResilienceOverviewController");

        try {
            if (typeof ResilienceDataManager === 'undefined' || typeof ResilienceOverviewChartManager === 'undefined') {
                console.error("Required managers not available");
                return;
            }

            if (!ResilienceDataManager.hasData()) {
                console.warn("No resilience data available");
                return;
            }

            ResilienceOverviewChartManager.initializeResilienceChart('overview-chart');
        } catch (error) {
            console.error("Error initializing overview:", error);
        }
    }
};

if (typeof window !== 'undefined') {
    window.ResilienceOverviewController = ResilienceOverviewController;
    window.OverviewController = ResilienceOverviewController;
}
