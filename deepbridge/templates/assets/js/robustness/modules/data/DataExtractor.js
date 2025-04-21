// DataExtractor module for extracting data from report_data
const DataExtractor = {
    getRawData: function() {
        if (\!window.reportData || \!window.reportData.raw) {
            console.error("Raw data not available in report data");
            return {};
        }
        return window.reportData.raw;
    },
    
    getQuantileData: function() {
        if (\!window.reportData || \!window.reportData.quantile) {
            console.error("Quantile data not available in report data");
            return {};
        }
        return window.reportData.quantile;
    },
    
    getMetadata: function() {
        if (\!window.reportData) {
            console.error("Report data not available");
            return {};
        }
        
        return {
            model_name: window.reportData.model_name || "Unknown Model",
            model_type: window.reportData.model_type || "Unknown Type",
            metric: window.reportData.metric || "score",
            base_score: window.reportData.base_score || 0,
            robustness_score: window.reportData.robustness_score || 0
        };
    }
};
