/**
 * Feature Importance Controller
 * Handles feature importance visualization and data management
 */
window.FeatureImportanceController = {
    /**
     * Initialize the controller
     */
    init: function() {
        console.log("FeatureImportanceController initialized");
        
        // Initialize the feature importance table
        this.initializeFeatureTable();
        
        // Set up event listeners for feature table interactions
        this.setupTableEvents();
        
        // Initialize feature charts when tab is active
        document.addEventListener('tabchange', (event) => {
            if (event.detail && event.detail.tabId === 'feature_impact') {
                console.log("Feature impact tab activated");
                this.refreshFeatureData();
            }
        });
    },
    
    /**
     * Initialize the feature importance table
     */
    initializeFeatureTable: function() {
        console.log("Initializing feature importance table");
        this.fillFeatureImportanceTable();
        
        // Update feature counts
        this.updateFeatureCounts();
    },
    
    /**
     * Set up event listeners for table interactions
     */
    setupTableEvents: function() {
        // Sort table when headers are clicked
        const headers = document.querySelectorAll('.feature-importance-table th.sortable');
        if (headers.length > 0) {
            headers.forEach(header => {
                header.addEventListener('click', (e) => {
                    const sortBy = e.currentTarget.getAttribute('data-sort');
                    this.sortTable(sortBy);
                });
            });
        }
        
        // Search feature table
        const searchInput = document.getElementById('feature-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.filterTable();
            });
        }
        
        // Toggle feature subset display
        const subsetToggle = document.getElementById('show-subset-only');
        if (subsetToggle) {
            subsetToggle.addEventListener('change', (e) => {
                this.filterTable();
            });
        }
    },
    
    /**
     * Fill the feature importance table with data
     */
    fillFeatureImportanceTable: function() {
        console.log("Filling feature importance table");
        
        // Get feature data from the table manager
        const featureData = FeatureImportanceTableManager.extractFeatureData();
        
        if (featureData.length === 0) {
            console.warn("No feature importance data available");
            this.showNoDataMessage();
            return;
        }
        
        // Sort by default column (impact/robustness)
        const sortedData = FeatureImportanceTableManager.sortData(featureData, 'impact', 'desc');
        
        // Generate table HTML
        const tableHTML = FeatureImportanceTableManager.generateTableRows(sortedData);
        
        // Update table content
        const tableBody = document.getElementById('feature-impact-data');
        if (tableBody) {
            tableBody.innerHTML = tableHTML;
        }
        
        // Log success
        console.log(`Feature table populated with ${featureData.length} features`);
    },
    
    /**
     * Show a message when no data is available
     */
    showNoDataMessage: function() {
        const tableBody = document.getElementById('feature-impact-data');
        if (tableBody) {
            tableBody.innerHTML = FeatureImportanceTableManager.generateNoDataMessage();
        }
    },
    
    /**
     * Sort the feature table by the specified column
     * @param {string} sortBy - Column to sort by
     */
    sortTable: function(sortBy) {
        // Reset sort indicators on all header cells
        document.querySelectorAll('.feature-importance-table th.sortable .sort-indicator')
            .forEach(indicator => indicator.textContent = '');
        
        // Set sort indicator on active column
        const activeHeader = document.querySelector(`.feature-importance-table th[data-sort="${sortBy}"]`);
        if (activeHeader) {
            activeHeader.querySelector('.sort-indicator').textContent = '▼';
        }
        
        // Get and sort feature data
        const featureData = FeatureImportanceTableManager.extractFeatureData();
        const sortedData = FeatureImportanceTableManager.sortData(featureData, sortBy, 'desc');
        
        // Generate and update table HTML
        const tableBody = document.getElementById('feature-impact-data');
        if (tableBody) {
            tableBody.innerHTML = FeatureImportanceTableManager.generateTableRows(sortedData);
        }
        
        console.log(`Table sorted by ${sortBy}`);
    },
    
    /**
     * Filter the feature table by search term and subset option
     */
    filterTable: function() {
        const searchTerm = document.getElementById('feature-search').value;
        const showOnlySubset = document.getElementById('show-subset-only').checked;
        
        // Get feature data
        const featureData = FeatureImportanceTableManager.extractFeatureData();
        
        // Filter data
        const filteredData = FeatureImportanceTableManager.filterData(featureData, searchTerm, showOnlySubset);
        
        // Get current sort column
        const activeHeader = document.querySelector('.feature-importance-table th.sortable .sort-indicator');
        const sortBy = activeHeader && activeHeader.parentElement.getAttribute('data-sort') || 'impact';
        
        // Sort filtered data
        const sortedData = FeatureImportanceTableManager.sortData(filteredData, sortBy, 'desc');
        
        // Update table
        const tableBody = document.getElementById('feature-impact-data');
        if (tableBody) {
            tableBody.innerHTML = FeatureImportanceTableManager.generateTableRows(sortedData);
        }
        
        console.log(`Table filtered with ${filteredData.length} of ${featureData.length} features visible`);
        
        // Update feature counts to reflect filtered data
        this.updateFeatureCountsWithData(filteredData, featureData);
    },
    
    /**
     * Update feature count displays in the UI
     */
    updateFeatureCounts: function() {
        const featureData = FeatureImportanceTableManager.extractFeatureData();
        this.updateFeatureCountsWithData(featureData, featureData);
    },
    
    /**
     * Update feature count displays with specific data
     * @param {Array} visibleData - Currently visible feature data
     * @param {Array} totalData - All feature data
     */
    updateFeatureCountsWithData: function(visibleData, totalData) {
        const totalFeaturesElement = document.getElementById('total-features-count');
        const subsetFeaturesElement = document.getElementById('subset-features-count');
        
        if (totalFeaturesElement) {
            totalFeaturesElement.textContent = visibleData.length;
        }
        
        if (subsetFeaturesElement) {
            const subsetCount = visibleData.filter(item => item.inSubset).length;
            subsetFeaturesElement.textContent = subsetCount;
        }
    },
    
    /**
     * Refresh feature data and update visualizations
     */
    refreshFeatureData: function() {
        // Reinitialize table with fresh data
        this.fillFeatureImportanceTable();
        
        // Also refresh charts if needed
        if (typeof window.ChartInitializer !== 'undefined' && 
            typeof window.ChartInitializer.initializeFeatureCharts === 'function') {
            window.ChartInitializer.initializeFeatureCharts();
        }
    }
};

// Manual initialization to ensure it runs
document.addEventListener('DOMContentLoaded', function() {
    // Delay initialization slightly to ensure other dependencies are loaded
    setTimeout(function() {
        if (typeof FeatureImportanceController !== 'undefined' && 
            typeof FeatureImportanceController.init === 'function') {
            FeatureImportanceController.init();
        }
    }, 100);
});