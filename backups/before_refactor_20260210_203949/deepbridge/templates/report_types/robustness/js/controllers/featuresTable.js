// Feature Importance Table Controller
window.FeatureImportanceTableController = {
    // State management
    state: {
        sortConfig: { key: 'impact', direction: 'desc' },
        searchTerm: '',
        showOnlySubset: false,
        hoveredRow: null,
        allData: [],
        filteredData: []
    },
    
    /**
     * Initialize the feature importance table
     */
    init: function() {
        console.log("Feature Importance Table initialized");
        
        // Get table elements
        this.tableElement = document.querySelector('.feature-importance-table');
        this.tableBody = document.getElementById('feature-impact-data');
        this.tableFooter = document.querySelector('.feature-importance-footer');
        this.searchInput = document.getElementById('feature-search');
        this.subsetToggle = document.getElementById('show-subset-only');
        this.totalCountEl = document.getElementById('total-features-count');
        this.subsetCountEl = document.getElementById('subset-features-count');
        
        // Check if required elements exist
        if (!this.tableBody) {
            console.error("Required table elements not found");
            return;
        }
        
        // Load initial data
        this.loadData();
        
        // Initialize event listeners
        this.initEventListeners();
        
        // Render the table
        this.renderTable();
    },
    
    /**
     * Load feature data from the report
     */
    loadData: function() {
        try {
            // Extract data using the manager
            this.state.allData = FeatureImportanceTableManager.extractFeatureData();
            
            // Apply initial sorting and filtering
            this.updateFilteredData();
            
            console.log(`Loaded ${this.state.allData.length} features in controller`);
        } catch (error) {
            console.error("Error loading feature data:", error);
            this.state.allData = [];
            this.state.filteredData = [];
        }
    },
    
    /**
     * Set up event listeners for the table
     */
    initEventListeners: function() {
        // Set up column header sorting
        if (this.tableElement) {
            const headers = this.tableElement.querySelectorAll('th.sortable');
            headers.forEach(header => {
                const key = header.getAttribute('data-sort');
                if (key) {
                    header.addEventListener('click', () => this.handleSort(key));
                }
            });
        }
        
        // Set up search input
        if (this.searchInput) {
            this.searchInput.addEventListener('input', (e) => {
                this.state.searchTerm = e.target.value;
                this.updateFilteredData();
                this.renderTable();
            });
        }
        
        // Set up subset toggle
        if (this.subsetToggle) {
            this.subsetToggle.addEventListener('change', (e) => {
                this.state.showOnlySubset = e.target.checked;
                this.updateFilteredData();
                this.renderTable();
            });
        }
        
        // Set up row hover effects (using event delegation)
        if (this.tableBody) {
            this.tableBody.addEventListener('mouseover', (e) => {
                const row = e.target.closest('tr[data-feature]');
                if (row) {
                    const feature = row.getAttribute('data-feature');
                    this.state.hoveredRow = feature;
                    this.highlightRow(feature);
                }
            });
            
            this.tableBody.addEventListener('mouseout', () => {
                this.state.hoveredRow = null;
                this.clearRowHighlights();
            });
        }
    },
    
    /**
     * Handle column sorting
     * @param {string} key - Key to sort by
     */
    handleSort: function(key) {
        const { sortConfig } = this.state;
        const direction = 
            sortConfig.key === key && sortConfig.direction === 'desc' ? 'asc' : 'desc';
        
        this.state.sortConfig = { key, direction };
        this.updateFilteredData();
        this.renderTable();
        
        // Update sort indicators
        this.updateSortIndicators(key, direction);
    },
    
    /**
     * Update filtered and sorted data based on current state
     */
    updateFilteredData: function() {
        const { allData, sortConfig, searchTerm, showOnlySubset } = this.state;
        
        // First filter the data
        const filtered = FeatureImportanceTableManager.filterData(
            allData, 
            searchTerm, 
            showOnlySubset
        );
        
        // Then sort the filtered data
        this.state.filteredData = FeatureImportanceTableManager.sortData(
            filtered,
            sortConfig.key,
            sortConfig.direction
        );
    },
    
    /**
     * Render the table with current data
     */
    renderTable: function() {
        if (!this.tableBody) return;
        
        try {
            const { filteredData, allData, hoveredRow } = this.state;
            
            // Generate and insert rows
            this.tableBody.innerHTML = FeatureImportanceTableManager.generateTableRows(filteredData, hoveredRow);
            
            // Update feature counts
            this.updateFeatureCounts();
            
        } catch (error) {
            console.error("Error rendering feature table:", error);
            this.tableBody.innerHTML = FeatureImportanceTableManager.generateErrorMessage(error.message);
        }
    },
    
    /**
     * Update feature count displays
     */
    updateFeatureCounts: function() {
        if (this.totalCountEl && this.subsetCountEl) {
            const counts = FeatureImportanceTableManager.getFeatureCounts(this.state.allData);
            this.totalCountEl.textContent = counts.total;
            this.subsetCountEl.textContent = counts.inSubset;
        }
    },
    
    /**
     * Update sort indicators in column headers
     * @param {string} activeKey - Currently active sort key
     * @param {string} direction - Sort direction
     */
    updateSortIndicators: function(activeKey, direction) {
        if (!this.tableElement) return;
        
        const headers = this.tableElement.querySelectorAll('th.sortable');
        
        headers.forEach(header => {
            const indicator = header.querySelector('.sort-indicator');
            if (!indicator) return;
            
            // Clear all indicators
            indicator.textContent = '';
            
            const key = header.getAttribute('data-sort');
            if (key === activeKey) {
                // Set the active indicator
                indicator.textContent = direction === 'asc' ? '▲' : '▼';
            }
        });
    },
    
    /**
     * Highlight a specific row
     * @param {string} featureName - Name of feature to highlight
     */
    highlightRow: function(featureName) {
        if (!this.tableBody) return;
        
        // Remove current highlights
        this.clearRowHighlights();
        
        // Add highlight to matching row
        const row = this.tableBody.querySelector(`tr[data-feature="${featureName}"]`);
        if (row) {
            row.classList.add('hovered-row');
        }
    },
    
    /**
     * Clear all row highlights
     */
    clearRowHighlights: function() {
        if (!this.tableBody) return;
        
        const rows = this.tableBody.querySelectorAll('tr.hovered-row');
        rows.forEach(row => row.classList.remove('hovered-row'));
    }
};

// Add auto-initialization on DOM load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the controller if we're on the feature impact tab
    const featureTab = document.getElementById('feature_impact');
    if (featureTab && featureTab.classList.contains('active')) {
        console.log("Feature tab is active on load, initializing table controller");
        FeatureImportanceTableController.init();
    }
    
    // Also initialize when the feature tab is clicked
    const featureTabBtn = document.querySelector('[data-tab="feature_impact"]');
    if (featureTabBtn) {
        featureTabBtn.addEventListener('click', function() {
            console.log("Feature tab clicked, initializing table controller");
            setTimeout(function() {
                FeatureImportanceTableController.init();
            }, 100);
        });
    }
});