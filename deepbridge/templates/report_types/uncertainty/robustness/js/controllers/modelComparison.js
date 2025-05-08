// Model Comparison Controller
const ModelComparisonController = {
    init: function() {
        console.log("Model Comparison section initialized");
        
        // Initialize tab navigation
        this.initTabNavigation();
        
        // Initialize metric selector if it exists
        this.initMetricSelector();
        
        // Initialize highlight toggle
        this.initHighlightToggle();
        
        // Initialize row expansion functionality
        this.initRowExpansion();
        
        // Initialize charts for robustness visualization
        this.initRobustnessCharts();
    },
    
    initTabNavigation: function() {
        const tabButtons = document.querySelectorAll('.model-comparison-tab');
        if (!tabButtons.length) return;
        
        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                // Remove active from all tabs
                tabButtons.forEach(tab => tab.classList.remove('active'));
                
                // Add active to clicked tab
                e.currentTarget.classList.add('active');
                
                // Show corresponding content
                const tabId = e.currentTarget.getAttribute('data-tab');
                this.showTabContent(tabId);
            });
        });
        
        // Show first tab by default
        tabButtons[0].click();
    },
    
    showTabContent: function(tabId) {
        // Hide all tab content
        const tabContents = document.querySelectorAll('.model-comparison-content');
        tabContents.forEach(content => content.classList.remove('active'));
        
        // Show selected tab content
        const selectedContent = document.getElementById(tabId);
        if (selectedContent) {
            selectedContent.classList.add('active');
        }
        
        // Re-render charts if it's the robustness tab
        if (tabId === 'robustness-tab') {
            this.initRobustnessCharts();
        }
    },
    
    initMetricSelector: function() {
        const metricSelector = document.getElementById('metric-selector');
        if (!metricSelector) return;
        
        metricSelector.addEventListener('change', (e) => {
            const metric = e.target.value;
            ModelComparisonManager.updateMetricsDisplay(metric);
        });
    },
    
    initHighlightToggle: function() {
        const highlightToggle = document.getElementById('highlight-best-toggle');
        if (!highlightToggle) return;
        
        highlightToggle.addEventListener('change', (e) => {
            const highlightBest = e.target.checked;
            ModelComparisonManager.setHighlightBest(highlightBest);
            
            // Re-render active table
            const activeTab = document.querySelector('.model-comparison-tab.active');
            if (activeTab) {
                const tabId = activeTab.getAttribute('data-tab');
                ModelComparisonManager.renderTable(tabId);
            }
        });
    },
    
    initRowExpansion: function() {
        // Use event delegation for row expansion since rows might be re-rendered
        const tableContainer = document.querySelector('.model-comparison-container');
        if (!tableContainer) return;
        
        tableContainer.addEventListener('click', (e) => {
            // Check if the click is on a model row (not on expanded content)
            const modelRow = e.target.closest('tr[data-model-key]');
            if (!modelRow) return;
            
            const modelKey = modelRow.getAttribute('data-model-key');
            ModelComparisonManager.toggleRowExpansion(modelKey);
        });
    },
    
    initRobustnessCharts: function() {
        console.log("Initializing robustness comparison charts");
        
        setTimeout(() => {
            if (typeof Plotly !== 'undefined') {
                ModelComparisonManager.renderPerturbationChart('perturbation-comparison-chart');
            } else {
                this.showChartError();
            }
        }, 500);
    },
    
    showChartError: function() {
        const chartContainers = document.querySelectorAll('.chart-container');
        chartContainers.forEach(container => {
            container.innerHTML = "<div style='padding: 20px; text-align: center; color: red;'>Plotly library not loaded. Charts cannot be displayed.</div>";
        });
    },
    
    refreshAllTables: function() {
        // Get currently active tab
        const activeTab = document.querySelector('.model-comparison-tab.active');
        if (activeTab) {
            const tabId = activeTab.getAttribute('data-tab');
            ModelComparisonManager.renderTable(tabId);
        }
    }
};