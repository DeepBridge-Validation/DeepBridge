// Safe initialization for reports
document.addEventListener('DOMContentLoaded', function() {
    console.log("Initializing report with safe mode");
    
    // Initialize tabs
    initTabs();
    
    // Initialize charts safely
    initCharts();
    
    // Initialize standalone fixed components
    initStandaloneComponents();
});

function initTabs() {
    try {
        const tabButtons = document.querySelectorAll('.tab-btn');
        if (tabButtons.length > 0) {
            tabButtons.forEach(button => {
                button.addEventListener('click', function() {
                    const targetTab = this.getAttribute('data-tab');
                    showTab(targetTab, this);
                });
            });
            
            // Show first tab by default (or one specified in URL hash)
            const hash = window.location.hash;
            if (hash && hash.length > 1) {
                const tabId = hash.substring(1);
                const tabButton = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
                if (tabButton) {
                    tabButton.click();
                } else {
                    tabButtons[0].click();
                }
            } else {
                tabButtons[0].click();
            }
        }
    } catch (error) {
        console.error("Error initializing tabs:", error);
    }
}

function showTab(tabId, buttonElement) {
    try {
        // Hide all tabs and remove active class from buttons
        document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        
        // Show selected tab and mark button as active
        const tabElement = document.getElementById(tabId);
        if (tabElement) {
            tabElement.classList.add('active');
            buttonElement.classList.add('active');
            
            // Update URL hash for bookmarking
            window.location.hash = tabId;
            
            // Trigger chart initialization for this tab
            initChartsForTab(tabId);
        }
    } catch (error) {
        console.error("Error showing tab:", error);
    }
}

function initCharts() {
    try {
        // First try the regular way
        if (typeof initializeCharts === 'function') {
            initializeCharts();
        } else if (typeof window.ChartManager !== 'undefined') {
            // Initialize charts using ChartManager
            initializeChartsWithManager();
        } else if (typeof Plotly !== 'undefined') {
            // Check current active tab
            const activeTab = document.querySelector('.tab-content.active');
            if (activeTab) {
                initChartsForTab(activeTab.id);
            }
        } else {
            console.warn("No chart library detected - charts may not render");
        }
    } catch (error) {
        console.error("Error initializing charts:", error);
    }
}

function initializeChartsWithManager() {
    try {
        if (typeof window.ChartManager === 'undefined') return;
        
        // Initialize perturbation chart
        const perturbationElement = document.getElementById('perturbation-chart');
        if (perturbationElement) {
            window.ChartManager.initializePerturbationChart('perturbation-chart');
        }
        
        // Initialize model comparison chart
        const modelComparisonElement = document.getElementById('model-comparison-chart');
        if (modelComparisonElement) {
            window.ChartManager.initializeModelComparisonChart('model-comparison-chart');
        }
        
        // Initialize worst score chart
        const worstScoreElement = document.getElementById('worst-score-chart');
        if (worstScoreElement) {
            window.ChartManager.initializeWorstScoreChart('worst-score-chart');
        }
        
        // Initialize mean score chart
        const meanScoreElement = document.getElementById('mean-score-chart');
        if (meanScoreElement) {
            window.ChartManager.initializeMeanScoreChart('mean-score-chart');
        }
        
        // Initialize model level details chart
        const modelLevelDetailsElement = document.getElementById('model-level-details-chart');
        if (modelLevelDetailsElement) {
            window.ChartManager.initializeModelLevelDetailsChart('model-level-details-chart');
        }
    } catch (error) {
        console.error("Error initializing charts with ChartManager:", error);
    }
}

function initChartsForTab(tabId) {
    setTimeout(function() {
        try {
            if (tabId === 'overview') {
                // Overview tab charts
                const perturbationElement = document.getElementById('perturbation-chart');
                if (perturbationElement && window.ChartManager) {
                    window.ChartManager.initializePerturbationChart('perturbation-chart');
                }
                
                const modelComparisonElement = document.getElementById('model-comparison-chart');
                if (modelComparisonElement && window.ChartManager) {
                    window.ChartManager.initializeModelComparisonChart('model-comparison-chart');
                }
            } else if (tabId === 'boxplot') {
                // Boxplot tab charts
                const boxplotElement = document.getElementById('boxplot-chart');
                if (boxplotElement && typeof initializeBoxplotChart === 'function') {
                    initializeBoxplotChart();
                }
            } else if (tabId === 'feature_impact') {
                // Feature impact tab
                const featureElement = document.getElementById('feature-importance-chart');
                if (featureElement && window.StandaloneFeatureImportanceChart) {
                    window.StandaloneFeatureImportanceChart.initialize('feature-importance-chart');
                }
            } else if (tabId === 'importance_comparison') {
                // Importance comparison tab - usar o ImportanceComparisonHandler específico
                if (window.ImportanceComparisonHandler && 
                    typeof window.ImportanceComparisonHandler.initialize === 'function') {
                    console.log("Initializing importance comparison chart using ImportanceComparisonHandler");
                    window.ImportanceComparisonHandler.initialize();
                }
            }
        } catch (error) {
            console.error(`Error initializing charts for tab ${tabId}:`, error);
        }
    }, 100); // Small delay to ensure the tab is fully visible
}

function initStandaloneComponents() {
    // Initialize any standalone components specific to this report
    try {
        // Initialize the safe perturbation results controller if it exists
        if (window.SafePerturbationResultsController && document.getElementById('perturbation-results-container')) {
            window.SafePerturbationResultsController.init();
        }
    } catch (error) {
        console.error("Error initializing standalone components:", error);
    }
}