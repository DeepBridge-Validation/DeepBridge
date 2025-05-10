/**
 * Improved initialization for robustness reports 
 * Fixes chart initialization issues with centralized rendering
 */

// Global chart initialization state to prevent duplicate rendering
window.chartInitialized = {
    features: false,
    overview: false,
    boxplot: false,
    details: false
};

// Central chart management
window.ChartInitializer = {
    // Check if Plotly is loaded and load if necessary
    ensurePlotlyLoaded: function(callback) {
        if (typeof Plotly !== 'undefined') {
            // Plotly already loaded, execute callback immediately
            if (callback) callback();
            return true;
        }
        
        console.log("Plotly not loaded, attempting to load from CDN");
        
        // Create script element to load Plotly
        const script = document.createElement('script');
        script.src = 'https://cdn.plot.ly/plotly-2.29.1.min.js';
        script.async = true;
        
        script.onload = function() {
            console.log("Plotly loaded successfully");
            if (callback) callback();
        };
        
        script.onerror = function() {
            console.error("Failed to load Plotly from CDN");
            document.querySelectorAll('.chart-plot').forEach(container => {
                container.innerHTML = `
                    <div style="padding: 20px; text-align: center;">
                        <div style="color: #e53935; margin-bottom: 10px;">⚠️ Chart library could not be loaded</div>
                        <div style="color: #555; font-size: 14px;">Please check your internet connection and refresh the page.</div>
                    </div>`;
            });
        };
        
        document.head.appendChild(script);
        return false;
    },
    
    // Initialize feature charts, with validation
    initializeFeatureCharts: function(sortBy) {
        // Allow forced reinitialization with new sort order
        console.log("Initializing feature charts" + (sortBy ? ` with sort: ${sortBy}` : ""));
        
        // Check data availability in both reportData and chart_data_json
        const hasFeatureData = (window.reportData && 
                             window.reportData.feature_importance && 
                             Object.keys(window.reportData.feature_importance).length > 0) || 
                            (window.reportData && 
                             window.reportData.chart_data_json && 
                             typeof window.reportData.chart_data_json === 'string' &&
                             window.reportData.chart_data_json.includes('feature_importance'));
        
        if (!hasFeatureData) {
            console.warn("No feature importance data available for charts");
            this.showNoDataForCharts('feature-importance-chart');
            this.showNoDataForCharts('importance-comparison-chart-plot');
            return;
        }
        
        this.ensurePlotlyLoaded(() => {
            // Clear any previous content and force redraw
            const chartContainer = document.getElementById('feature-importance-chart');
            if (chartContainer) {
                // Clear previous chart
                Plotly.purge(chartContainer);
                chartContainer.innerHTML = '';
                
                if (typeof FeatureImportanceChartManager !== 'undefined' && 
                    typeof FeatureImportanceChartManager.initializeFeatureImportanceChart === 'function') {
                    FeatureImportanceChartManager.initializeFeatureImportanceChart('feature-importance-chart');
                } else {
                    console.error("FeatureImportanceChartManager not available");
                    this.showErrorForCharts('feature-importance-chart', "Chart manager not available");
                }
            } else {
                console.warn("Feature chart container not found");
            }
            
            // Initialize comparison chart if container exists
            const comparisonContainer = document.getElementById('importance-comparison-chart-plot');
            if (comparisonContainer) {
                // Clear previous chart
                Plotly.purge(comparisonContainer);
                comparisonContainer.innerHTML = '';
                
                if (typeof FeatureImportanceChartManager !== 'undefined' && 
                    typeof FeatureImportanceChartManager.initializeImportanceComparisonChart === 'function') {
                    try {
                        FeatureImportanceChartManager.initializeImportanceComparisonChart('importance-comparison-chart-plot');
                    } catch (e) {
                        console.error("Error initializing importance comparison chart:", e);
                        // Show graceful error
                        comparisonContainer.innerHTML = `
                            <div style="padding: 30px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto;">
                                <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                                <h3 style="font-size: 18px; margin-bottom: 10px; color: #e53935;">Chart Error</h3>
                                <p style="color: #666; font-size: 14px;">${e.message || "Error rendering chart"}</p>
                            </div>`;
                    }
                }
            }
            
            window.chartInitialized.features = true;
        });
    },
    
    // Initialize overview charts
    initializeOverviewCharts: function() {
        if (window.chartInitialized.overview) return;
        
        console.log("Initializing overview charts");
        
        this.ensurePlotlyLoaded(() => {
            if (typeof ChartManager !== 'undefined') {
                // Clear any previous error messages
                const chartContainers = document.querySelectorAll('.overview-chart');
                chartContainers.forEach(container => {
                    container.innerHTML = '';
                });
                
                // Check if data is available
                const hasOverviewData = window.reportData && 
                                       (window.reportData.raw || window.reportData.perturbation_chart_data);
                
                if (!hasOverviewData) {
                    console.warn("No overview data available for charts");
                    this.showNoDataForCharts('perturbation-chart-plot');
                    this.showNoDataForCharts('worst-score-chart-plot');
                    this.showNoDataForCharts('mean-score-chart-plot');
                    return;
                }
                
                // Initialize each chart with try-catch for robustness
                try {
                    if (typeof ChartManager.initializePerturbationChart === 'function') {
                        ChartManager.initializePerturbationChart('perturbation-chart-plot');
                    }
                } catch (e) {
                    console.error("Error initializing perturbation chart:", e);
                    this.showErrorForCharts('perturbation-chart-plot', e.message);
                }
                
                try {
                    if (typeof ChartManager.initializeWorstScoreChart === 'function') {
                        ChartManager.initializeWorstScoreChart('worst-score-chart-plot');
                    }
                } catch (e) {
                    console.error("Error initializing worst score chart:", e);
                    this.showErrorForCharts('worst-score-chart-plot', e.message);
                }
                
                try {
                    if (typeof ChartManager.initializeMeanScoreChart === 'function') {
                        ChartManager.initializeMeanScoreChart('mean-score-chart-plot');
                    }
                } catch (e) {
                    console.error("Error initializing mean score chart:", e);
                    this.showErrorForCharts('mean-score-chart-plot', e.message);
                }
                
                // Only initialize model comparison charts if we have alternative models
                const hasAlternativeModels = window.reportData && 
                                            window.reportData.alternative_models &&
                                            Object.keys(window.reportData.alternative_models).length > 0;
                                            
                if (hasAlternativeModels) {
                    try {
                        if (typeof ChartManager.initializeModelComparisonChart === 'function') {
                            ChartManager.initializeModelComparisonChart('model-comparison-chart-plot');
                        }
                    } catch (e) {
                        console.error("Error initializing model comparison chart:", e);
                        this.showErrorForCharts('model-comparison-chart-plot', e.message);
                    }
                    
                    try {
                        if (typeof ChartManager.initializeModelLevelDetailsChart === 'function') {
                            ChartManager.initializeModelLevelDetailsChart('model-level-details-chart-plot');
                        }
                    } catch (e) {
                        console.error("Error initializing model level details chart:", e);
                        this.showErrorForCharts('model-level-details-chart-plot', e.message);
                    }
                } else {
                    console.log("No alternative models available, skipping comparison charts");
                }
            } else {
                console.error("ChartManager not available");
                document.querySelectorAll('.overview-chart').forEach(container => {
                    this.showErrorForCharts(container.id, "Chart manager not available");
                });
            }
            
            window.chartInitialized.overview = true;
        });
    },
    
    // Initialize boxplot charts
    initializeBoxplotCharts: function() {
        if (window.chartInitialized.boxplot) return;
        
        console.log("Initializing boxplot charts");
        
        // Check if boxplot data is available
        const hasBoxplotData = window.reportData && 
                              (window.reportData.boxplot_data || 
                              (window.reportData.raw && window.reportData.raw.by_level));
        
        if (!hasBoxplotData) {
            console.warn("No boxplot data available");
            this.showNoDataForCharts('boxplot-chart-container');
            return;
        }
        
        // First make sure the container is visible and has dimensions
        const boxplotContainer = document.getElementById('boxplot-chart-container');
        if (boxplotContainer) {
            // Force a minimum size to ensure rendering works
            boxplotContainer.style.minHeight = '400px';
            boxplotContainer.style.minWidth = '100%';
            
            this.ensurePlotlyLoaded(() => {
                if (typeof BoxplotChartManager !== 'undefined' && 
                    typeof BoxplotChartManager.initializeBoxplotChart === 'function') {
                    try {
                        BoxplotChartManager.initializeBoxplotChart('boxplot-chart-container');
                    } catch (e) {
                        console.error("Error initializing boxplot chart:", e);
                        this.showErrorForCharts('boxplot-chart-container', e.message);
                    }
                } else {
                    console.error("BoxplotChartManager not available");
                    this.showErrorForCharts('boxplot-chart-container', "Chart manager not available");
                }
            });
        } else {
            console.warn("Boxplot container not found");
        }
        
        window.chartInitialized.boxplot = true;
    },
    
    // Utility function to show no data message
    showNoDataForCharts: function(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        element.innerHTML = `
            <div style="padding: 30px; text-align: center; background-color: #f8f9fa; border-radius: 8px; margin: 20px auto;">
                <div style="font-size: 48px; margin-bottom: 10px;">📊</div>
                <h3 style="font-size: 18px; margin-bottom: 10px;">No Chart Data Available</h3>
                <p style="color: #666; font-size: 14px;">The required data for this chart is not available.</p>
            </div>`;
    },
    
    // Utility function to show error message
    showErrorForCharts: function(elementId, message) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        element.innerHTML = `
            <div style="padding: 30px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto;">
                <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                <h3 style="font-size: 18px; margin-bottom: 10px; color: #e53935;">Chart Error</h3>
                <p style="color: #666; font-size: 14px;">${message || "Error rendering chart"}</p>
            </div>`;
    }
};

// Tab change handler to ensure charts are rendered when tabs become visible
function handleTabChange(tabId) {
    // Reset layout for any charts in this tab
    window.dispatchEvent(new Event('resize'));
    
    // Initialize appropriate charts based on tab ID
    switch (tabId) {
        case 'overview':
            window.ChartInitializer.initializeOverviewCharts();
            break;
        case 'feature_impact':
            window.ChartInitializer.initializeFeatureCharts();
            // Check if the table controller is initialized
            if (typeof FeatureImportanceTableController !== 'undefined' && 
                typeof FeatureImportanceTableController.init === 'function') {
                console.log("Ensuring feature table is initialized");
                FeatureImportanceTableController.init();
            } else if (typeof FeatureImportanceController !== 'undefined' && 
                     typeof FeatureImportanceController.fillFeatureImportanceTable === 'function') {
                console.log("Using fallback feature table initialization");
                FeatureImportanceController.fillFeatureImportanceTable();
            }
            break;
        case 'boxplot':
            window.ChartInitializer.initializeBoxplotCharts();
            break;
        case 'importance_comparison':
            // Deixamos o ImportanceComparisonHandler cuidar desta inicialização
            // para evitar conflitos de renderização múltipla
            console.log("Importance comparison tab activated - using ImportanceComparisonHandler");
            // Não inicializamos aqui, deixamos o ImportanceComparisonHandler fazer isso
            break;
    }
    
    // Broadcast tab change event for other components
    document.dispatchEvent(new CustomEvent('tabchange', { 
        detail: { tabId: tabId } 
    }));
}

// Initialize on DOM ready with improved structure
document.addEventListener('DOMContentLoaded', function() {
    console.log("Report initialized");
    
    // Carregar script para corrigir erros de continue
    try {
        // Verificar se o script de correção já foi carregado
        if (!window.fixedSyntaxLoaded) {
            console.log("Carregando correção para erros de 'continue' fora de loops");
            window.fixedSyntaxLoaded = true;
            
            // Carregar o script fixed_syntax.js
            const fixScript = document.createElement('script');
            fixScript.src = 'js/fixed_syntax.js';
            fixScript.onload = function() {
                console.log("Script de correção carregado com sucesso");
            };
            fixScript.onerror = function(e) {
                console.error("Erro ao carregar script de correção:", e);
            };
            document.head.appendChild(fixScript);
        }
    } catch (e) {
        console.error("Erro ao configurar correção de sintaxe:", e);
    }
    
    // Set up tab navigation with chart initialization on tab change
    setupTabNavigation();
    
    // Initialize first tab's charts
    const initialTab = document.querySelector('.tab-btn.active');
    if (initialTab) {
        const tabId = initialTab.getAttribute('data-tab');
        handleTabChange(tabId);
    } else {
        // Default to overview if no tab is active
        window.ChartInitializer.initializeOverviewCharts();
    }
    
    // Initialize controllers with error handling
    initializeControllers();
});

// Set up tab navigation
function setupTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    if (tabButtons.length === 0) {
        console.warn("No tab buttons found");
        return;
    }
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Show target tab content
            const targetTab = this.getAttribute('data-tab');
            const targetElement = document.getElementById(targetTab);
            
            if (targetElement) {
                targetElement.classList.add('active');
                
                // Initialize charts for this tab
                handleTabChange(targetTab);
            } else {
                console.error(`Tab content not found: #${targetTab}`);
            }
        });
    });
    
    // Activate first tab by default if none is active
    if (!document.querySelector('.tab-btn.active')) {
        tabButtons[0]?.click();
    }
}

// Initialize controllers with error handling
function initializeControllers() {
    // Define controllers to initialize
    const controllers = [
        { name: 'MainController', initializer: function() {
            if (typeof MainController !== 'undefined' && typeof MainController.init === 'function') {
                MainController.init();
            }
        }},
        { name: 'OverviewController', initializer: function() {
            if (typeof OverviewController !== 'undefined' && typeof OverviewController.init === 'function') {
                OverviewController.init();
            }
        }},
        { name: 'DetailsController', initializer: function() {
            if (typeof DetailsController !== 'undefined' && typeof DetailsController.init === 'function') {
                DetailsController.init();
            }
        }},
        { name: 'BoxplotController', initializer: function() {
            if (typeof BoxplotController !== 'undefined' && typeof BoxplotController.init === 'function') {
                BoxplotController.init();
            }
        }},
        { name: 'FeatureImportanceController', initializer: function() {
            if (typeof FeatureImportanceController !== 'undefined' && typeof FeatureImportanceController.init === 'function') {
                FeatureImportanceController.init();
            } else {
                console.log("FeatureImportanceController not found, using fallback initialization");
                // Try direct chart initialization if controller not available
                window.ChartInitializer.initializeFeatureCharts();
            }
        }},
        { name: 'FeatureImportanceTableController', initializer: function() {
            if (typeof FeatureImportanceTableController !== 'undefined' && typeof FeatureImportanceTableController.init === 'function') {
                FeatureImportanceTableController.init();
            }
        }}
    ];

    // Add additional initialization for results tabs
    const initResultTabs = function() {
        const resultTabsContainer = document.getElementById('result_tables_tabs');
        if (resultTabsContainer) {
            const tabButtons = resultTabsContainer.querySelectorAll('.chart-selector-option');
            if (tabButtons.length > 0) {
                tabButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        // Remove active class from all buttons
                        tabButtons.forEach(btn => btn.classList.remove('active'));
                        // Add active class to clicked button
                        this.classList.add('active');

                        // Hide all tab contents
                        const tabContents = document.querySelectorAll('.results-tables-section .tab-content');
                        tabContents.forEach(content => content.classList.remove('active'));

                        // Show target tab content
                        const targetTab = this.getAttribute('data-tab');
                        const targetContent = document.getElementById(targetTab);
                        if (targetContent) {
                            targetContent.classList.add('active');
                        }
                    });
                });
            }
        }
    };

    // Add the results tabs initializer to controllers
    controllers.push({
        name: 'ResultTabsInitializer',
        initializer: initResultTabs
    });
    
    // Initialize each controller with error handling
    controllers.forEach(controller => {
        try {
            controller.initializer();
        } catch (error) {
            console.error(`Error initializing ${controller.name}:`, error);
        }
    });
}

// Simple MainController implementation (keep backward compatibility)
const MainController = {
    init: function() {
        console.log("MainController initialized with enhanced initialization");
        // Tab navigation is now handled by setupTabNavigation function
    }
};