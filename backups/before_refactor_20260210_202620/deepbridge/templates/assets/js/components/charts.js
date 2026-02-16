/**
 * Chart utilities for DeepBridge reports
 */
const ChartUtils = {
    /**
     * Initialize Plotly if not available
     * @param {Function} callback - Callback to run after initialization
     */
    initializePlotly: function(callback) {
        if (typeof Plotly !== 'undefined') {
            if (callback) callback();
            return true;
        }
        
        console.log("Plotly not found, loading from CDN");
        const script = document.createElement('script');
        script.src = 'https://cdn.plot.ly/plotly-latest.min.js';
        
        if (callback) {
            script.onload = callback;
        }
        
        script.onerror = () => {
            console.error("Failed to load Plotly from CDN");
            this.showChartError();
        };
        
        document.head.appendChild(script);
        return false;
    },
    
    /**
     * Show error when chart library fails to load
     */
    showChartError: function() {
        const chartContainers = document.querySelectorAll('.chart-plot, .chart-container');
        chartContainers.forEach(container => {
            container.innerHTML = `
                <div class="chart-error">
                    <p>Chart rendering library not available.</p>
                    <p class="text-sm">Please check your internet connection and refresh the page.</p>
                </div>`;
        });
    },
    
    /**
     * Show no data message in chart container
     * @param {HTMLElement} element - Chart container element
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(element, message) {
        if (!element) return;
        
        element.innerHTML = `
            <div class="data-unavailable">
                <div class="data-message">
                    <span class="message-icon">📊</span>
                    <h3>No Chart Data Available</h3>
                    <p>${message}</p>
                </div>
            </div>`;
    },
    
    /**
     * Default color palette for charts
     */
    colorPalette: {
        primary: 'rgb(31, 119, 180)',
        secondary: 'rgb(255, 127, 14)',
        tertiary: 'rgb(44, 160, 44)',
        quaternary: 'rgb(214, 39, 40)',
        quinary: 'rgb(148, 103, 189)',
        senary: 'rgb(140, 86, 75)',
        septenary: 'rgb(227, 119, 194)',
        octonary: 'rgb(127, 127, 127)',
        nonary: 'rgb(188, 189, 34)',
        denary: 'rgb(23, 190, 207)'
    },
    
    /**
     * Common layout options for Plotly charts
     * @param {string} title - Chart title
     * @param {string} xTitle - X axis title
     * @param {string} yTitle - Y axis title
     * @returns {Object} Layout configuration
     */
    getCommonLayout: function(title, xTitle, yTitle) {
        return {
            title: title,
            xaxis: {
                title: xTitle,
                automargin: true
            },
            yaxis: {
                title: yTitle,
                automargin: true
            },
            margin: {
                l: 50,
                r: 20,
                t: 60,
                b: 80
            },
            hovermode: 'closest',
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: -0.2,
                xanchor: 'center',
                x: 0.5
            }
        };
    },
    
    /**
     * Common config options for Plotly charts
     * @param {boolean} responsive - Whether chart should be responsive
     * @returns {Object} Config options
     */
    getCommonConfig: function(responsive = true) {
        return {
            responsive: responsive,
            displayModeBar: false,
            displaylogo: false
        };
    },
    
    /**
     * Create a line chart
     * @param {string} elementId - Container element ID
     * @param {Array} data - Chart data
     * @param {Object} layout - Layout configuration
     * @param {Object} config - Config options
     */
    createLineChart: function(elementId, data, layout = {}, config = {}) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.error(`Element not found: ${elementId}`);
            return;
        }
        
        if (!this.initializePlotly(() => this.createLineChart(elementId, data, layout, config))) {
            return;
        }
        
        const fullLayout = {...this.getCommonLayout('', '', ''), ...layout};
        const fullConfig = {...this.getCommonConfig(), ...config};
        
        Plotly.newPlot(element, data, fullLayout, fullConfig);
    },
    
    /**
     * Create a bar chart
     * @param {string} elementId - Container element ID
     * @param {Array} data - Chart data
     * @param {Object} layout - Layout configuration
     * @param {Object} config - Config options
     */
    createBarChart: function(elementId, data, layout = {}, config = {}) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.error(`Element not found: ${elementId}`);
            return;
        }
        
        if (!this.initializePlotly(() => this.createBarChart(elementId, data, layout, config))) {
            return;
        }
        
        const fullLayout = {...this.getCommonLayout('', '', ''), ...layout};
        const fullConfig = {...this.getCommonConfig(), ...config};
        
        Plotly.newPlot(element, data, fullLayout, fullConfig);
    },
    
    /**
     * Create a scatter plot
     * @param {string} elementId - Container element ID
     * @param {Array} data - Chart data
     * @param {Object} layout - Layout configuration
     * @param {Object} config - Config options
     */
    createScatterPlot: function(elementId, data, layout = {}, config = {}) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.error(`Element not found: ${elementId}`);
            return;
        }
        
        if (!this.initializePlotly(() => this.createScatterPlot(elementId, data, layout, config))) {
            return;
        }
        
        const fullLayout = {...this.getCommonLayout('', '', ''), ...layout};
        const fullConfig = {...this.getCommonConfig(), ...config};
        
        Plotly.newPlot(element, data, fullLayout, fullConfig);
    },
    
    /**
     * Create a boxplot
     * @param {string} elementId - Container element ID
     * @param {Array} data - Chart data
     * @param {Object} layout - Layout configuration
     * @param {Object} config - Config options
     */
    createBoxPlot: function(elementId, data, layout = {}, config = {}) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.error(`Element not found: ${elementId}`);
            return;
        }
        
        if (!this.initializePlotly(() => this.createBoxPlot(elementId, data, layout, config))) {
            return;
        }
        
        const fullLayout = {...this.getCommonLayout('', '', ''), ...layout};
        const fullConfig = {...this.getCommonConfig(), ...config};
        
        Plotly.newPlot(element, data, fullLayout, fullConfig);
    }
};