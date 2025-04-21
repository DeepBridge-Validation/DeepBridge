// ChartFactory.js - placeholder
/**
 * Chart Factory
 * 
 * Factory class for creating various chart types used in the resilience report.
 * Delegates chart creation to specialized chart implementation classes.
 */
import BarCharts from './BarCharts.js';
import LineCharts from './LineCharts.js';
import DistributionCharts from './DistributionCharts.js';
import ChartUtils from './ChartUtils.js';

class ChartFactory {
  /**
   * Initialize the chart factory
   */
  constructor() {
    // Default chart configuration
    this.defaults = {
      layout: {
        // Default layout configuration
        margin: {
          l: 60,
          r: 20,
          t: 40,
          b: 60
        },
        font: {
          family: 'Roboto, Arial, sans-serif',
          size: 12,
          color: '#333333'
        },
        paper_bgcolor: 'rgba(255,255,255,0)',
        plot_bgcolor: 'rgba(255,255,255,0)',
        showlegend: true,
        legend: {
          orientation: 'h',
          xanchor: 'center',
          yanchor: 'top',
          x: 0.5,
          y: -0.15
        }
      },
      config: {
        // Default plotly config
        responsive: true,
        displayModeBar: false,
        showTips: false
      }
    };
    
    // Check if Plotly is available
    this.plotlyAvailable = typeof Plotly !== 'undefined';
    
    if (!this.plotlyAvailable) {
      console.error('Plotly.js not found. Charts will not be rendered.');
    }
    
    // Initialize chart modules
    this.barCharts = new BarCharts(this.defaults, this.plotlyAvailable);
    this.lineCharts = new LineCharts(this.defaults, this.plotlyAvailable);
    this.distributionCharts = new DistributionCharts(this.defaults, this.plotlyAvailable);
    this.chartUtils = new ChartUtils();
    
    // Bind theme change handler
    this.bindThemeChangeHandler();
  }
  
  /**
   * Create a bar chart
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createBarChart(container, chartConfig) {
    return this.barCharts.createBarChart(container, chartConfig);
  }
  
  /**
   * Create a horizontal bar chart
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createHorizontalBarChart(container, chartConfig) {
    return this.barCharts.createHorizontalBarChart(container, chartConfig);
  }
  
  /**
   * Create a stacked bar chart
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createStackedBarChart(container, chartConfig) {
    return this.barCharts.createStackedBarChart(container, chartConfig);
  }
  
  /**
   * Create a line chart
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createLineChart(container, chartConfig) {
    return this.lineCharts.createLineChart(container, chartConfig);
  }
  
  /**
   * Create a multi-line chart
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createMultiLineChart(container, chartConfig) {
    return this.lineCharts.createMultiLineChart(container, chartConfig);
  }
  
  /**
   * Create an area chart
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createAreaChart(container, chartConfig) {
    return this.lineCharts.createAreaChart(container, chartConfig);
  }
  
  /**
   * Create a combination chart (bar + line)
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createCombinationChart(container, chartConfig) {
    // Combination chart can include both bars and lines
    return this.barCharts.createCombinationChart(container, chartConfig);
  }
  
  /**
   * Create a distribution comparison chart
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createDistributionComparisonChart(container, chartConfig) {
    return this.distributionCharts.createDistributionComparisonChart(container, chartConfig);
  }
  
  /**
   * Create a histogram chart
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createHistogramChart(container, chartConfig) {
    return this.distributionCharts.createHistogramChart(container, chartConfig);
  }
  
  /**
   * Create a density plot
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createDensityPlot(container, chartConfig) {
    return this.distributionCharts.createDensityPlot(container, chartConfig);
  }
  
  /**
   * Create a box plot chart
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createBoxPlot(container, chartConfig) {
    return this.distributionCharts.createBoxPlot(container, chartConfig);
  }
  
  /**
   * Create a scatter plot chart
   * @param {HTMLElement} container - Container element for the chart
   * @param {Object} chartConfig - Chart configuration
   * @return {Object} Chart instance
   */
  createScatterPlot(container, chartConfig) {
    return this.lineCharts.createScatterPlot(container, chartConfig);
  }
  
  /**
   * Bind theme change handler to update charts on theme change
   */
  bindThemeChangeHandler() {
    document.addEventListener('theme-changed', (e) => {
      const theme = e.detail.theme;
      
      // Update theme for all chart modules
      this.updateTheme(theme);
    });
    
    // Handle chart resize events
    document.addEventListener('chart-resize', () => {
      this.resizeAllCharts();
    });
  }
  
  /**
   * Update theme for all chart modules
   * @param {string} theme - Theme name ('light' or 'dark')
   */
  updateTheme(theme) {
    // Update font colors and background colors for charts
    const isDark = theme === 'dark';
    
    const updatedLayout = {
      font: {
        color: isDark ? '#e0e0e0' : '#333333'
      },
      xaxis: {
        color: isDark ? '#e0e0e0' : '#333333',
        gridcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'
      },
      yaxis: {
        color: isDark ? '#e0e0e0' : '#333333',
        gridcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'
      }
    };
    
    // Update defaults
    this.defaults.layout.font.color = updatedLayout.font.color;
    
    // Refresh any existing charts with new theme
    this.refreshChartsWithTheme(updatedLayout);
  }
  
  /**
   * Refresh all charts with new theme
   * @param {Object} themeLayout - Layout updates for the theme
   */
  refreshChartsWithTheme(themeLayout) {
    if (!this.plotlyAvailable) return;
    
    // Find all chart containers
    const chartContainers = document.querySelectorAll('.chart-plot');
    
    chartContainers.forEach(container => {
      if (container && container.data && container.layout) {
        const newLayout = {
          ...container.layout,
          font: {
            ...container.layout.font,
            ...themeLayout.font
          },
          xaxis: {
            ...container.layout.xaxis,
            ...themeLayout.xaxis
          },
          yaxis: {
            ...container.layout.yaxis,
            ...themeLayout.yaxis
          }
        };
        
        // Update the chart with new layout
        Plotly.relayout(container, newLayout);
      }
    });
  }
  
  /**
   * Resize all charts
   */
  resizeAllCharts() {
    if (!this.plotlyAvailable) return;
    
    // Find all chart containers
    const chartContainers = document.querySelectorAll('.chart-plot');
    
    chartContainers.forEach(container => {
      if (container) {
        Plotly.Plots.resize(container);
      }
    });
  }
}

export default ChartFactory;