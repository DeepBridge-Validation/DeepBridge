// ChartFactory.js - placeholder
/**
 * Chart Factory
 * 
 * Factory class for creating different types of charts for the uncertainty report,
 * delegating to specialized chart modules for implementation.
 */
import BasicCharts from './BasicCharts.js';
import ChartThemes from './ChartThemes.js';
import ChartInteractions from './ChartInteractions.js';

class ChartFactory {
  /**
   * Initialize the chart factory
   */
  constructor() {
    // Default chart configuration
    this.defaults = {
      layout: {
        autosize: true,
        margin: {
          l: 60,
          r: 20,
          t: 40,
          b: 60
        },
        paper_bgcolor: 'rgba(255, 255, 255, 0)',
        plot_bgcolor: 'rgba(255, 255, 255, 0)',
        font: {
          family: 'Roboto, Arial, sans-serif',
          size: 12
        },
        hovermode: 'closest',
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
    this.basicCharts = new BasicCharts(this.defaults, this.plotlyAvailable);
    this.chartThemes = new ChartThemes();
    this.chartInteractions = new ChartInteractions();
  }
  
  /**
   * Apply a theme to a chart configuration
   * @param {Object} chartConfig - Original chart configuration
   * @param {string} themeName - Name of the theme to apply
   * @return {Object} Themed chart configuration
   */
  applyTheme(chartConfig, themeName = 'light') {
    return this.chartThemes.applyTheme(chartConfig, themeName);
  }
  
  /**
   * Create a line chart
   * @param {HTMLElement} container - The container element
   * @param {Object} chartConfig - Chart configuration
   * @param {string} themeName - Name of the theme to apply (default: 'light')
   */
  createLineChart(container, chartConfig, themeName = 'light') {
    const themedConfig = this.applyTheme(chartConfig, themeName);
    this.basicCharts.createLineChart(container, themedConfig);
    this.chartInteractions.applyToChart(container, themedConfig);
  }
  
  /**
   * Create a bar chart
   * @param {HTMLElement} container - The container element
   * @param {Object} chartConfig - Chart configuration
   * @param {string} themeName - Name of the theme to apply (default: 'light')
   */
  createBarChart(container, chartConfig, themeName = 'light') {
    const themedConfig = this.applyTheme(chartConfig, themeName);
    this.basicCharts.createBarChart(container, themedConfig);
    this.chartInteractions.applyToChart(container, themedConfig);
  }
  
  /**
   * Create an area chart
   * @param {HTMLElement} container - The container element
   * @param {Object} chartConfig - Chart configuration
   * @param {string} themeName - Name of the theme to apply (default: 'light')
   */
  createAreaChart(container, chartConfig, themeName = 'light') {
    const themedConfig = this.applyTheme(chartConfig, themeName);
    this.basicCharts.createAreaChart(container, themedConfig);
    this.chartInteractions.applyToChart(container, themedConfig);
  }
  
  /**
   * Create a horizontal bar chart
   * @param {HTMLElement} container - The container element
   * @param {Object} chartConfig - Chart configuration
   * @param {string} themeName - Name of the theme to apply (default: 'light')
   */
  createHorizontalBarChart(container, chartConfig, themeName = 'light') {
    const themedConfig = this.applyTheme(chartConfig, themeName);
    this.basicCharts.createHorizontalBarChart(container, themedConfig);
    this.chartInteractions.applyToChart(container, themedConfig);
  }
  
  /**
   * Create a calibration curve chart (specialized for uncertainty visualization)
   * @param {HTMLElement} container - The container element
   * @param {Object} chartConfig - Chart configuration
   * @param {string} themeName - Name of the theme to apply (default: 'light')
   */
  createCalibrationChart(container, chartConfig, themeName = 'light') {
    // Calibration charts are line charts with specialized configuration
    const themedConfig = this.applyTheme(chartConfig, themeName);
    
    // Add diagonal reference line representing perfect calibration
    if (!themedConfig.referenceLine) {
      themedConfig.referenceLine = {
        name: 'Perfect Calibration',
        value: null, // Will be handled specially
        color: 'rgba(180, 180, 180, 0.7)'
      };
    }
    
    // Create a line chart with the perfect calibration diagonal
    const traces = themedConfig.series.map(series => ({
      type: 'scatter',
      mode: 'lines+markers',
      name: series.name,
      x: series.data.map(d => typeof d === 'object' ? d.x : d),
      y: series.data.map(d => typeof d === 'object' ? d.y : d),
      line: {
        color: series.color,
        width: series.lineWidth || 2,
        dash: series.dashStyle
      },
      marker: {
        size: series.markerSize || 6,
        color: series.markerColor || series.color
      }
    }));
    
    // Add diagonal perfect calibration line
    traces.push({
      type: 'scatter',
      mode: 'lines',
      name: themedConfig.referenceLine.name,
      x: [0, 1],
      y: [0, 1],
      line: {
        color: themedConfig.referenceLine.color,
        width: 2,
        dash: 'dash'
      },
      hoverinfo: 'name'
    });
    
    // Layout configuration
    const layout = {
      ...this.defaults.layout,
      title: {
        text: themedConfig.title || 'Calibration Curve',
        font: {
          size: 16,
          color: '#2c3e50'
        }
      },
      xaxis: {
        title: {
          text: themedConfig.xAxis.title || 'Predicted Probability',
          font: {
            size: 14,
            color: '#7f8c8d'
          }
        },
        tickfont: {
          size: 12,
          color: '#7f8c8d'
        },
        gridcolor: '#ecf0f1',
        zeroline: true,
        zerolinecolor: '#bdc3c7',
        range: [0, 1]
      },
      yaxis: {
        title: {
          text: themedConfig.yAxis.title || 'Observed Frequency',
          font: {
            size: 14,
            color: '#7f8c8d'
          }
        },
        tickfont: {
          size: 12,
          color: '#7f8c8d'
        },
        gridcolor: '#ecf0f1',
        zeroline: true,
        zerolinecolor: '#bdc3c7',
        range: [0, 1]
      }
    };
    
    // Create the chart
    if (this.plotlyAvailable) {
      Plotly.newPlot(container, traces, layout, this.defaults.config);
      this.chartInteractions.applyToChart(container, themedConfig);
    }
  }
  
  /**
   * Create a coverage chart (specialized for uncertainty visualization)
   * @param {HTMLElement} container - The container element
   * @param {Object} chartConfig - Chart configuration
   * @param {string} themeName - Name of the theme to apply (default: 'light')
   */
  createCoverageChart(container, chartConfig, themeName = 'light') {
    const themedConfig = this.applyTheme(chartConfig, themeName);
    
    // Add reference line for target coverage
    if (!themedConfig.referenceLine && themedConfig.targetCoverage) {
      themedConfig.referenceLine = {
        name: 'Target Coverage',
        value: themedConfig.targetCoverage,
        color: 'rgba(180, 180, 180, 0.7)'
      };
    }
    
    this.basicCharts.createLineChart(container, themedConfig);
    this.chartInteractions.applyToChart(container, themedConfig);
  }
  
  /**
   * Create an interval width chart (specialized for uncertainty visualization)
   * @param {HTMLElement} container - The container element
   * @param {Object} chartConfig - Chart configuration
   * @param {string} themeName - Name of the theme to apply (default: 'light')
   */
  createIntervalWidthChart(container, chartConfig, themeName = 'light') {
    const themedConfig = this.applyTheme(chartConfig, themeName);
    this.basicCharts.createLineChart(container, themedConfig);
    this.chartInteractions.applyToChart(container, themedConfig);
  }
  
  /**
   * Create an uncertainty distribution chart
   * @param {HTMLElement} container - The container element
   * @param {Object} chartConfig - Chart configuration
   * @param {string} themeName - Name of the theme to apply (default: 'light')
   */
  createUncertaintyDistributionChart(container, chartConfig, themeName = 'light') {
    const themedConfig = this.applyTheme(chartConfig, themeName);
    
    // Use an area chart with specialized configuration
    this.basicCharts.createAreaChart(container, themedConfig);
    this.chartInteractions.applyToChart(container, themedConfig);
  }
}

export default ChartFactory;