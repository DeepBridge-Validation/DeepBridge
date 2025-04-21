import BasicCharts from './BasicCharts.js';
import AdvancedCharts from './AdvancedCharts.js';
import StatisticalCharts from './StatisticalCharts.js';

class ChartFactory {
  constructor() {
    // Default chart configuration
    this.defaults = {
      layout: {
        // Default layout configuration
      },
      config: {
        // Default plotly config
      }
    };
    
    // Check if Plotly is available
    this.plotlyAvailable = typeof Plotly !== 'undefined';
    
    if (!this.plotlyAvailable) {
      console.error('Plotly.js not found. Charts will not be rendered.');
    }
    
    // Initialize chart modules
    this.basicCharts = new BasicCharts(this.defaults, this.plotlyAvailable);
    this.advancedCharts = new AdvancedCharts(this.defaults, this.plotlyAvailable);
    this.statisticalCharts = new StatisticalCharts(this.defaults, this.plotlyAvailable);
  }
  
  // Delegate to the appropriate module
  createLineChart(container, chartConfig) {
    return this.basicCharts.createLineChart(container, chartConfig);
  }
  
  createBarChart(container, chartConfig) {
    return this.basicCharts.createBarChart(container, chartConfig);
  }
  
  createScatterChart(container, chartConfig) {
    return this.basicCharts.createScatterChart(container, chartConfig);
  }
  
  createBoxplotChart(container, chartConfig) {
    return this.statisticalCharts.createBoxplotChart(container, chartConfig);
  }
  
  // ... other chart creation methods that delegate to the appropriate module
}

export default ChartFactory;