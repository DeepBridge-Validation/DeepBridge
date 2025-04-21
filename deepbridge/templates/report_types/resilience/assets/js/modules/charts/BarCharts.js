// BarCharts.js - placeholder
/**
 * Bar Charts
 * 
 * Implementation of bar chart types for the resilience report,
 * including regular, horizontal, stacked, and grouped bar charts.
 */

class BarCharts {
    /**
     * Initialize the bar charts module
     * @param {Object} defaults - Default chart configuration
     * @param {boolean} plotlyAvailable - Whether Plotly.js is available
     */
    constructor(defaults, plotlyAvailable) {
      this.defaults = defaults;
      this.plotlyAvailable = plotlyAvailable;
    }
    
    /**
     * Create a bar chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     * @return {Object} Chart instance
     */
    createBarChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = [];
      
      // Add series
      if (config.series) {
        config.series.forEach(series => {
          traces.push({
            type: 'bar',
            name: series.name,
            x: config.xAxis.categories || Array.from({ length: series.data.length }, (_, i) => i),
            y: Array.isArray(series.data) ? 
               series.data.map(d => typeof d === 'object' ? d.y : d) : 
               [series.data],
            marker: {
              color: series.data && Array.isArray(series.data) && series.data[0] && typeof series.data[0] === 'object' ?
                    series.data.map(d => d.color || series.color) :
                    series.color
            }
          });
        });
      }
      
      // Layout configuration
      const layout = {
        ...this.defaults.layout,
        title: {
          text: config.title || '',
          font: {
            size: 16,
            color: this.defaults.layout.font.color
          }
        },
        xaxis: {
          title: {
            text: config.xAxis.title || '',
            font: {
              size: 14,
              color: this.defaults.layout.font.color
            }
          },
          tickfont: {
            size: 12,
            color: this.defaults.layout.font.color
          },
          gridcolor: 'rgba(0, 0, 0, 0.1)',
          type: config.xAxis.type || 'category'
        },
        yaxis: {
          title: {
            text: config.yAxis.title || '',
            font: {
              size: 14,
              color: this.defaults.layout.font.color
            }
          },
          tickfont: {
            size: 12,
            color: this.defaults.layout.font.color
          },
          gridcolor: 'rgba(0, 0, 0, 0.1)',
          zeroline: true,
          zerolinecolor: 'rgba(0, 0, 0, 0.2)'
        },
        barmode: 'group',
        bargap: 0.15,
        bargroupgap: 0.1
      };
      
      // Add reference line if provided
      if (config.referenceLine) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          name: config.referenceLine.name || 'Reference',
          x: config.xAxis.categories,
          y: Array(config.xAxis.categories.length).fill(config.referenceLine.value),
          line: {
            color: config.referenceLine.color || 'rgba(0, 0, 0, 0.5)',
            width: 2,
            dash: 'dash'
          }
        });
      }
      
      // Set axis ranges if provided
      if (config.yAxis.min !== undefined) layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
      
      // Add tooltip formatter if provided
      if (config.tooltipFormatter) {
        container.on('plotly_hover', (data) => {
          const point = data.points[0];
          
          // Call the formatter with point data
          const tooltipContent = config.tooltipFormatter.call({
            point: {
              x: point.x,
              y: point.y,
              series: {
                name: point.data.name
              }
            }
          });
          
          // Update the tooltip content
          const tooltipEl = document.getElementsByClassName('hovertext')[0];
          if (tooltipEl) {
            tooltipEl.innerHTML = tooltipContent;
          }
        });
      }
      
      return container;
    }
    
    /**
     * Create a horizontal bar chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     * @return {Object} Chart instance
     */
    createHorizontalBarChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = [];
      
      // Add series
      if (config.series) {
        config.series.forEach(series => {
          traces.push({
            type: 'bar',
            orientation: 'h',
            name: series.name,
            y: config.yAxis.categories || Array.from({ length: series.data.length }, (_, i) => i),
            x: Array.isArray(series.data) ? 
               series.data.map(d => typeof d === 'object' ? d.x : d) : 
               [series.data],
            marker: {
              color: series.data && Array.isArray(series.data) && series.data[0] && typeof series.data[0] === 'object' ?
                    series.data.map(d => d.color || series.color) :
                    series.color
            }
          });
        });
      }
      
      // Layout configuration
      const layout = {
        ...this.defaults.layout,
        title: {
          text: config.title || '',
          font: {
            size: 16,
            color: this.defaults.layout.font.color
          }
        },
        xaxis: {
          title: {
            text: config.xAxis.title || '',
            font: {
              size: 14,
              color: this.defaults.layout.font.color
            }
          },
          tickfont: {
            size: 12,
            color: this.defaults.layout.font.color
          },
          gridcolor: 'rgba(0, 0, 0, 0.1)',
          zeroline: true,
          zerolinecolor: 'rgba(0, 0, 0, 0.2)'
        },
        yaxis: {
          title: {
            text: config.yAxis.title || '',
            font: {
              size: 14,
              color: this.defaults.layout.font.color
            }
          },
          tickfont: {
            size: 12,
            color: this.defaults.layout.font.color
          },
          gridcolor: 'rgba(0, 0, 0, 0.1)',
          type: 'category'
        },
        barmode: 'group',
        bargap: 0.15,
        bargroupgap: 0.1
      };
      
      // Set axis ranges if provided
      if (config.xAxis.min !== undefined) layout.xaxis.range = [config.xAxis.min, config.xAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
      
      // Add tooltip formatter if provided
      if (config.tooltipFormatter) {
        container.on('plotly_hover', (data) => {
          const point = data.points[0];
          
          // Call the formatter with point data
          const tooltipContent = config.tooltipFormatter.call({
            point: {
              x: point.x,
              y: point.y,
              series: {
                name: point.data.name
              }
            }
          });
          
          // Update the tooltip content
          const tooltipEl = document.getElementsByClassName('hovertext')[0];
          if (tooltipEl) {
            tooltipEl.innerHTML = tooltipContent;
          }
        });
      }
      
      return container;
    }
    
    /**
     * Create a stacked bar chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     * @return {Object} Chart instance
     */
    createStackedBarChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Modify the config to create a stacked chart
      const stackedConfig = {
        ...config,
        layout: {
          ...config.layout,
          barmode: 'stack'
        }
      };
      
      // Prepare data traces for the chart
      const traces = [];
      
      // Add series
      if (config.series) {
        config.series.forEach(series => {
          traces.push({
            type: 'bar',
            name: series.name,
            x: config.xAxis.categories || Array.from({ length: series.data.length }, (_, i) => i),
            y: Array.isArray(series.data) ? 
               series.data.map(d => typeof d === 'object' ? d.y : d) : 
               [series.data],
            marker: {
              color: series.color
            }
          });
        });
      }
      
      // Layout configuration
      const layout = {
        ...this.defaults.layout,
        title: {
          text: config.title || '',
          font: {
            size: 16,
            color: this.defaults.layout.font.color
          }
        },
        xaxis: {
          title: {
            text: config.xAxis.title || '',
            font: {
              size: 14,
              color: this.defaults.layout.font.color
            }
          },
          tickfont: {
            size: 12,
            color: this.defaults.layout.font.color
          },
          gridcolor: 'rgba(0, 0, 0, 0.1)',
          type: config.xAxis.type || 'category'
        },
        yaxis: {
          title: {
            text: config.yAxis.title || '',
            font: {
              size: 14,
              color: this.defaults.layout.font.color
            }
          },
          tickfont: {
            size: 12,
            color: this.defaults.layout.font.color
          },
          gridcolor: 'rgba(0, 0, 0, 0.1)',
          zeroline: true,
          zerolinecolor: 'rgba(0, 0, 0, 0.2)'
        },
        barmode: 'stack', // Stacked mode
        bargap: 0.15,
        bargroupgap: 0.1
      };
      
      // Set axis ranges if provided
      if (config.yAxis.min !== undefined) layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
      
      // Add tooltip formatter if provided
      if (config.tooltipFormatter) {
        container.on('plotly_hover', (data) => {
          const point = data.points[0];
          
          // Call the formatter with point data
          const tooltipContent = config.tooltipFormatter.call({
            point: {
              x: point.x,
              y: point.y,
              series: {
                name: point.data.name
              }
            }
          });
          
          // Update the tooltip content
          const tooltipEl = document.getElementsByClassName('hovertext')[0];
          if (tooltipEl) {
            tooltipEl.innerHTML = tooltipContent;
          }
        });
      }
      
      return container;
    }
    
    /**
     * Create a combination chart (bar + line)
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     * @return {Object} Chart instance
     */
    createCombinationChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = [];
      
      // Add series
      if (config.series) {
        config.series.forEach(series => {
          const isLine = series.type === 'line';
          const trace = {
            type: isLine ? 'scatter' : 'bar',
            mode: isLine ? 'lines+markers' : undefined,
            name: series.name,
            x: config.xAxis.categories || Array.from({ length: series.data.length }, (_, i) => i),
            y: Array.isArray(series.data) ? 
               series.data.map(d => typeof d === 'object' ? d.y : d) : 
               [series.data],
            marker: {
              color: series.color
            }
          };
          
          if (isLine) {
            trace.line = {
              width: 2,
              color: series.color
            };
            
            // Add markers for line
            trace.marker = {
              size: 6,
              color: series.color
            };
          }
          
          traces.push(trace);
        });
      }
      
      // Layout configuration
      const layout = {
        ...this.defaults.layout,
        title: {
          text: config.title || '',
          font: {
            size: 16,
            color: this.defaults.layout.font.color
          }
        },
        xaxis: {
          title: {
            text: config.xAxis.title || '',
            font: {
              size: 14,
              color: this.defaults.layout.font.color
            }
          },
          tickfont: {
            size: 12,
            color: this.defaults.layout.font.color
          },
          gridcolor: 'rgba(0, 0, 0, 0.1)',
          type: config.xAxis.type || 'category'
        },
        yaxis: {
          title: {
            text: config.yAxis.title || '',
            font: {
              size: 14,
              color: this.defaults.layout.font.color
            }
          },
          tickfont: {
            size: 12,
            color: this.defaults.layout.font.color
          },
          gridcolor: 'rgba(0, 0, 0, 0.1)',
          zeroline: true,
          zerolinecolor: 'rgba(0, 0, 0, 0.2)'
        },
        barmode: 'group',
        bargap: 0.15,
        bargroupgap: 0.1
      };
      
      // Set axis ranges if provided
      if (config.yAxis.min !== undefined) layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
      
      // Add tooltip formatter if provided
      if (config.tooltipFormatter) {
        container.on('plotly_hover', (data) => {
          const point = data.points[0];
          
          // Call the formatter with point data
          const tooltipContent = config.tooltipFormatter.call({
            point: {
              x: point.x,
              y: point.y,
              series: {
                name: point.data.name
              }
            }
          });
          
          // Update the tooltip content
          const tooltipEl = document.getElementsByClassName('hovertext')[0];
          if (tooltipEl) {
            tooltipEl.innerHTML = tooltipContent;
          }
        });
      }
      
      return container;
    }
  }
  
  export default BarCharts;