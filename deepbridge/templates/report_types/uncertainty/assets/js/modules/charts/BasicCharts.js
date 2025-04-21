// BasicCharts.js - placeholder
/**
 * Basic Charts
 * 
 * Implementation of basic chart types for the uncertainty report,
 * including line charts, bar charts, and area charts.
 */

class BasicCharts {
    /**
     * Initialize the basic charts module
     * @param {Object} defaults - Default chart configuration
     * @param {boolean} plotlyAvailable - Whether Plotly.js is available
     */
    constructor(defaults, plotlyAvailable) {
      this.defaults = defaults;
      this.plotlyAvailable = plotlyAvailable;
    }
    
    /**
     * Create a line chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createLineChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = config.series.map(series => ({
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
      
      // Add reference lines if provided
      if (config.referenceLine) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          name: config.referenceLine.name,
          x: [config.xAxis.min || 0, config.xAxis.max || 1],
          y: [config.referenceLine.value, config.referenceLine.value],
          line: {
            color: config.referenceLine.color,
            width: 2,
            dash: 'dash'
          },
          hoverinfo: 'name+y'
        });
      }
      
      // Add uncertainty intervals if provided
      if (config.showConfidenceIntervals && config.series[0].confidenceInterval) {
        const mainSeries = config.series[0];
        
        traces.push({
          type: 'scatter',
          name: 'Confidence Interval',
          x: [...mainSeries.data.map(d => typeof d === 'object' ? d.x : d), 
              ...mainSeries.data.map(d => typeof d === 'object' ? d.x : d).reverse()],
          y: [...mainSeries.confidenceInterval.map(ci => ci[1]), 
              ...mainSeries.confidenceInterval.map(ci => ci[0]).reverse()],
          fill: 'toself',
          fillcolor: `${mainSeries.color}30`, // 30% opacity
          line: { color: 'transparent' },
          showlegend: false,
          hoverinfo: 'none'
        });
      }
      
      // Layout configuration
      const layout = {
        ...this.defaults.layout,
        title: {
          text: config.title || '',
          font: {
            size: 16,
            color: '#2c3e50'
          }
        },
        xaxis: {
          title: {
            text: config.xAxis.title || '',
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
          zerolinecolor: '#bdc3c7'
        },
        yaxis: {
          title: {
            text: config.yAxis.title || '',
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
          zerolinecolor: '#bdc3c7'
        }
      };
      
      // Set axis categories if provided
      if (config.xAxis.categories) {
        layout.xaxis.tickmode = 'array';
        layout.xaxis.tickvals = Array.from({length: config.xAxis.categories.length}, (_, i) => i);
        layout.xaxis.ticktext = config.xAxis.categories;
      }
      
      // Set axis ranges if provided
      if (config.xAxis.min !== undefined) layout.xaxis.range = [config.xAxis.min, config.xAxis.max];
      if (config.yAxis.min !== undefined) layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
    }
    
    /**
     * Create a bar chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createBarChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = config.series.map(series => ({
        type: 'bar',
        name: series.name,
        x: config.xAxis.categories || Array.from({length: series.data.length}, (_, i) => i),
        y: series.data.map(d => typeof d === 'object' ? d.y : d),
        marker: {
          color: series.data.map(d => typeof d === 'object' && d.color ? d.color : series.color)
        },
        text: series.data.map(d => typeof d === 'object' && d.label ? d.label : null),
        textposition: 'auto',
        hoverinfo: 'x+y+name'
      }));
      
      // Add threshold line if provided
      if (config.threshold) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          name: config.threshold.name,
          x: [-0.5, config.xAxis.categories.length - 0.5],
          y: [config.threshold.value, config.threshold.value],
          line: {
            color: config.threshold.color,
            width: 2,
            dash: 'dash'
          },
          hoverinfo: 'name+y'
        });
      }
      
      // Layout configuration
      const layout = {
        ...this.defaults.layout,
        title: {
          text: config.title || '',
          font: {
            size: 16,
            color: '#2c3e50'
          }
        },
        xaxis: {
          title: {
            text: config.xAxis.title || '',
            font: {
              size: 14,
              color: '#7f8c8d'
            }
          },
          tickfont: {
            size: 12,
            color: '#7f8c8d'
          },
          gridcolor: '#ecf0f1'
        },
        yaxis: {
          title: {
            text: config.yAxis.title || '',
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
          zerolinecolor: '#bdc3c7'
        },
        barmode: config.stacked ? 'stack' : 'group'
      };
      
      // Set axis ranges if provided
      if (config.yAxis.min !== undefined) layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
    }
    
    /**
     * Create an area chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createAreaChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = config.series.map(series => ({
        type: 'scatter',
        mode: 'lines',
        name: series.name,
        x: series.data.map(d => typeof d === 'object' ? d.x : d),
        y: series.data.map(d => typeof d === 'object' ? d.y : d),
        fill: 'tozeroy',
        fillcolor: `${series.color}50`, // 50% opacity
        line: {
          color: series.color,
          width: series.lineWidth || 2
        },
        hoverinfo: 'x+y+name'
      }));
      
      // Layout configuration
      const layout = {
        ...this.defaults.layout,
        title: {
          text: config.title || '',
          font: {
            size: 16,
            color: '#2c3e50'
          }
        },
        xaxis: {
          title: {
            text: config.xAxis.title || '',
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
          zerolinecolor: '#bdc3c7'
        },
        yaxis: {
          title: {
            text: config.yAxis.title || '',
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
          zerolinecolor: '#bdc3c7'
        }
      };
      
      // Set axis categories if provided
      if (config.xAxis.categories) {
        layout.xaxis.tickmode = 'array';
        layout.xaxis.tickvals = Array.from({length: config.xAxis.categories.length}, (_, i) => i);
        layout.xaxis.ticktext = config.xAxis.categories;
      }
      
      // Set axis ranges if provided
      if (config.xAxis.min !== undefined) layout.xaxis.range = [config.xAxis.min, config.xAxis.max];
      if (config.yAxis.min !== undefined) layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
    }
    
    /**
     * Create a horizontal bar chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createHorizontalBarChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = config.series.map(series => ({
        type: 'bar',
        orientation: 'h',
        name: series.name,
        y: config.yAxis.categories || Array.from({length: series.data.length}, (_, i) => i),
        x: series.data.map(d => typeof d === 'object' ? d.x : d),
        marker: {
          color: series.data.map(d => typeof d === 'object' && d.color ? d.color : series.color)
        },
        text: series.data.map(d => typeof d === 'object' && d.label ? d.label : null),
        textposition: 'auto',
        hoverinfo: 'x+y+name'
      }));
      
      // Layout configuration
      const layout = {
        ...this.defaults.layout,
        title: {
          text: config.title || '',
          font: {
            size: 16,
            color: '#2c3e50'
          }
        },
        xaxis: {
          title: {
            text: config.xAxis.title || '',
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
          zerolinecolor: '#bdc3c7'
        },
        yaxis: {
          title: {
            text: config.yAxis.title || '',
            font: {
              size: 14,
              color: '#7f8c8d'
            }
          },
          tickfont: {
            size: 12,
            color: '#7f8c8d'
          },
          gridcolor: '#ecf0f1'
        },
        barmode: config.stacked ? 'stack' : 'group'
      };
      
      // Set axis ranges if provided
      if (config.xAxis.min !== undefined) layout.xaxis.range = [config.xAxis.min, config.xAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
    }
  }
  
  export default BasicCharts;