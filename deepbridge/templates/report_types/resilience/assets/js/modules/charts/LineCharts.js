// LineCharts.js - placeholder
/**
 * Line Charts
 * 
 * Implementation of line chart types for the resilience report,
 * including regular line, multi-line, area, and scatter charts.
 */

class LineCharts {
    /**
     * Initialize the line charts module
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
     * @return {Object} Chart instance
     */
    createLineChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = [];
      
      // Add series
      if (config.series) {
        config.series.forEach(series => {
          traces.push({
            type: 'scatter',
            mode: 'lines+markers',
            name: series.name,
            x: config.xAxis.categories || Array.from({ length: series.data.length }, (_, i) => i),
            y: Array.isArray(series.data) ? 
               series.data.map(d => typeof d === 'object' ? d.y : d) : 
               [series.data],
            line: {
              shape: series.lineShape || 'linear',
              width: 2,
              color: series.color
            },
            marker: {
              size: 6,
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
        }
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
     * Create a multi-line chart with multiple series
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     * @return {Object} Chart instance
     */
    createMultiLineChart(container, chartConfig) {
      // Multi-line is essentially the same as a regular line chart
      // but with multiple series
      return this.createLineChart(container, chartConfig);
    }
    
    /**
     * Create an area chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     * @return {Object} Chart instance
     */
    createAreaChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = [];
      
      // Add series
      if (config.series) {
        config.series.forEach(series => {
          traces.push({
            type: 'scatter',
            mode: 'lines',
            name: series.name,
            x: config.xAxis.categories || Array.from({ length: series.data.length }, (_, i) => i),
            y: Array.isArray(series.data) ? 
               series.data.map(d => typeof d === 'object' ? d.y : d) : 
               [series.data],
            fill: 'tozeroy',
            fillcolor: `${series.color}50`, // 50% opacity
            line: {
              shape: series.lineShape || 'linear',
              width: 2,
              color: series.color
            }
          });
        });
      }
      
      // Add confidence intervals if specified
      if (config.showConfidenceIntervals && config.series) {
        config.series.forEach(series => {
          if (series.confidenceInterval) {
            // Lower bound
            traces.push({
              type: 'scatter',
              mode: 'lines',
              name: `${series.name} (Lower CI)`,
              x: config.xAxis.categories,
              y: series.confidenceInterval.map(interval => interval[0]),
              line: {
                width: 0
              },
              showlegend: false,
              hoverinfo: 'skip'
            });
            
            // Upper bound
            traces.push({
              type: 'scatter',
              mode: 'lines',
              name: `${series.name} (Upper CI)`,
              x: config.xAxis.categories,
              y: series.confidenceInterval.map(interval => interval[1]),
              fill: 'tonexty',
              fillcolor: `${series.color}40`, // 25% opacity
              line: {
                width: 0
              },
              showlegend: false,
              hoverinfo: 'skip'
            });
          }
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
        }
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
     * Create a scatter plot
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     * @return {Object} Chart instance
     */
    createScatterPlot(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = [];
      
      // Add series
      if (config.series) {
        config.series.forEach(series => {
          traces.push({
            type: 'scatter',
            mode: 'markers',
            name: series.name,
            x: series.data.map(d => d.x),
            y: series.data.map(d => d.y),
            text: series.data.map(d => d.name || ''),
            marker: {
              size: series.data.map(d => d.size || 8),
              color: series.data.map(d => d.color || series.color),
              line: {
                width: 1,
                color: 'rgba(255, 255, 255, 0.5)'
              }
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
          zeroline: true,
          zerolinecolor: 'rgba(0, 0, 0, 0.2)'
        }
      };
      
      // Add trend line if specified
      if (config.showTrendLine && config.series && config.series.length > 0) {
        // Calculate trend line
        const series = config.series[0];
        const xValues = series.data.map(d => d.x);
        const yValues = series.data.map(d => d.y);
        
        // Simple linear regression
        const n = xValues.length;
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        
        for (let i = 0; i < n; i++) {
          sumX += xValues[i];
          sumY += yValues[i];
          sumXY += xValues[i] * yValues[i];
          sumX2 += xValues[i] * xValues[i];
        }
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        // Create trend line trace
        const minX = Math.min(...xValues);
        const maxX = Math.max(...xValues);
        
        traces.push({
          type: 'scatter',
          mode: 'lines',
          name: 'Trend Line',
          x: [minX, maxX],
          y: [slope * minX + intercept, slope * maxX + intercept],
          line: {
            color: 'rgba(255, 0, 0, 0.7)',
            width: 2,
            dash: 'dash'
          },
          showlegend: true
        });
      }
      
      // Set axis ranges if provided
      if (config.xAxis.min !== undefined) layout.xaxis.range = [config.xAxis.min, config.xAxis.max];
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
              text: point.text,
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
  
  export default LineCharts;