/**
 * Statistical Charts
 * 
 * Implementation of charts specific to statistical analysis in the robustness report,
 * including boxplots, violin plots, histograms, and distribution visualizations.
 */

class StatisticalCharts {
    /**
     * Initialize the statistical charts module
     * @param {Object} defaults - Default chart configuration
     * @param {boolean} plotlyAvailable - Whether Plotly.js is available
     */
    constructor(defaults, plotlyAvailable) {
      this.defaults = defaults;
      this.plotlyAvailable = plotlyAvailable;
    }
    
    /**
     * Create a boxplot chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createBoxplotChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = [];
      
      // Add boxplot series
      if (config.boxplotSeries) {
        config.boxplotSeries.forEach(series => {
          traces.push({
            type: 'box',
            name: series.name,
            x: series.data.map(d => d.x),
            y: null,
            lowerfence: series.data.map(d => d.low),
            q1: series.data.map(d => d.q1),
            median: series.data.map(d => d.median),
            q3: series.data.map(d => d.q3),
            upperfence: series.data.map(d => d.high),
            mean: series.data.map(d => d.mean),
            outliers: series.data.map(d => d.outliers || []).flat(),
            marker: {
              color: series.color,
              line: {
                width: 1,
                color: 'rgba(255, 255, 255, 0.5)'
              }
            },
            boxmean: 'sd'
          });
        });
      }
      
      // Add line series if provided
      if (config.lineSeries) {
        config.lineSeries.forEach(series => {
          traces.push({
            type: 'scatter',
            mode: 'lines+markers',
            name: series.name,
            x: series.data.map(d => d.x),
            y: series.data.map(d => d.y),
            line: {
              color: series.color,
              width: 2
            },
            marker: {
              size: 6,
              color: series.color
            }
          });
        });
      }
      
      // Add reference lines if provided
      if (config.referenceSeries) {
        config.referenceSeries.forEach(series => {
          traces.push({
            type: 'scatter',
            mode: 'lines',
            name: series.name,
            x: series.data.map(d => d.x),
            y: series.data.map(d => d.y),
            line: {
              color: series.color,
              width: 2,
              dash: series.dashStyle || 'dash'
            },
            marker: { size: 0 },
            hoverinfo: series.enableMouseTracking === false ? 'none' : 'all'
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
          type: 'category',
          categoryorder: 'array',
          categoryarray: config.xAxis.categories
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
        boxmode: 'group',
        boxgap: 0.1,
        boxgroupgap: 0.2
      };
      
      // Set axis ranges if provided
      if (config.yAxis.min !== undefined) layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
    }
    
    /**
     * Create a category boxplot chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createCategoryBoxplotChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = [];
      
      // Add boxplot series
      if (config.boxplotSeries) {
        config.boxplotSeries.forEach(series => {
          traces.push({
            type: 'box',
            name: series.name,
            y: series.data.map(d => typeof d === 'object' ? d.low : d), // Lower whisker
            q1: series.data.map(d => typeof d === 'object' ? d.q1 : null),
            median: series.data.map(d => typeof d === 'object' ? d.median : null),
            q3: series.data.map(d => typeof d === 'object' ? d.q3 : null),
            upperfence: series.data.map(d => typeof d === 'object' ? d.high : null), // Upper whisker
            lowerfence: series.data.map(d => typeof d === 'object' ? d.low : null), // Lower whisker
            mean: series.data.map(d => typeof d === 'object' ? d.mean : null),
            x: series.data.map(d => typeof d === 'object' ? d.x : null),
            marker: {
              color: series.color,
              outliercolor: 'rgba(0, 0, 0, 0.3)',
              line: {
                width: 1,
                color: 'rgba(255, 255, 255, 0.5)'
              }
            },
            boxmean: 'sd'
          });
        });
      }
      
      // Add legend series if provided
      if (config.legendSeries) {
        config.legendSeries.forEach(series => {
          traces.push({
            type: 'scatter',
            mode: 'markers',
            name: series.name,
            x: series.data.length > 0 ? series.data : [null],
            y: series.data.length > 0 ? series.data : [null],
            marker: {
              color: series.color,
              size: 8,
              symbol: series.marker ? series.marker.symbol : 'circle'
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
          type: 'category',
          categoryorder: 'array',
          categoryarray: config.xAxis.categories
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
        boxmode: 'group',
        boxgap: 0.1,
        boxgroupgap: 0.2
      };
      
      // Handle rotated labels if specified
      if (config.xAxis.labels && config.xAxis.labels.rotation) {
        layout.xaxis.tickangle = config.xAxis.labels.rotation;
      }
      
      // Set axis ranges if provided
      if (config.yAxis.min !== undefined) layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
    }
    
    /**
     * Create a violin chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createViolinChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = [];
      
      // Add violin series
      if (config.violinSeries) {
        config.violinSeries.forEach((series, index) => {
          traces.push({
            type: 'violin',
            name: series.name,
            y: series.data,
            x: Array(series.data.length).fill(config.xAxis.categories[index] || index),
            points: 'all',
            pointpos: 0,
            jitter: 0.05,
            marker: {
              size: 3,
              opacity: 0.5,
              color: series.color
            },
            box: {
              visible: true
            },
            line: {
              color: series.color
            },
            meanline: {
              visible: true
            },
            fillcolor: `${series.color}50` // 50% opacity
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
          type: 'category',
          categoryorder: 'array',
          categoryarray: config.xAxis.categories
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
        violinmode: 'group',
        violingap: 0.1,
        violingroupgap: 0.2
      };
      
      // Set axis ranges if provided
      if (config.yAxis.min !== undefined) layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
    }
    
    /**
     * Create a histogram chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createHistogramChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the histogram
      const traces = config.series.map(series => ({
        type: 'histogram',
        name: series.name,
        x: series.data,
        opacity: 0.7,
        marker: {
          color: series.color
        },
        histnorm: config.normalized ? 'probability' : '',
        autobinx: !config.binCount,
        nbinsx: config.binCount
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
            text: config.normalized ? 'Probability' : 'Frequency',
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
        barmode: 'overlay'
      };
      
      // Set axis ranges if provided
      if (config.xAxis.min !== undefined) layout.xaxis.range = [config.xAxis.min, config.xAxis.max];
      if (config.yAxis.min !== undefined) layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
    }
    
    /**
     * Create a gauge chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createGaugeChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Default values
      const value = config.value || 0;
      const min = config.min || 0;
      const max = config.max || 1;
      const title = config.title || '';
      const color = config.color || '#1b78de';
      
      // Calculate the gauge position (0-180 degrees)
      const position = (value - min) / (max - min) * 180;
      
      // Create the gauge trace
      const trace = {
        type: 'indicator',
        mode: 'gauge',
        value: value,
        title: {
          text: title,
          font: {
            size: 16,
            color: '#2c3e50'
          }
        },
        gauge: {
          axis: {
            range: [min, max],
            tickwidth: 1,
            tickcolor: '#7f8c8d',
            tickfont: {
              size: 10,
              color: '#7f8c8d'
            }
          },
          bar: {
            color: color,
            thickness: 0.6
          },
          bgcolor: '#ecf0f1',
          bordercolor: '#bdc3c7',
          borderwidth: 1,
          steps: config.steps || []
        }
      };
      
      // Add threshold steps if provided
      if (config.thresholds) {
        trace.gauge.steps = config.thresholds.map(threshold => ({
          range: [threshold.min, threshold.max],
          color: threshold.color,
          line: {
            color: 'transparent',
            width: 0
          }
        }));
      }
      
      // Layout configuration
      const layout = {
        ...this.defaults.layout,
        margin: {
          l: 20,
          r: 20,
          t: 50,
          b: 20
        },
        width: container.offsetWidth,
        height: container.offsetWidth * 0.6, // 60% height to width ratio
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {
          family: 'Roboto, Arial, sans-serif',
          size: 12
        }
      };
      
      // Create the chart
      Plotly.newPlot(container, [trace], layout, this.defaults.config);
    }
  }
  
  export default StatisticalCharts;