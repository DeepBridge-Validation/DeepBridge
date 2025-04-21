/**
 * Advanced Charts
 * 
 * Implementation of advanced chart types for the robustness report,
 * including scatter plots, radar charts, heatmaps, and ranking charts.
 */

class AdvancedCharts {
    /**
     * Initialize the advanced charts module
     * @param {Object} defaults - Default chart configuration
     * @param {boolean} plotlyAvailable - Whether Plotly.js is available
     */
    constructor(defaults, plotlyAvailable) {
      this.defaults = defaults;
      this.plotlyAvailable = plotlyAvailable;
    }
    
    /**
     * Create a scatter chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createScatterChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = config.series.map(series => ({
        type: 'scatter',
        mode: 'markers',
        name: series.name,
        x: series.data.map(d => d.x),
        y: series.data.map(d => d.y),
        text: series.data.map(d => d.name || ''),
        marker: {
          size: series.data.map(d => (d.marker && d.marker.radius) || 8),
          color: series.data.map(d => d.color || series.color),
          line: {
            width: 1,
            color: 'rgba(255, 255, 255, 0.5)'
          }
        }
      }));
      
      // Add quadrant lines if specified
      if (config.quadrantLines) {
        // Vertical line (x-axis)
        traces.push({
          type: 'scatter',
          mode: 'lines',
          name: 'X Mean',
          x: [config.quadrantLines.xValue, config.quadrantLines.xValue],
          y: [config.yAxis.min || 0, config.yAxis.max || 1],
          line: {
            color: 'rgba(0, 0, 0, 0.3)',
            width: 1,
            dash: 'dash'
          },
          showlegend: false,
          hoverinfo: 'none'
        });
        
        // Horizontal line (y-axis)
        traces.push({
          type: 'scatter',
          mode: 'lines',
          name: 'Y Mean',
          x: [config.xAxis.min || 0, config.xAxis.max || 1],
          y: [config.quadrantLines.yValue, config.quadrantLines.yValue],
          line: {
            color: 'rgba(0, 0, 0, 0.3)',
            width: 1,
            dash: 'dash'
          },
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
      
      // Add quadrant labels if specified
      if (config.quadrantLabels) {
        layout.annotations = config.quadrantLabels.map(label => ({
          x: label.position.x,
          y: label.position.y,
          text: label.text,
          showarrow: false,
          font: {
            size: 10,
            color: 'rgba(0, 0, 0, 0.5)'
          },
          bgcolor: 'rgba(255, 255, 255, 0.7)',
          bordercolor: 'rgba(0, 0, 0, 0.2)',
          borderwidth: 1,
          borderpad: 4
        }));
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
              name: point.text,
              inSubset: point.data.marker.color.indexOf('1.0') > -1 // Check if color indicates "in subset"
            }
          });
          
          // Update the tooltip content
          const tooltipEl = document.getElementsByClassName('hovertext')[0];
          if (tooltipEl) {
            tooltipEl.innerHTML = tooltipContent;
          }
        });
      }
    }
    
    /**
     * Create a radar chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createRadarChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = config.series.map(series => ({
        type: 'scatterpolar',
        name: series.name,
        r: series.data,
        theta: config.categories,
        fill: 'toself',
        fillcolor: `${series.color}50`, // 50% opacity
        line: {
          color: series.color,
          width: 2
        }
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
        polar: {
          radialaxis: {
            visible: true,
            range: [0, 1],
            tickfont: {
              size: 10,
              color: '#7f8c8d'
            }
          },
          angularaxis: {
            tickfont: {
              size: 10,
              color: '#7f8c8d'
            }
          },
          gridshape: 'circular'
        }
      };
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
    }
    
    /**
     * Create a heatmap chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createHeatmapChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data trace for the heatmap
      const trace = {
        type: 'heatmap',
        z: config.data,
        x: config.xAxis.categories,
        y: config.yAxis.categories,
        colorscale: 'Viridis',
        zmin: config.colorAxis ? config.colorAxis.min : undefined,
        zmax: config.colorAxis ? config.colorAxis.max : undefined,
        showscale: true,
        colorbar: {
          title: 'Importance',
          titleside: 'right',
          titlefont: {
            size: 12,
            color: '#7f8c8d'
          },
          tickfont: {
            size: 10,
            color: '#7f8c8d'
          }
        }
      };
      
      // Add text labels if required
      if (config.dataLabels && config.dataLabels.enabled) {
        trace.text = config.data.map(row => row.map(val => 
          config.dataLabels.formatter ? config.dataLabels.formatter.call({ point: { value: val } }) : val.toFixed(2)
        ));
        trace.texttemplate = '%{text}';
        trace.textfont = {
          color: 'white',
          size: 10
        };
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
          type: 'category'
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
          type: 'category'
        }
      };
      
      // Add annotations if provided
      if (config.annotations) {
        layout.annotations = config.annotations.map(annotation => ({
          x: config.xAxis.categories[annotation.x],
          y: config.yAxis.categories[annotation.y],
          showarrow: true,
          arrowhead: 0,
          arrowsize: 1,
          arrowcolor: annotation.marker.fillColor,
          arrowwidth: 2,
          ax: 0,
          ay: 0
        }));
      }
      
      // Create the chart
      Plotly.newPlot(container, [trace], layout, this.defaults.config);
    }
    
    /**
     * Create a ranking chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createRankingChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = config.series.map(series => ({
        type: 'scatter',
        mode: 'markers',
        name: series.name,
        x: config.xAxis.categories || Array.from({ length: series.data.length }, (_, i) => i),
        y: series.data.map(d => typeof d === 'object' ? d.y : d),
        marker: {
          size: series.data.map(d => typeof d === 'object' && d.marker ? d.marker.radius : 8),
          color: series.color,
          line: {
            width: 1,
            color: 'rgba(255, 255, 255, 0.5)'
          }
        }
      }));
      
      // Add connecting lines between pairs of points
      if (config.connectingLines) {
        config.connectingLines.forEach(line => {
          traces.push({
            type: 'scatter',
            mode: 'lines',
            name: `Connection-${line.x}`,
            x: [config.xAxis.categories[line.x], config.xAxis.categories[line.x]],
            y: [line.y1, line.y2],
            line: {
              color: line.color || 'rgba(0, 0, 0, 0.2)',
              width: 2
            },
            showlegend: false,
            hoverinfo: 'none'
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
          zeroline: false
        }
      };
      
      // Set axis ranges if provided
      if (config.yAxis.min !== undefined) {
        layout.yaxis.range = [config.yAxis.min, config.yAxis.max];
        layout.yaxis.autorange = config.yAxis.reversed ? 'reversed' : undefined;
      } else if (config.yAxis.reversed) {
        layout.yaxis.autorange = 'reversed';
      }
      
      // Create the chart
      Plotly.newPlot(container, traces, layout, this.defaults.config);
    }
    
    /**
     * Create a bubble chart
     * @param {HTMLElement} container - The container element
     * @param {Object} chartConfig - Chart configuration
     */
    createBubbleChart(container, chartConfig) {
      if (!this.plotlyAvailable || !container) return;
      
      const config = chartConfig || {};
      
      // Prepare data traces for the chart
      const traces = config.series.map(series => ({
        type: 'scatter',
        mode: 'markers',
        name: series.name,
        x: series.data.map(d => d.x),
        y: series.data.map(d => d.y),
        text: series.data.map(d => d.name || ''),
        marker: {
          size: series.data.map(d => d.z || d.size || 10),
          sizemode: 'diameter',
          sizeref: 0.1,
          sizemin: 5,
          color: series.data.map(d => d.color || series.color),
          line: {
            width: 1,
            color: 'rgba(255, 255, 255, 0.5)'
          }
        }
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
              z: point.marker.size,
              name: point.text
            }
          });
          
          // Update the tooltip content
          const tooltipEl = document.getElementsByClassName('hovertext')[0];
          if (tooltipEl) {
            tooltipEl.innerHTML = tooltipContent;
          }
        });
      }
    }
  }
  
  export default AdvancedCharts;