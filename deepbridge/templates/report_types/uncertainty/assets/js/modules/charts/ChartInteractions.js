// ChartInteractions.js - placeholder
/**
 * Chart Interactions
 * 
 * Handles interactive chart features including tooltips, zooming,
 * selection, and dynamic updates.
 */

class ChartInteractions {
    /**
     * Initialize the chart interactions module
     * @param {Object} defaults - Default interaction configuration
     */
    constructor(defaults = {}) {
      // Default interaction options
      this.defaults = {
        tooltip: {
          enabled: true,
          formatter: null,
          shared: true,
          followPointer: true
        },
        zoom: {
          enabled: false,
          type: 'xy',  // 'x', 'y', or 'xy'
          resetButton: true
        },
        selection: {
          enabled: false,
          type: 'xy'   // 'x', 'y', or 'xy'
        },
        events: {
          click: null,
          hover: null,
          selection: null
        },
        ...defaults
      };
      
      // Check if Plotly is available
      this.plotlyAvailable = typeof Plotly !== 'undefined';
    }
    
    /**
     * Apply interactions to a chart
     * @param {HTMLElement} container - The chart container element
     * @param {Object} config - Chart-specific interaction configuration
     */
    applyToChart(container, config = {}) {
      if (!this.plotlyAvailable || !container) return;
      
      // Merge default and chart-specific configuration
      const interactionConfig = {
        ...this.defaults,
        ...config
      };
      
      // Apply event handlers
      this.applyEventHandlers(container, interactionConfig);
      
      // Apply tooltip configuration
      if (interactionConfig.tooltip.enabled) {
        this.configureTooltip(container, interactionConfig.tooltip);
      }
      
      // Apply zoom configuration
      if (interactionConfig.zoom.enabled) {
        this.configureZoom(container, interactionConfig.zoom);
      }
      
      // Apply selection configuration
      if (interactionConfig.selection.enabled) {
        this.configureSelection(container, interactionConfig.selection);
      }
    }
    
    /**
     * Apply event handlers to a chart
     * @param {HTMLElement} container - The chart container element
     * @param {Object} config - Interaction configuration
     */
    applyEventHandlers(container, config) {
      // Click event
      if (config.events.click) {
        container.on('plotly_click', (data) => {
          config.events.click(data);
        });
      }
      
      // Hover event
      if (config.events.hover) {
        container.on('plotly_hover', (data) => {
          config.events.hover(data);
        });
      }
      
      // Selection event
      if (config.events.selection) {
        container.on('plotly_selected', (data) => {
          config.events.selection(data);
        });
      }
    }
    
    /**
     * Configure tooltip behavior
     * @param {HTMLElement} container - The chart container element
     * @param {Object} tooltipConfig - Tooltip configuration
     */
    configureTooltip(container, tooltipConfig) {
      // Customize tooltip behavior
      if (tooltipConfig.formatter) {
        container.on('plotly_hover', (data) => {
          const point = data.points[0];
          
          // Call the formatter with point data
          const tooltipContent = tooltipConfig.formatter.call({
            point: {
              x: point.x,
              y: point.y,
              name: point.text || point.data.name,
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
    }
    
    /**
     * Configure zoom behavior
     * @param {HTMLElement} container - The chart container element
     * @param {Object} zoomConfig - Zoom configuration
     */
    configureZoom(container, zoomConfig) {
      // Get the chart data
      const plotlyContainer = container;
      const plotInstance = plotlyContainer.data;
      
      if (!plotInstance) return;
      
      // Apply zoom configuration
      const updateOptions = {
        scrollZoom: true,
        modeBar: {
          orientation: 'v',
          color: '#7f8c8d',
          activecolor: '#2c3e50'
        }
      };
      
      // Add reset button if enabled
      if (zoomConfig.resetButton) {
        updateOptions.modeBar.buttons = ['resetScale2d'];
      }
      
      // Configure zoom direction
      switch (zoomConfig.type) {
        case 'x':
          updateOptions.dragmode = 'zoom';
          break;
        case 'y':
          updateOptions.dragmode = 'zoom';
          break;
        case 'xy':
        default:
          updateOptions.dragmode = 'zoom';
          break;
      }
      
      // Update the chart with new options
      Plotly.update(container, {}, {dragmode: updateOptions.dragmode}, [0]);
      
      // Update plot layout to show zoom options in mode bar
      const update = {
        config: {
          ...this.defaults.config,
          scrollZoom: true,
          displayModeBar: true,
          modeBarButtonsToAdd: ['resetScale2d'],
          ...updateOptions
        }
      };
      
      Plotly.react(container, plotlyContainer.data, plotlyContainer.layout, update.config);
    }
    
    /**
     * Configure selection behavior
     * @param {HTMLElement} container - The chart container element
     * @param {Object} selectionConfig - Selection configuration
     */
    configureSelection(container, selectionConfig) {
      // Get the chart data
      const plotlyContainer = container;
      const plotInstance = plotlyContainer.data;
      
      if (!plotInstance) return;
      
      // Apply selection configuration
      const updateOptions = {
        dragmode: 'select'
      };
      
      // Configure selection direction
      switch (selectionConfig.type) {
        case 'x':
          updateOptions.dragmode = 'select';
          break;
        case 'y':
          updateOptions.dragmode = 'select';
          break;
        case 'xy':
        default:
          updateOptions.dragmode = 'lasso';
          break;
      }
      
      // Update the chart with new options
      Plotly.update(container, {}, {dragmode: updateOptions.dragmode}, [0]);
      
      // Update plot layout to show selection options in mode bar
      const update = {
        config: {
          ...this.defaults.config,
          displayModeBar: true,
          ...updateOptions
        }
      };
      
      Plotly.react(container, plotlyContainer.data, plotlyContainer.layout, update.config);
    }
    
    /**
     * Update chart data dynamically
     * @param {HTMLElement} container - The chart container element
     * @param {Object} newData - New data to update the chart with
     */
    updateChartData(container, newData) {
      if (!this.plotlyAvailable || !container) return;
      
      // Get current data
      const plotlyContainer = container;
      const plotInstance = plotlyContainer.data;
      
      if (!plotInstance) return;
      
      // Update data for each trace
      Object.keys(newData).forEach((traceIndex) => {
        const traceData = newData[traceIndex];
        
        if (traceData.x) {
          Plotly.restyle(container, {'x': [traceData.x]}, [parseInt(traceIndex)]);
        }
        
        if (traceData.y) {
          Plotly.restyle(container, {'y': [traceData.y]}, [parseInt(traceIndex)]);
        }
        
        if (traceData.text) {
          Plotly.restyle(container, {'text': [traceData.text]}, [parseInt(traceIndex)]);
        }
        
        if (traceData.marker) {
          Plotly.restyle(container, {'marker': traceData.marker}, [parseInt(traceIndex)]);
        }
      });
    }
    
    /**
     * Add new traces to a chart
     * @param {HTMLElement} container - The chart container element
     * @param {Array} newTraces - New traces to add to the chart
     */
    addChartTraces(container, newTraces) {
      if (!this.plotlyAvailable || !container) return;
      
      // Add each new trace
      newTraces.forEach((trace) => {
        Plotly.addTraces(container, trace);
      });
    }
    
    /**
     * Remove traces from a chart
     * @param {HTMLElement} container - The chart container element
     * @param {Array} traceIndices - Indices of traces to remove
     */
    removeChartTraces(container, traceIndices) {
      if (!this.plotlyAvailable || !container) return;
      
      // Remove specified traces
      Plotly.deleteTraces(container, traceIndices);
    }
  }
  
  export default ChartInteractions;