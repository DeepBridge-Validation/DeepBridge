/**
 * Chart Themes
 * 
 * Provides predefined themes for charts to ensure consistent styling
 * across the robustness report visualization.
 */

class ChartThemes {
    /**
     * Initialize the chart themes module
     */
    constructor() {
      // Define common theme properties
      this.commonTheme = {
        font: {
          family: 'Roboto, Arial, sans-serif',
          size: 12
        },
        margin: {
          l: 60,
          r: 20,
          t: 40,
          b: 60
        },
        autosize: true,
        showlegend: true,
        legend: {
          orientation: 'h',
          xanchor: 'center',
          yanchor: 'top',
          x: 0.5,
          y: -0.15
        },
        hovermode: 'closest',
        transition: { duration: 300 }
      };
      
      // Define color palettes
      this.colorPalettes = {
        // Main color palette
        main: ['#1b78de', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#34495e'],
        
        // Alternative palettes
        cool: ['#3498db', '#1abc9c', '#2980b9', '#16a085', '#2c3e50', '#7f8c8d'],
        warm: ['#e74c3c', '#f39c12', '#d35400', '#c0392b', '#e67e22', '#f1c40f'],
        pastel: ['#74b9ff', '#55efc4', '#fab1a0', '#81ecec', '#a29bfe', '#ffeaa7'],
        dark: ['#2c3e50', '#34495e', '#7f8c8d', '#2c3e50', '#34495e', '#7f8c8d']
      };
      
      // Initialize themes
      this.themes = {
        light: this.createLightTheme(),
        dark: this.createDarkTheme(),
        minimal: this.createMinimalTheme(),
        corporate: this.createCorporateTheme(),
        presentation: this.createPresentationTheme()
      };
    }
    
    /**
     * Get a theme by name
     * @param {string} name - Theme name
     * @return {Object} Theme configuration
     */
    getTheme(name) {
      return this.themes[name] || this.themes.light;
    }
    
    /**
     * Apply a theme to a chart configuration
     * @param {Object} chartConfig - Chart configuration
     * @param {string} themeName - Theme name
     * @return {Object} Updated chart configuration
     */
    applyTheme(chartConfig, themeName = 'light') {
      // Get the theme
      const theme = this.getTheme(themeName);
      
      // Create a new configuration with theme applied
      const themedConfig = {
        ...chartConfig,
        layout: {
          ...theme.layout,
          ...chartConfig.layout,
          xaxis: {
            ...theme.layout.xaxis,
            ...(chartConfig.layout && chartConfig.layout.xaxis)
          },
          yaxis: {
            ...theme.layout.yaxis,
            ...(chartConfig.layout && chartConfig.layout.yaxis)
          }
        },
        config: {
          ...theme.config,
          ...chartConfig.config
        }
      };
      
      // Apply color palette if series are present
      if (chartConfig.series) {
        themedConfig.series = chartConfig.series.map((series, index) => ({
          ...series,
          color: series.color || theme.colorway[index % theme.colorway.length]
        }));
      }
      
      return themedConfig;
    }
    
    /**
     * Create the light theme
     * @return {Object} Light theme configuration
     */
    createLightTheme() {
      return {
        layout: {
          ...this.commonTheme,
          paper_bgcolor: 'rgba(255, 255, 255, 0)',
          plot_bgcolor: 'rgba(255, 255, 255, 0)',
          font: {
            ...this.commonTheme.font,
            color: '#2c3e50'
          },
          title: {
            font: {
              size: 16,
              color: '#2c3e50'
            }
          },
          xaxis: {
            title: {
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
            zerolinecolor: '#bdc3c7'
          },
          yaxis: {
            title: {
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
            zerolinecolor: '#bdc3c7'
          }
        },
        config: {
          responsive: true,
          displayModeBar: false,
          showTips: false
        },
        colorway: this.colorPalettes.main
      };
    }
    
    /**
     * Create the dark theme
     * @return {Object} Dark theme configuration
     */
    createDarkTheme() {
      return {
      layout: {
        ...this.commonTheme,
        paper_bgcolor: '#2c3e50',
        plot_bgcolor: '#34495e',
        font: {
          ...this.commonTheme.font,
          color: '#ecf0f1'
        },
        title: {
          font: {
            size: 16,
            color: '#ecf0f1'
          }
        },
        xaxis: {
          title: {
            font: {
              size: 14,
              color: '#bdc3c7'
            }
          },
          tickfont: {
            size: 12,
            color: '#bdc3c7'
          },
          gridcolor: '#7f8c8d',
          zerolinecolor: '#95a5a6'
        },
        yaxis: {
          title: {
            font: {
              size: 14,
              color: '#bdc3c7'
            }
          },
          tickfont: {
            size: 12,
            color: '#bdc3c7'
          },
          gridcolor: '#7f8c8d',
          zerolinecolor: '#95a5a6'
        }
      },
      config: {
        responsive: true,
        displayModeBar: false,
        showTips: false
      },
      colorway: this.colorPalettes.cool
    };
  }
  
  /**
   * Create the minimal theme
   * @return {Object} Minimal theme configuration
   */
  createMinimalTheme() {
    return {
      layout: {
        ...this.commonTheme,
        paper_bgcolor: 'rgba(255, 255, 255, 0)',
        plot_bgcolor: 'rgba(255, 255, 255, 0)',
        font: {
          ...this.commonTheme.font,
          color: '#2c3e50'
        },
        title: {
          font: {
            size: 16,
            color: '#2c3e50'
          }
        },
        xaxis: {
          title: {
            font: {
              size: 14,
              color: '#7f8c8d'
            }
          },
          tickfont: {
            size: 12,
            color: '#7f8c8d'
          },
          gridcolor: 'rgba(236, 240, 241, 0.5)',
          zerolinecolor: 'rgba(189, 195, 199, 0.5)',
          showgrid: false,
          zeroline: false
        },
        yaxis: {
          title: {
            font: {
              size: 14,
              color: '#7f8c8d'
            }
          },
          tickfont: {
            size: 12,
            color: '#7f8c8d'
          },
          gridcolor: 'rgba(236, 240, 241, 0.5)',
          zerolinecolor: 'rgba(189, 195, 199, 0.5)',
          showgrid: false,
          zeroline: false
        },
        showlegend: true,
        legend: {
          ...this.commonTheme.legend,
          borderwidth: 0,
          bgcolor: 'rgba(255, 255, 255, 0)'
        }
      },
      config: {
        responsive: true,
        displayModeBar: false,
        showTips: false
      },
      colorway: this.colorPalettes.pastel
    };
  }
  
  /**
   * Create the corporate theme
   * @return {Object} Corporate theme configuration
   */
  createCorporateTheme() {
    return {
      layout: {
        ...this.commonTheme,
        paper_bgcolor: 'rgba(255, 255, 255, 0)',
        plot_bgcolor: 'rgba(255, 255, 255, 0)',
        font: {
          family: 'Arial, Helvetica, sans-serif',
          size: 12,
          color: '#333333'
        },
        title: {
          font: {
            family: 'Arial, Helvetica, sans-serif',
            size: 16,
            color: '#333333',
            weight: 'bold'
          }
        },
        xaxis: {
          title: {
            font: {
              family: 'Arial, Helvetica, sans-serif',
              size: 14,
              color: '#666666'
            }
          },
          tickfont: {
            family: 'Arial, Helvetica, sans-serif',
            size: 12,
            color: '#666666'
          },
          gridcolor: '#e6e6e6',
          zerolinecolor: '#cccccc'
        },
        yaxis: {
          title: {
            font: {
              family: 'Arial, Helvetica, sans-serif',
              size: 14,
              color: '#666666'
            }
          },
          tickfont: {
            family: 'Arial, Helvetica, sans-serif',
            size: 12,
            color: '#666666'
          },
          gridcolor: '#e6e6e6',
          zerolinecolor: '#cccccc'
        },
        showlegend: true,
        legend: {
          ...this.commonTheme.legend,
          font: {
            family: 'Arial, Helvetica, sans-serif',
            size: 12,
            color: '#333333'
          },
          borderwidth: 1,
          bordercolor: '#e6e6e6',
          bgcolor: '#ffffff'
        }
      },
      config: {
        responsive: true,
        displayModeBar: false,
        showTips: false
      },
      colorway: ['#0066cc', '#009933', '#ff6600', '#cc0000', '#9933cc', '#666666']
    };
  }
  
  /**
   * Create the presentation theme
   * @return {Object} Presentation theme configuration
   */
  createPresentationTheme() {
    return {
      layout: {
        ...this.commonTheme,
        paper_bgcolor: 'rgba(255, 255, 255, 0)',
        plot_bgcolor: 'rgba(255, 255, 255, 0)',
        font: {
          family: 'Verdana, Geneva, sans-serif',
          size: 14,
          color: '#333333'
        },
        title: {
          font: {
            family: 'Verdana, Geneva, sans-serif',
            size: 20,
            color: '#333333',
            weight: 'bold'
          }
        },
        xaxis: {
          title: {
            font: {
              family: 'Verdana, Geneva, sans-serif',
              size: 16,
              color: '#666666'
            }
          },
          tickfont: {
            family: 'Verdana, Geneva, sans-serif',
            size: 14,
            color: '#666666'
          },
          gridcolor: '#e6e6e6',
          zerolinecolor: '#cccccc',
          tickwidth: 2,
          ticklen: 8
        },
        yaxis: {
          title: {
            font: {
              family: 'Verdana, Geneva, sans-serif',
              size: 16,
              color: '#666666'
            }
          },
          tickfont: {
            family: 'Verdana, Geneva, sans-serif',
            size: 14,
            color: '#666666'
          },
          gridcolor: '#e6e6e6',
          zerolinecolor: '#cccccc',
          tickwidth: 2,
          ticklen: 8
        },
        showlegend: true,
        legend: {
          ...this.commonTheme.legend,
          font: {
            family: 'Verdana, Geneva, sans-serif',
            size: 14,
            color: '#333333'
          },
          borderwidth: 0,
          bgcolor: 'rgba(255, 255, 255, 0.8)'
        },
        margin: {
          l: 80,
          r: 40,
          t: 60,
          b: 80
        }
      },
      config: {
        responsive: true,
        displayModeBar: true,
        showTips: true
      },
      colorway: ['#3366cc', '#dc3912', '#ff9900', '#109618', '#990099', '#0099c6']
    };
  }
  
  /**
   * Create a custom theme based on a base theme
   * @param {string} baseTheme - Base theme name
   * @param {Object} customOptions - Custom theme options
   * @return {Object} Custom theme configuration
   */
  createCustomTheme(baseTheme, customOptions) {
    // Get the base theme
    const base = this.getTheme(baseTheme);
    
    // Create custom theme by merging base theme with custom options
    return {
      layout: {
        ...base.layout,
        ...(customOptions.layout || {}),
        xaxis: {
          ...base.layout.xaxis,
          ...(customOptions.layout && customOptions.layout.xaxis)
        },
        yaxis: {
          ...base.layout.yaxis,
          ...(customOptions.layout && customOptions.layout.yaxis)
        }
      },
      config: {
        ...base.config,
        ...(customOptions.config || {})
      },
      colorway: customOptions.colorway || base.colorway
    };
  }
}

export default ChartThemes;