// Colors.js - placeholder
/**
 * Colors.js - Color manipulation utilities for robustness visualization
 * Provides helper functions for working with colors in charts and visualizations
 */

// Color palettes for different types of data visualization
const COLOR_PALETTES = {
    // General purpose palette for categorical data
    categorical: [
      '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f',
      '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab'
    ],
    
    // Sequential color scales
    sequential: {
      blues: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
      greens: ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
      reds: ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],
      purples: ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
    },
    
    // Diverging color scales
    diverging: {
      redBlue: ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#f7f7f7', '#92c5de', '#4393c3', '#2166ac', '#053061'],
      purpleGreen: ['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#f7f7f7', '#a6dba0', '#5aae61', '#1b7837', '#00441b'],
      brownTeal: ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e']
    },
    
    // Specific colors for performance indicators
    performance: {
      excellent: '#1a9850',
      good: '#91cf60',
      moderate: '#d9ef8b',
      fair: '#fee08b',
      poor: '#fc8d59',
      critical: '#d73027'
    }
  };
  
  /**
   * Interpolates between two colors based on a factor
   * @param {string} color1 - Starting color in hex format
   * @param {string} color2 - Ending color in hex format
   * @param {number} factor - Value between 0 and 1 
   * @returns {string} - Interpolated color in hex format
   */
  function interpolateColor(color1, color2, factor) {
    if (factor < 0) factor = 0;
    if (factor > 1) factor = 1;
    
    const c1 = hexToRgb(color1);
    const c2 = hexToRgb(color2);
    
    const r = Math.round(c1.r + factor * (c2.r - c1.r));
    const g = Math.round(c1.g + factor * (c2.g - c1.g));
    const b = Math.round(c1.b + factor * (c2.b - c1.b));
    
    return rgbToHex(r, g, b);
  }
  
  /**
   * Converts hex color to RGB object
   * @param {string} hex - Hex color code
   * @returns {Object} - RGB object with r, g, b properties
   */
  function hexToRgb(hex) {
    // Remove hash if present
    hex = hex.replace(/^#/, '');
    
    // Parse the hex value
    const bigint = parseInt(hex, 16);
    
    return {
      r: (bigint >> 16) & 255,
      g: (bigint >> 8) & 255,
      b: bigint & 255
    };
  }
  
  /**
   * Converts RGB values to hex color string
   * @param {number} r - Red value (0-255)
   * @param {number} g - Green value (0-255)
   * @param {number} b - Blue value (0-255)
   * @returns {string} - Hex color code
   */
  function rgbToHex(r, g, b) {
    return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
  }
  
  /**
   * Generates a color scale based on a value
   * @param {number} value - Value between 0 and 1
   * @param {string} type - Scale type ('performance', 'impact', or 'custom')
   * @param {Object} options - Custom scale options
   * @returns {string} - Color in hex format
   */
  function getColorForValue(value, type = 'performance', options = {}) {
    if (value < 0) value = 0;
    if (value > 1) value = 1;
    
    let colors;
    let thresholds;
    
    switch (type) {
      case 'performance':
        // Higher values are better
        colors = [
          COLOR_PALETTES.performance.critical,
          COLOR_PALETTES.performance.poor,
          COLOR_PALETTES.performance.fair,
          COLOR_PALETTES.performance.moderate,
          COLOR_PALETTES.performance.good,
          COLOR_PALETTES.performance.excellent
        ];
        thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1];
        break;
        
      case 'impact':
        // Lower values are better (impacts, degradation, etc.)
        colors = [
          COLOR_PALETTES.performance.excellent,
          COLOR_PALETTES.performance.good,
          COLOR_PALETTES.performance.moderate,
          COLOR_PALETTES.performance.fair,
          COLOR_PALETTES.performance.poor,
          COLOR_PALETTES.performance.critical
        ];
        thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1];
        break;
        
      case 'custom':
        colors = options.colors || COLOR_PALETTES.sequential.blues;
        thresholds = options.thresholds || Array.from({length: colors.length}, (_, i) => i / (colors.length - 1));
        break;
        
      default:
        colors = COLOR_PALETTES.sequential.blues;
        thresholds = Array.from({length: colors.length}, (_, i) => i / (colors.length - 1));
    }
    
    // Find the appropriate color segment
    for (let i = 1; i < thresholds.length; i++) {
      if (value <= thresholds[i]) {
        const rangeMin = thresholds[i - 1];
        const rangeMax = thresholds[i];
        const rangeFactor = (value - rangeMin) / (rangeMax - rangeMin);
        
        return interpolateColor(colors[i - 1], colors[i], rangeFactor);
      }
    }
    
    return colors[colors.length - 1];
  }
  
  /**
   * Adjusts the brightness of a color
   * @param {string} hex - Hex color code
   * @param {number} factor - Adjustment factor (-1 to 1, negative darkens, positive brightens)
   * @returns {string} - Adjusted hex color
   */
  function adjustBrightness(hex, factor) {
    const rgb = hexToRgb(hex);
    
    // Adjust brightness
    let r = rgb.r + Math.round(rgb.r * factor);
    let g = rgb.g + Math.round(rgb.g * factor);
    let b = rgb.b + Math.round(rgb.b * factor);
    
    // Ensure values stay in range
    r = Math.max(0, Math.min(255, r));
    g = Math.max(0, Math.min(255, g));
    b = Math.max(0, Math.min(255, b));
    
    return rgbToHex(r, g, b);
  }
  
  /**
   * Generates a contrasting text color (black or white) based on background color
   * @param {string} backgroundColor - Hex color code
   * @returns {string} - '#000000' for dark text or '#ffffff' for light text
   */
  function getContrastTextColor(backgroundColor) {
    const rgb = hexToRgb(backgroundColor);
    
    // Calculate relative luminance using the sRGB color space formula
    // See: https://www.w3.org/TR/WCAG20/#relativeluminancedef
    const r = rgb.r / 255;
    const g = rgb.g / 255;
    const b = rgb.b / 255;
    
    const R = r <= 0.03928 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4);
    const G = g <= 0.03928 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4);
    const B = b <= 0.03928 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4);
    
    const luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B;
    
    // Return white for dark backgrounds, black for light backgrounds
    return luminance > 0.5 ? '#000000' : '#ffffff';
  }
  
  // Export the utilities
  export {
    COLOR_PALETTES,
    interpolateColor,
    hexToRgb,
    rgbToHex,
    getColorForValue,
    adjustBrightness,
    getContrastTextColor
  };