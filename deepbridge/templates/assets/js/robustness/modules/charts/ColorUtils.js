/**
 * Color Utilities
 * 
 * Utility functions for working with colors, including generating color scales,
 * interpolating between colors, and creating accessible color palettes.
 */

class ColorUtils {
    /**
     * Initialize the color utilities module
     */
    constructor() {
      // Define preset color scales
      this.colorScales = {
        // Sequential color scales
        blues: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
        greens: ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
        reds: ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],
        purples: ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d'],
        oranges: ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704'],
        
        // Diverging color scales
        redBlue: ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#f7f7f7', '#92c5de', '#4393c3', '#2166ac', '#053061'],
        redGreen: ['#a50026', '#d73027', '#f46d43', '#fdae61', '#ffffbf', '#a6d96a', '#66bd63', '#1a9850', '#006837'],
        purpleGreen: ['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#f7f7f7', '#a6dba0', '#5aae61', '#1b7837', '#00441b'],
        
        // Categorical color scales
        category10: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        pastel: ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'],
        colorblind: ['#0072b2', '#e69f00', '#009e73', '#f0e442', '#cc79a7', '#56b4e9', '#d55e00', '#000000']
      };
    }
    
    /**
     * Get a color scale by name
     * @param {string} name - Name of the color scale
     * @return {Array} Array of color hex strings
     */
    getColorScale(name) {
      return this.colorScales[name] || this.colorScales.category10;
    }
    
    /**
     * Interpolate between two colors
     * @param {string} color1 - Starting color (hex or rgba)
     * @param {string} color2 - Ending color (hex or rgba)
     * @param {number} factor - Interpolation factor (0-1)
     * @return {string} Interpolated color in rgba format
     */
    interpolateColor(color1, color2, factor) {
      // Convert hex to rgba if needed
      const rgba1 = this.hexToRgba(color1);
      const rgba2 = this.hexToRgba(color2);
      
      // Interpolate between the colors
      const result = rgba1.map((value, index) => {
        return Math.round(value + factor * (rgba2[index] - value));
      });
      
      // Return as rgba string
      return `rgba(${result[0]}, ${result[1]}, ${result[2]}, ${result[3]})`;
    }
    
    /**
     * Create a color gradient function for a value range
     * @param {number} min - Minimum value
     * @param {number} max - Maximum value
     * @param {Array} colors - Array of color hex strings
     * @return {Function} Function that returns color for a value
     */
    createColorGradient(min, max, colors) {
      const range = max - min;
      
      return (value) => {
        // Handle values outside the range
        if (value <= min) return colors[0];
        if (value >= max) return colors[colors.length - 1];
        
        // Calculate position in the range
        const position = (value - min) / range;
        const scaledPosition = position * (colors.length - 1);
        
        // Get the two colors to interpolate between
        const index = Math.floor(scaledPosition);
        const factor = scaledPosition - index;
        
        return this.interpolateColor(colors[index], colors[index + 1], factor);
      };
    }
    
    /**
     * Convert a hex color to rgba
     * @param {string} color - Color in hex or rgba format
     * @return {Array} RGBA values as [r, g, b, a]
     */
    hexToRgba(color) {
      // Handle rgba format
      if (color.startsWith('rgba')) {
        return color.match(/[\d.]+/g).map(Number);
      }
      
      // Handle hex format
      let hex = color.replace('#', '');
      
      // Handle short hex format (#123)
      if (hex.length === 3) {
        hex = hex.split('').map(char => char + char).join('');
      }
      
      // Convert to rgb values
      const r = parseInt(hex.substring(0, 2), 16);
      const g = parseInt(hex.substring(2, 4), 16);
      const b = parseInt(hex.substring(4, 6), 16);
      const a = 1;
      
      return [r, g, b, a];
    }
    
    /**
     * Generate a sequential color scale
     * @param {string} baseColor - Base color (hex)
     * @param {number} steps - Number of steps in the scale
     * @return {Array} Array of color hex strings
     */
    generateSequentialScale(baseColor, steps = 9) {
      const rgba = this.hexToRgba(baseColor);
      const scale = [];
      
      for (let i = 0; i < steps; i++) {
        const factor = i / (steps - 1);
        
        // Create a gradient from white to the base color
        const color = this.interpolateColor(
          'rgba(255, 255, 255, 1)', 
          `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, 1)`, 
          factor
        );
        
        scale.push(color);
      }
      
      return scale;
    }
    
    /**
     * Generate a bivariate color scale (two-dimensional)
     * @param {string} xColor - Color for X axis (hex)
     * @param {string} yColor - Color for Y axis (hex)
     * @param {number} steps - Number of steps in each dimension
     * @return {Array} 2D array of color hex strings
     */
    generateBivariateScale(xColor, yColor, steps = 5) {
      const scale = [];
      
      for (let i = 0; i < steps; i++) {
        const row = [];
        const xFactor = i / (steps - 1);
        
        for (let j = 0; j < steps; j++) {
          const yFactor = j / (steps - 1);
          
          // Create a bivariate color by blending the two colors
          const xColorRgba = this.hexToRgba(xColor);
          const yColorRgba = this.hexToRgba(yColor);
          
          const blendedColor = [
            Math.min(255, Math.round(xColorRgba[0] * xFactor + yColorRgba[0] * yFactor)),
            Math.min(255, Math.round(xColorRgba[1] * xFactor + yColorRgba[1] * yFactor)),
            Math.min(255, Math.round(xColorRgba[2] * xFactor + yColorRgba[2] * yFactor)),
            1
          ];
          
          row.push(`rgba(${blendedColor[0]}, ${blendedColor[1]}, ${blendedColor[2]}, ${blendedColor[3]})`);
        }
        
        scale.push(row);
      }
      
      return scale;
    }
    
    /**
     * Get a color for a value in a range
     * @param {number} value - The value to get a color for
     * @param {number} min - Minimum value in the range
     * @param {number} max - Maximum value in the range
     * @param {string} scale - Name of the color scale or array of colors
     * @return {string} Color for the value
     */
    getColorForValue(value, min, max, scale = 'blues') {
      // Get the color scale
      const colors = Array.isArray(scale) ? scale : this.getColorScale(scale);
      
      // Create a color gradient function
      const gradient = this.createColorGradient(min, max, colors);
      
      // Return the color for the value
      return gradient(value);
    }
    
    /**
     * Generate a contrasting text color (black or white) for a background color
     * @param {string} backgroundColor - Background color (hex or rgba)
     * @return {string} Contrasting text color (black or white)
     */
    getContrastingTextColor(backgroundColor) {
      const rgba = this.hexToRgba(backgroundColor);
      
      // Calculate relative luminance
      const luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2];
      
      // Return black for light backgrounds, white for dark backgrounds
      return luminance > 128 ? '#000000' : '#ffffff';
    }
  }
  
  export default ColorUtils;