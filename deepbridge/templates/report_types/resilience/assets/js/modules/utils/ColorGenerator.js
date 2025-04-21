// ColorGenerator.js - placeholder
/**
 * Color Generator
 * 
 * Utility for generating color schemes, palettes, and color-related
 * functions for visualizations in the resilience report.
 */

class ColorGenerator {
    /**
     * Initialize the color generator
     */
    constructor() {
      // Pre-defined color palettes for different visualization types
      this.palettes = {
        // Main palette for primary charts
        main: [
          '#3366cc', '#dc3912', '#ff9900', '#109618', 
          '#990099', '#0099c6', '#dd4477', '#66aa00'
        ],
        
        // Categorical color scales
        categorical: {
          blues: ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
          greens: ['#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'],
          reds: ['#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d'],
          purples: ['#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486']
        },
        
        // Sequential color scales (light to dark)
        sequential: {
          blues: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
          greens: ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
          reds: ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],
          purples: ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
        },
        
        // Diverging color scales (negative to positive)
        diverging: {
          redBlue: ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'],
          redGrey: ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#e0e0e0', '#bababa', '#878787', '#4d4d4d', '#1a1a1a'],
          redGreen: ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837']
        },
        
        // Special purpose palettes
        resilience: {
          score: ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63'], // Low to high resilience
          impact: ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#e34a33', '#b30000']  // Low to high impact
        }
      };
    }
    
    /**
     * Get a color palette by name
     * @param {string} name - Palette name
     * @return {Array} Array of color strings
     */
    getPalette(name) {
      if (name.includes('.')) {
        // Handle nested palettes like 'sequential.blues'
        const [category, palette] = name.split('.');
        return this.palettes[category] && this.palettes[category][palette] ? 
               this.palettes[category][palette] : this.palettes.main;
      }
      
      return this.palettes[name] || this.palettes.main;
    }
    
    /**
     * Generate a color for a specific value in a range
     * @param {number} value - Value to map to a color
     * @param {number} min - Minimum value in the range
     * @param {number} max - Maximum value in the range
     * @param {string|Array} palette - Palette name or array of colors
     * @return {string} Color for the value
     */
    getColorForValue(value, min, max, palette = 'resilience.score') {
      // Get the palette
      const colors = Array.isArray(palette) ? palette : this.getPalette(palette);
      
      // Normalize the value to [0, 1]
      const normalizedValue = Math.max(0, Math.min(1, (value - min) / (max - min)));
      
      // Map to color index
      const index = Math.min(
        colors.length - 1, 
        Math.floor(normalizedValue * colors.length)
      );
      
      return colors[index];
    }
    
    /**
     * Generate a gradient color between two colors
     * @param {string} color1 - Start color (hex or rgb)
     * @param {string} color2 - End color (hex or rgb)
     * @param {number} ratio - Ratio between colors (0-1)
     * @return {string} Interpolated color
     */
    interpolateColor(color1, color2, ratio) {
      // Convert colors to RGB
      const rgb1 = this.toRGB(color1);
      const rgb2 = this.toRGB(color2);
      
      // Interpolate each RGB component
      const r = Math.round(rgb1.r + ratio * (rgb2.r - rgb1.r));
      const g = Math.round(rgb1.g + ratio * (rgb2.g - rgb1.g));
      const b = Math.round(rgb1.b + ratio * (rgb2.b - rgb1.b));
      
      // Convert back to hex
      return this.toHex({ r, g, b });
    }
    
    /**
     * Generate a multi-stop gradient array
     * @param {Array} colors - Array of color stops
     * @param {number} steps - Number of gradient steps
     * @return {Array} Array of gradient colors
     */
    generateGradient(colors, steps) {
      if (!colors || colors.length < 2 || steps < 2) {
        return colors || ['#000000'];
      }
      
      const gradient = [];
      
      // Calculate number of segments based on color stops
      const segments = colors.length - 1;
      const stepsPerSegment = Math.floor(steps / segments);
      
      // Generate gradient for each segment
      for (let i = 0; i < segments; i++) {
        const color1 = colors[i];
        const color2 = colors[i + 1];
        
        const segmentSteps = (i === segments - 1) ? 
          steps - (stepsPerSegment * i) : stepsPerSegment;
        
        for (let j = 0; j < segmentSteps; j++) {
          const ratio = j / (segmentSteps - 1);
          gradient.push(this.interpolateColor(color1, color2, ratio));
        }
      }
      
      return gradient;
    }
    
    /**
     * Convert color to RGB object
     * @param {string} color - Color in hex or rgb format
     * @return {Object} RGB object with r, g, b properties
     */
    toRGB(color) {
      // Check if already in rgb/rgba format
      if (color.startsWith('rgb')) {
        const match = color.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)$/);
        if (match) {
          return {
            r: parseInt(match[1], 10),
            g: parseInt(match[2], 10),
            b: parseInt(match[3], 10)
          };
        }
      }
      
      // Convert hex to rgb
      let hex = color.replace('#', '');
      
      // Handle shorthand hex (#rgb)
      if (hex.length === 3) {
        hex = hex.split('').map(c => c + c).join('');
      }
      
      return {
        r: parseInt(hex.substring(0, 2), 16),
        g: parseInt(hex.substring(2, 4), 16),
        b: parseInt(hex.substring(4, 6), 16)
      };
    }
    
    /**
     * Convert RGB object to hex color
     * @param {Object} rgb - RGB object with r, g, b properties
     * @return {string} Hex color string
     */
    toHex(rgb) {
      return '#' + [rgb.r, rgb.g, rgb.b]
        .map(v => Math.max(0, Math.min(255, v)).toString(16).padStart(2, '0'))
        .join('');
    }
    
    /**
     * Generate contrasting text color (black or white) for a background
     * @param {string} backgroundColor - Background color
     * @return {string} Text color (#000000 or #ffffff)
     */
    getContrastColor(backgroundColor) {
      const rgb = this.toRGB(backgroundColor);
      
      // Calculate luminance
      // Formula: 0.299*R + 0.587*G + 0.114*B
      const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
      
      // Use white text for dark backgrounds, black for light backgrounds
      return luminance > 0.5 ? '#000000' : '#ffffff';
    }
    
    /**
     * Generate a unique color for a category or feature
     * @param {string} name - Category or feature name
     * @param {Array} existingColors - Array of already used colors
     * @return {string} Unique color for the category
     */
    getCategoryColor(name, existingColors = []) {
      // Use hash function to deterministically generate a color from a name
      let hash = 0;
      for (let i = 0; i < name.length; i++) {
        hash = name.charCodeAt(i) + ((hash << 5) - hash);
      }
      
      // Generate hex color from hash
      let color = '#';
      for (let i = 0; i < 3; i++) {
        const value = (hash >> (i * 8)) & 0xFF;
        color += value.toString(16).padStart(2, '0');
      }
      
      // Check if color is already used and is too similar to existing colors
      if (existingColors && existingColors.length > 0) {
        const rgb = this.toRGB(color);
        
        // Check color similarity
        const isTooSimilar = existingColors.some(existingColor => {
          const existingRGB = this.toRGB(existingColor);
          
          // Calculate Euclidean distance in RGB space
          const distance = Math.sqrt(
            Math.pow(rgb.r - existingRGB.r, 2) +
            Math.pow(rgb.g - existingRGB.g, 2) +
            Math.pow(rgb.b - existingRGB.b, 2)
          );
          
          // Colors are too similar if distance is less than threshold
          return distance < 100;
        });
        
        if (isTooSimilar) {
          // Modify the color to make it more distinct
          const r = (rgb.r + 128) % 256;
          const g = (rgb.g + 128) % 256;
          const b = (rgb.b + 128) % 256;
          
          color = this.toHex({ r, g, b });
        }
      }
      
      return color;
    }
    
    /**
     * Get color for resilience score
     * @param {number} score - Resilience score (0-1)
     * @return {Object} Color information
     */
    getResilienceColor(score) {
      const palette = this.palettes.resilience.score;
      const color = this.getColorForValue(score, 0, 1, palette);
      const textColor = this.getContrastColor(color);
      
      return {
        color,
        textColor,
        label: this.getResilienceLabel(score)
      };
    }
    
    /**
     * Get color for impact level
     * @param {number} impact - Impact level (0-1)
     * @return {Object} Color information
     */
    getImpactColor(impact) {
      const palette = this.palettes.resilience.impact;
      const color = this.getColorForValue(1 - impact, 0, 1, palette);
      const textColor = this.getContrastColor(color);
      
      return {
        color,
        textColor,
        label: this.getImpactLabel(impact)
      };
    }
    
    /**
     * Get resilience rating label
     * @param {number} score - Resilience score (0-1)
     * @return {string} Rating label
     */
    getResilienceLabel(score) {
      if (score >= 0.9) return 'Excellent';
      if (score >= 0.75) return 'Good';
      if (score >= 0.6) return 'Moderate';
      if (score >= 0.4) return 'Fair';
      if (score >= 0.2) return 'Poor';
      return 'Critical';
    }
    
    /**
     * Get impact level label
     * @param {number} impact - Impact level (0-1)
     * @return {string} Impact label
     */
    getImpactLabel(impact) {
      if (impact <= 0.1) return 'Minimal';
      if (impact <= 0.2) return 'Low';
      if (impact <= 0.3) return 'Moderate';
      if (impact <= 0.5) return 'Significant';
      if (impact <= 0.7) return 'High';
      return 'Severe';
    }
  }
  
  export default ColorGenerator;