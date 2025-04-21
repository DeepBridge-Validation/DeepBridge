// Colors.js - placeholder
/**
 * Colors Utility
 * 
 * Provides color generation and manipulation functions for
 * visualizations in the uncertainty report.
 */

class Colors {
    /**
     * Initialize the colors utility
     */
    constructor() {
      // Define color palettes
      this.palettes = {
        // Main color palette (blues with accent colors for uncertainty)
        main: ['#1b78de', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#2ecc71', '#f39c12', '#e74c3c'],
        
        // Alternative palettes
        cool: ['#3498db', '#1abc9c', '#2980b9', '#16a085', '#2c3e50', '#7f8c8d'],
        warm: ['#e74c3c', '#f39c12', '#d35400', '#c0392b', '#e67e22', '#f1c40f'],
        uncertainty: ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#34495e'],
        confidence: ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
      };
      
      // Define color ranges for specific metrics
      this.ranges = {
        // Calibration (better is higher)
        calibration: {
          excellent: '#2ecc71', // Green
          good: '#27ae60',      // Dark Green
          moderate: '#f1c40f',  // Yellow
          fair: '#f39c12',      // Orange
          poor: '#e74c3c'       // Red
        },
        
        // Coverage (better is closer to target)
        coverage: {
          excellent: '#2ecc71', // Green
          good: '#27ae60',      // Dark Green
          moderate: '#f1c40f',  // Yellow
          fair: '#f39c12',      // Orange
          poor: '#e74c3c'       // Red
        },
        
        // Interval width (narrower is generally better)
        width: {
          excellent: '#2ecc71', // Green (narrow)
          good: '#27ae60',      // Dark Green
          moderate: '#f1c40f',  // Yellow
          fair: '#f39c12',      // Orange
          poor: '#e74c3c'       // Red (wide)
        }
      };
    }
    
    /**
     * Get a color for a specific value from a palette
     * @param {number} value - Value to map to a color (0-1)
     * @param {string|Array} palette - Palette name or array of colors
     * @return {string} Color hex code
     */
    getColorForValue(value, palette = 'main') {
      // Get the palette (either by name or directly)
      const colors = Array.isArray(palette) ? palette : this.palettes[palette] || this.palettes.main;
      
      // Normalize value to 0-1 range
      const normalizedValue = Math.max(0, Math.min(1, value));
      
      // Map to color index
      const index = Math.min(
        colors.length - 1, 
        Math.floor(normalizedValue * colors.length)
      );
      
      return colors[index];
    }
    
    /**
     * Get a color for a calibration score
     * @param {number} score - Calibration score (0-1)
     * @return {string} Color hex code
     */
    getCalibrationColor(score) {
      if (score >= 0.9) return this.ranges.calibration.excellent;
      if (score >= 0.75) return this.ranges.calibration.good;
      if (score >= 0.6) return this.ranges.calibration.moderate;
      if (score >= 0.4) return this.ranges.calibration.fair;
      return this.ranges.calibration.poor;
    }
    
    /**
     * Get a color for a coverage score
     * @param {number} actual - Actual coverage (0-1)
     * @param {number} target - Target coverage (0-1)
     * @return {string} Color hex code
     */
    getCoverageColor(actual, target) {
      // Calculate absolute error
      const error = Math.abs(actual - target);
      
      // Color based on error magnitude
      if (error <= 0.01) return this.ranges.coverage.excellent;
      if (error <= 0.05) return this.ranges.coverage.good;
      if (error <= 0.1) return this.ranges.coverage.moderate;
      if (error <= 0.15) return this.ranges.coverage.fair;
      return this.ranges.coverage.poor;
    }
    
    /**
     * Get a color for an interval width
     * @param {number} width - Interval width
     * @param {number} maxWidth - Maximum expected width
     * @return {string} Color hex code
     */
    getIntervalWidthColor(width, maxWidth) {
      // Normalize width to 0-1 range
      const normalizedWidth = Math.min(1, width / maxWidth);
      
      // Invert so that smaller widths get better colors
      const score = 1 - normalizedWidth;
      
      if (score >= 0.8) return this.ranges.width.excellent;
      if (score >= 0.6) return this.ranges.width.good;
      if (score >= 0.4) return this.ranges.width.moderate;
      if (score >= 0.2) return this.ranges.width.fair;
      return this.ranges.width.poor;
    }
    
    /**
     * Get a color for uncertainty score
     * @param {number} score - Uncertainty score (0-1)
     * @return {Object} Color information
     */
    getUncertaintyScoreColor(score) {
      let color;
      let textColor = '#ffffff'; // Default white text
      let label;
      
      if (score >= 0.9) {
        color = '#2ecc71'; // Green
        label = 'Excellent';
      } else if (score >= 0.75) {
        color = '#27ae60'; // Dark Green
        label = 'Good';
      } else if (score >= 0.6) {
        color = '#f1c40f'; // Yellow
        textColor = '#333333'; // Dark text for light background
        label = 'Moderate';
      } else if (score >= 0.4) {
        color = '#f39c12'; // Orange
        label = 'Fair';
      } else if (score >= 0.2) {
        color = '#e74c3c'; // Red
        label = 'Poor';
      } else {
        color = '#c0392b'; // Dark Red
        label = 'Critical';
      }
      
      return {
        color,
        textColor,
        label
      };
    }
    
    /**
     * Generate a color gradient between two colors
     * @param {string} color1 - Starting color (hex)
     * @param {string} color2 - Ending color (hex)
     * @param {number} steps - Number of gradient steps
     * @return {Array} Array of hex color strings
     */
    generateGradient(color1, color2, steps) {
      // Convert hex to RGB
      const rgb1 = this.hexToRgb(color1);
      const rgb2 = this.hexToRgb(color2);
      
      const gradient = [];
      
      // Generate gradient steps
      for (let i = 0; i < steps; i++) {
        const ratio = i / (steps - 1);
        
        // Interpolate RGB components
        const r = Math.round(rgb1.r + ratio * (rgb2.r - rgb1.r));
        const g = Math.round(rgb1.g + ratio * (rgb2.g - rgb1.g));
        const b = Math.round(rgb1.b + ratio * (rgb2.b - rgb1.b));
        
        // Convert back to hex
        gradient.push(this.rgbToHex(r, g, b));
      }
      
      return gradient;
    }
    
    /**
     * Convert hex color to RGB
     * @param {string} hex - Hex color string
     * @return {Object} RGB values
     */
    hexToRgb(hex) {
      // Remove # if present
      hex = hex.replace('#', '');
      
      // Handle shorthand hex
      if (hex.length === 3) {
        hex = hex.split('').map(c => c + c).join('');
      }
      
      // Parse hex values
      const r = parseInt(hex.substring(0, 2), 16);
      const g = parseInt(hex.substring(2, 4), 16);
      const b = parseInt(hex.substring(4, 6), 16);
      
      return { r, g, b };
    }
    
    /**
     * Convert RGB values to hex color
     * @param {number} r - Red component (0-255)
     * @param {number} g - Green component (0-255)
     * @param {number} b - Blue component (0-255)
     * @return {string} Hex color string
     */
    rgbToHex(r, g, b) {
      // Ensure values are in valid range
      r = Math.max(0, Math.min(255, Math.round(r)));
      g = Math.max(0, Math.min(255, Math.round(g)));
      b = Math.max(0, Math.min(255, Math.round(b)));
      
      // Convert to hex
      const componentToHex = c => {
        const hex = c.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
      };
      
      return `#${componentToHex(r)}${componentToHex(g)}${componentToHex(b)}`;
    }
    
    /**
     * Get a contrasting text color (black or white) for a background
     * @param {string} backgroundColor - Background hex color
     * @return {string} Text color (black or white)
     */
    getContrastColor(backgroundColor) {
      const rgb = this.hexToRgb(backgroundColor);
      
      // Calculate luminance (perceived brightness)
      // Formula: 0.299*R + 0.587*G + 0.114*B
      const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
      
      // Return white for dark backgrounds, black for light backgrounds
      return luminance > 0.5 ? '#000000' : '#ffffff';
    }
    
    /**
     * Generate HTML for a color badge
     * @param {string} text - Badge text
     * @param {string} color - Badge background color
     * @param {string} textColor - Badge text color
     * @return {string} HTML for badge
     */
    generateBadgeHtml(text, color, textColor = null) {
      // Determine text color if not provided
      const finalTextColor = textColor || this.getContrastColor(color);
      
      return `
        <span class="color-badge" style="
          background-color: ${color};
          color: ${finalTextColor};
          padding: 2px 6px;
          border-radius: 3px;
          font-weight: bold;
          font-size: 0.85em;
          display: inline-block;
        ">${text}</span>
      `;
    }
  }
  
  export default Colors;