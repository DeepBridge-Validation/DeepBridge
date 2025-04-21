// UncertaintyData.js - placeholder
/**
 * Uncertainty Data Handler
 * 
 * Specialized handler for extracting and processing uncertainty data
 * from the report structure.
 */

class UncertaintyData {
    /**
     * Initialize the uncertainty data handler
     */
    constructor() {
      // Default alpha levels for confidence intervals
      this.defaultAlphaLevels = [0.01, 0.05, 0.1, 0.2, 0.5];
    }
    
    /**
     * Extract uncertainty data from report data
     * @param {Object} reportData - The full report data object
     * @return {Object|null} Extracted uncertainty data or null if not available
     */
    extract(reportData) {
      if (!reportData) return null;
      
      // Extract key components of uncertainty data
      const uncertaintyData = {
        calibration: this.extractCalibrationData(reportData),
        coverage: this.extractCoverageData(reportData),
        intervalWidths: this.extractIntervalWidthData(reportData),
        alphaDetails: this.extractAlphaLevelDetails(reportData),
        
        // Calculate overall metrics
        calibrationError: null,
        averageCoverage: null,
        averageIntervalWidth: null,
        uncertaintyScore: null
      };
      
      // Calculate derived metrics
      uncertaintyData.calibrationError = this.calculateCalibrationError(uncertaintyData.calibration);
      uncertaintyData.averageCoverage = this.calculateAverageCoverage(uncertaintyData.coverage);
      uncertaintyData.averageIntervalWidth = this.calculateAverageIntervalWidth(uncertaintyData.intervalWidths);
      
      // Calculate overall uncertainty score
      uncertaintyData.uncertaintyScore = this.calculateUncertaintyScore(
        uncertaintyData.calibrationError,
        uncertaintyData.averageCoverage,
        uncertaintyData.averageIntervalWidth
      );
      
      return uncertaintyData;
    }
    
    /**
     * Extract calibration data from report data
     * @param {Object} reportData - The full report data object
     * @return {Object|null} Calibration data or null if not available
     */
    extractCalibrationData(reportData) {
      if (!reportData || !reportData.calibration) return null;
      
      const calibrationData = reportData.calibration;
      
      // Process the calibration bins data if available
      const bins = [];
      if (Array.isArray(calibrationData.bins)) {
        calibrationData.bins.forEach(bin => {
          bins.push({
            lowerBound: bin.lower_bound,
            upperBound: bin.upper_bound,
            predictedProbability: bin.predicted_probability,
            observedFrequency: bin.observed_frequency,
            sampleCount: bin.sample_count
          });
        });
      }
      
      return {
        bins: bins,
        error: calibrationData.error || null,
        method: calibrationData.method || "expected_calibration_error",
        isCalibrated: calibrationData.is_calibrated || false
      };
    }
    
    /**
     * Extract coverage data from report data
     * @param {Object} reportData - The full report data object
     * @return {Object|null} Coverage data or null if not available
     */
    extractCoverageData(reportData) {
      if (!reportData || !reportData.coverage) return null;
      
      const coverageData = reportData.coverage;
      
      // Process alpha levels and their corresponding coverages
      const alphaCoverage = {};
      if (coverageData.alpha_coverage && typeof coverageData.alpha_coverage === 'object') {
        Object.entries(coverageData.alpha_coverage).forEach(([alpha, coverage]) => {
          alphaCoverage[alpha] = coverage;
        });
      }
      
      return {
        alphaCoverage: alphaCoverage,
        averageCoverage: coverageData.average_coverage || null,
        method: coverageData.method || "empirical"
      };
    }
    
    /**
     * Extract interval width data from report data
     * @param {Object} reportData - The full report data object
     * @return {Object|null} Interval width data or null if not available
     */
    extractIntervalWidthData(reportData) {
      if (!reportData || !reportData.interval_widths) return null;
      
      const intervalData = reportData.interval_widths;
      
      // Process alpha levels and their corresponding interval widths
      const alphaWidths = {};
      if (intervalData.alpha_widths && typeof intervalData.alpha_widths === 'object') {
        Object.entries(intervalData.alpha_widths).forEach(([alpha, width]) => {
          alphaWidths[alpha] = width;
        });
      }
      
      return {
        alphaWidths: alphaWidths,
        averageWidth: intervalData.average_width || null,
        method: intervalData.method || "prediction_interval"
      };
    }
    
    /**
     * Extract alpha level details from report data
     * @param {Object} reportData - The full report data object
     * @return {Object} Alpha level details
     */
    extractAlphaLevelDetails(reportData) {
      if (!reportData || !reportData.alpha_details) return {};
      
      const alphaDetails = {};
      
      // Process details for each alpha level
      if (typeof reportData.alpha_details === 'object') {
        Object.entries(reportData.alpha_details).forEach(([alpha, details]) => {
          alphaDetails[alpha] = {
            alpha: parseFloat(alpha),
            coverage: details.coverage || null,
            intervalWidth: details.interval_width || null,
            sampleCount: details.sample_count || null,
            lowerBound: details.lower_bound || null,
            upperBound: details.upper_bound || null,
            distribution: details.distribution || null
          };
        });
      }
      
      return alphaDetails;
    }
    
    /**
     * Calculate expected calibration error (ECE) from calibration data
     * @param {Object} calibrationData - Calibration data
     * @return {number} Expected calibration error (0-1)
     */
    calculateCalibrationError(calibrationData) {
      if (!calibrationData || !Array.isArray(calibrationData.bins) || calibrationData.bins.length === 0) {
        return null;
      }
      
      // If error is already calculated in the data, return it
      if (calibrationData.error !== undefined && calibrationData.error !== null) {
        return calibrationData.error;
      }
      
      // Calculate ECE manually if not provided
      // ECE = sum(n_k / n * |p_k - f_k|) where:
      // n_k is the number of samples in bin k
      // n is the total number of samples
      // p_k is the predicted probability for bin k
      // f_k is the observed frequency in bin k
      
      const bins = calibrationData.bins;
      let totalSamples = 0;
      let errorSum = 0;
      
      // Calculate total number of samples
      bins.forEach(bin => {
        totalSamples += bin.sampleCount || 0;
      });
      
      if (totalSamples === 0) return null;
      
      // Calculate ECE
      bins.forEach(bin => {
        const binWeight = (bin.sampleCount || 0) / totalSamples;
        const binError = Math.abs((bin.predictedProbability || 0) - (bin.observedFrequency || 0));
        errorSum += binWeight * binError;
      });
      
      return errorSum;
    }
    
    /**
     * Calculate average coverage across alpha levels
     * @param {Object} coverageData - Coverage data
     * @return {number} Average coverage (0-1)
     */
    calculateAverageCoverage(coverageData) {
      if (!coverageData || !coverageData.alphaCoverage || 
          Object.keys(coverageData.alphaCoverage).length === 0) {
        return null;
      }
      
      // If average coverage is already calculated in the data, return it
      if (coverageData.averageCoverage !== undefined && coverageData.averageCoverage !== null) {
        return coverageData.averageCoverage;
      }
      
      // Calculate average coverage manually if not provided
      const alphaLevels = Object.keys(coverageData.alphaCoverage);
      let coverageSum = 0;
      
      alphaLevels.forEach(alpha => {
        coverageSum += coverageData.alphaCoverage[alpha] || 0;
      });
      
      return coverageSum / alphaLevels.length;
    }
    
    /**
     * Calculate average interval width
     * @param {Object} intervalWidthData - Interval width data
     * @return {number} Average interval width
     */
    calculateAverageIntervalWidth(intervalWidthData) {
      if (!intervalWidthData || !intervalWidthData.alphaWidths || 
          Object.keys(intervalWidthData.alphaWidths).length === 0) {
        return null;
      }
      
      // If average width is already calculated in the data, return it
      if (intervalWidthData.averageWidth !== undefined && intervalWidthData.averageWidth !== null) {
        return intervalWidthData.averageWidth;
      }
      
      // Calculate average width manually if not provided
      const alphaLevels = Object.keys(intervalWidthData.alphaWidths);
      let widthSum = 0;
      
      alphaLevels.forEach(alpha => {
        widthSum += intervalWidthData.alphaWidths[alpha] || 0;
      });
      
      return widthSum / alphaLevels.length;
    }
    
    /**
     * Calculate overall uncertainty score
     * @param {number} calibrationError - Calibration error (0-1)
     * @param {number} averageCoverage - Average coverage (0-1)
     * @param {number} averageIntervalWidth - Average interval width
     * @return {number} Uncertainty score (0-1)
     */
    calculateUncertaintyScore(calibrationError, averageCoverage, averageIntervalWidth) {
      if (calibrationError === null || averageCoverage === null) {
        return null;
      }
      
      // Convert calibration error to calibration accuracy (1 - error)
      const calibrationAccuracy = 1 - (calibrationError || 0);
      
      // Balance between calibration and coverage
      // A good uncertainty model should have high coverage and high calibration
      const rawScore = (calibrationAccuracy * 0.6) + (averageCoverage * 0.4);
      
      // Adjust for interval width if available (penalize excessively wide intervals)
      // An ideal model has high coverage with narrow intervals
      let finalScore = rawScore;
      
      if (averageIntervalWidth !== null && averageIntervalWidth > 0) {
        // Normalize interval width to 0-1 range assuming reasonable bounds
        // This is a simplified approach - in practice, the normalization would
        // depend on the specific task and scale of the target variable
        const normalizedWidth = Math.min(1, averageIntervalWidth / 2);
        
        // Apply a gentle penalty for wider intervals
        finalScore = rawScore * (1 - normalizedWidth * 0.2);
      }
      
      return Math.max(0, Math.min(1, finalScore));
    }
    
    /**
     * Calculate calibration curve data for visualization
     * @param {Object} calibrationData - Calibration data
     * @return {Object} Processed calibration curve data
     */
    processCalibrationCurveData(calibrationData) {
      if (!calibrationData || !Array.isArray(calibrationData.bins) || calibrationData.bins.length === 0) {
        return {
          predictedProbabilities: [],
          observedFrequencies: [],
          sampleCounts: []
        };
      }
      
      // Extract data for visualization
      const predictedProbabilities = [];
      const observedFrequencies = [];
      const sampleCounts = [];
      
      calibrationData.bins.forEach(bin => {
        predictedProbabilities.push(bin.predictedProbability || 0);
        observedFrequencies.push(bin.observedFrequency || 0);
        sampleCounts.push(bin.sampleCount || 0);
      });
      
      return {
        predictedProbabilities,
        observedFrequencies,
        sampleCounts
      };
    }
    
    /**
     * Process coverage data for visualization
     * @param {Object} coverageData - Coverage data
     * @return {Object} Processed coverage data
     */
    processCoverageData(coverageData) {
      if (!coverageData || !coverageData.alphaCoverage || 
          Object.keys(coverageData.alphaCoverage).length === 0) {
        return {
          alphaLevels: [],
          coverage: [],
          targetCoverage: []
        };
      }
      
      // Extract data for visualization
      const alphaLevels = [];
      const coverage = [];
      const targetCoverage = [];
      
      Object.entries(coverageData.alphaCoverage)
        .sort(([a], [b]) => parseFloat(a) - parseFloat(b))
        .forEach(([alpha, cov]) => {
          const alphaValue = parseFloat(alpha);
          alphaLevels.push(alphaValue);
          coverage.push(cov);
          targetCoverage.push(1 - alphaValue); // Expected coverage = 1 - alpha
        });
      
      return {
        alphaLevels,
        coverage,
        targetCoverage
      };
    }
    
    /**
     * Process interval width data for visualization
     * @param {Object} intervalWidthData - Interval width data
     * @return {Object} Processed interval width data
     */
    processIntervalWidthData(intervalWidthData) {
      if (!intervalWidthData || !intervalWidthData.alphaWidths || 
          Object.keys(intervalWidthData.alphaWidths).length === 0) {
        return {
          alphaLevels: [],
          widths: []
        };
      }
      
      // Extract data for visualization
      const alphaLevels = [];
      const widths = [];
      
      Object.entries(intervalWidthData.alphaWidths)
        .sort(([a], [b]) => parseFloat(a) - parseFloat(b))
        .forEach(([alpha, width]) => {
          alphaLevels.push(parseFloat(alpha));
          widths.push(width);
        });
      
      return {
        alphaLevels,
        widths
      };
    }
  }
  
  export default UncertaintyData;