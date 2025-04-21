// DataExtractor.js - placeholder
/**
 * Data Extractor
 * 
 * Utility class to extract and normalize data from uncertainty report structure
 * for use in charts, tables, and visualizations.
 */
import UncertaintyData from './UncertaintyData.js';

class DataExtractor {
  /**
   * Initialize the data extractor
   */
  constructor() {
    // Initialize specialized data handlers
    this.uncertaintyHandler = new UncertaintyData();
    
    // Default alpha levels for confidence intervals
    this.defaultAlphaLevels = [0.01, 0.05, 0.1, 0.2, 0.5];
  }
  
  /**
   * Extract uncertainty data from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted and normalized uncertainty data, or null if not available
   */
  getUncertaintyData(reportData) {
    if (!reportData) return null;
    
    return this.uncertaintyHandler.extract(reportData);
  }
  
  /**
   * Extract calibration data from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted calibration data, or null if not available
   */
  getCalibrationData(reportData) {
    if (!reportData) return null;
    
    // Get uncertainty data first
    const uncertaintyData = this.getUncertaintyData(reportData);
    if (!uncertaintyData || !uncertaintyData.calibration) {
      return null;
    }
    
    return uncertaintyData.calibration;
  }
  
  /**
   * Extract coverage data from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted coverage data, or null if not available
   */
  getCoverageData(reportData) {
    if (!reportData) return null;
    
    // Get uncertainty data first
    const uncertaintyData = this.getUncertaintyData(reportData);
    if (!uncertaintyData || !uncertaintyData.coverage) {
      return null;
    }
    
    return uncertaintyData.coverage;
  }
  
  /**
   * Extract interval width data from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted interval width data, or null if not available
   */
  getIntervalWidthData(reportData) {
    if (!reportData) return null;
    
    // Get uncertainty data first
    const uncertaintyData = this.getUncertaintyData(reportData);
    if (!uncertaintyData || !uncertaintyData.intervalWidths) {
      return null;
    }
    
    return uncertaintyData.intervalWidths;
  }
  
  /**
   * Extract alpha level details from the report structure
   * @param {Object} reportData - The full report data object
   * @param {number} alpha - The alpha level to extract details for
   * @return {Object|null} Extracted alpha level details, or null if not available
   */
  getAlphaLevelDetails(reportData, alpha) {
    if (!reportData) return null;
    
    // Get uncertainty data first
    const uncertaintyData = this.getUncertaintyData(reportData);
    if (!uncertaintyData || !uncertaintyData.alphaDetails) {
      return null;
    }
    
    // Find the matching alpha level
    const alphaStr = alpha.toString();
    return uncertaintyData.alphaDetails[alphaStr] || null;
  }
  
  /**
   * Get model metadata from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object} Model metadata
   */
  getModelMetadata(reportData) {
    if (!reportData) return {};
    
    const metadata = {
      name: reportData.model_name || 'Unknown Model',
      type: reportData.model_type || 'Unknown Type',
      metric: reportData.metric || 'Score',
      uncertaintyMethod: reportData.uncertainty_method || 'Unknown Method',
      calibrated: !!reportData.is_calibrated,
      score: reportData.uncertainty_score || null
    };
    
    return metadata;
  }
  
  /**
   * Get uncertainty score from report data
   * @param {Object} reportData - The full report data object
   * @return {number} Uncertainty score (0-1), or null if not available
   */
  getUncertaintyScore(reportData) {
    if (!reportData) return null;
    
    // Check if uncertainty score is directly in reportData
    if (reportData.uncertainty_score !== undefined) {
      return reportData.uncertainty_score;
    }
    
    // If not directly available, calculate from calibration and coverage data
    const uncertaintyData = this.getUncertaintyData(reportData);
    if (uncertaintyData) {
      return uncertaintyData.uncertaintyScore || null;
    }
    
    return null;
  }
  
  /**
   * Get summary metrics for the report
   * @param {Object} reportData - The full report data object
   * @return {Object} Extracted summary metrics
   */
  getSummaryMetrics(reportData) {
    if (!reportData) return {};
    
    // Get uncertainty data first
    const uncertaintyData = this.getUncertaintyData(reportData);
    
    // Base metrics
    const metrics = {
      uncertaintyScore: this.getUncertaintyScore(reportData),
      averageCoverage: uncertaintyData ? uncertaintyData.averageCoverage : null,
      calibrationError: uncertaintyData ? uncertaintyData.calibrationError : null,
      averageIntervalWidth: uncertaintyData ? uncertaintyData.averageIntervalWidth : null,
      modelName: reportData.model_name || 'Unknown Model',
      metric: reportData.metric || 'Score',
      uncertaintyMethod: reportData.uncertainty_method || 'Unknown Method'
    };
    
    return metrics;
  }
  
  /**
   * Extract model comparison data from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted model comparison data, or null if not available
   */
  getModelComparisonData(reportData) {
    if (!reportData || !reportData.model_comparison) {
      return null;
    }
    
    const comparisonData = {
      primaryModel: {
        name: reportData.model_name || 'Primary Model',
        score: this.getUncertaintyScore(reportData),
        coverage: reportData.average_coverage || null,
        calibrationError: reportData.calibration_error || null,
        intervalWidth: reportData.average_interval_width || null
      },
      alternativeModels: []
    };
    
    // Process each alternative model
    if (reportData.model_comparison && Array.isArray(reportData.model_comparison)) {
      reportData.model_comparison.forEach(model => {
        comparisonData.alternativeModels.push({
          name: model.model_name || 'Alternative Model',
          score: model.uncertainty_score || null,
          coverage: model.average_coverage || null,
          calibrationError: model.calibration_error || null,
          intervalWidth: model.average_interval_width || null
        });
      });
    }
    
    return comparisonData;
  }
  
  /**
   * Extract dataset information from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object} Dataset information
   */
  getDatasetInfo(reportData) {
    if (!reportData) return {};
    
    return {
      name: reportData.dataset_name || 'Unknown Dataset',
      size: reportData.dataset_size || null,
      testSize: reportData.test_size || null,
      testCount: reportData.test_count || null
    };
  }
}

export default DataExtractor;