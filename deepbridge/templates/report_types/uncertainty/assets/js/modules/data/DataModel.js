// DataModel.js - placeholder
/**
 * Data Model
 * 
 * Combines data handling components to provide a unified interface for
 * accessing and manipulating uncertainty report data.
 */
import DataExtractor from './DataExtractor.js';
import UncertaintyData from './UncertaintyData.js';

class DataModel {
  /**
   * Initialize the data model
   * @param {Object} reportData - The full report data object
   */
  constructor(reportData = null) {
    // Initialize components
    this.dataExtractor = new DataExtractor();
    this.uncertaintyHandler = new UncertaintyData();
    
    // Store report data
    this.reportData = reportData;
    
    // Cache for extracted data
    this.cache = {
      uncertaintyData: null,
      calibrationData: null,
      coverageData: null,
      intervalWidthData: null,
      summaryMetrics: null,
      modelComparisonData: null
    };
  }
  
  /**
   * Set new report data
   * @param {Object} reportData - The full report data object
   */
  setReportData(reportData) {
    this.reportData = reportData;
    
    // Clear cache when report data changes
    this.clearCache();
  }
  
  /**
   * Clear cached data
   */
  clearCache() {
    this.cache = {
      uncertaintyData: null,
      calibrationData: null,
      coverageData: null,
      intervalWidthData: null,
      summaryMetrics: null,
      modelComparisonData: null
    };
  }
  
  /**
   * Get uncertainty data
   * @return {Object} Uncertainty data
   */
  getUncertaintyData() {
    if (!this.reportData) return null;
    
    // Use cached data if available
    if (this.cache.uncertaintyData) {
      return this.cache.uncertaintyData;
    }
    
    // Extract uncertainty data
    const uncertaintyData = this.dataExtractor.getUncertaintyData(this.reportData);
    
    // Cache and return
    this.cache.uncertaintyData = uncertaintyData;
    return uncertaintyData;
  }
  
  /**
   * Get calibration data
   * @return {Object} Calibration data
   */
  getCalibrationData() {
    if (!this.reportData) return null;
    
    // Use cached data if available
    if (this.cache.calibrationData) {
      return this.cache.calibrationData;
    }
    
    // Extract calibration data
    const calibrationData = this.dataExtractor.getCalibrationData(this.reportData);
    
    // Cache and return
    this.cache.calibrationData = calibrationData;
    return calibrationData;
  }
  
  /**
   * Get coverage data
   * @return {Object} Coverage data
   */
  getCoverageData() {
    if (!this.reportData) return null;
    
    // Use cached data if available
    if (this.cache.coverageData) {
      return this.cache.coverageData;
    }
    
    // Extract coverage data
    const coverageData = this.dataExtractor.getCoverageData(this.reportData);
    
    // Cache and return
    this.cache.coverageData = coverageData;
    return coverageData;
  }
  
  /**
   * Get interval width data
   * @return {Object} Interval width data
   */
  getIntervalWidthData() {
    if (!this.reportData) return null;
    
    // Use cached data if available
    if (this.cache.intervalWidthData) {
      return this.cache.intervalWidthData;
    }
    
    // Extract interval width data
    const intervalWidthData = this.dataExtractor.getIntervalWidthData(this.reportData);
    
    // Cache and return
    this.cache.intervalWidthData = intervalWidthData;
    return intervalWidthData;
  }
  
  /**
   * Get summary metrics
   * @return {Object} Summary metrics
   */
  getSummaryMetrics() {
    if (!this.reportData) return null;
    
    // Use cached data if available
    if (this.cache.summaryMetrics) {
      return this.cache.summaryMetrics;
    }
    
    // Extract summary metrics
    const summaryMetrics = this.dataExtractor.getSummaryMetrics(this.reportData);
    
    // Cache and return
    this.cache.summaryMetrics = summaryMetrics;
    return summaryMetrics;
  }
  
  /**
   * Get model comparison data
   * @return {Object} Model comparison data
   */
  getModelComparisonData() {
    if (!this.reportData) return null;
    
    // Use cached data if available
    if (this.cache.modelComparisonData) {
      return this.cache.modelComparisonData;
    }
    
    // Extract model comparison data
    const modelComparisonData = this.dataExtractor.getModelComparisonData(this.reportData);
    
    // Cache and return
    this.cache.modelComparisonData = modelComparisonData;
    return modelComparisonData;
  }
  
  /**
   * Get alpha level details
   * @param {number} alpha - Alpha level to get details for
   * @return {Object} Alpha level details
   */
  getAlphaLevelDetails(alpha) {
    if (!this.reportData) return null;
    
    // Alpha level details are not cached as they might be requested for different alphas
    return this.dataExtractor.getAlphaLevelDetails(this.reportData, alpha);
  }
  
  /**
   * Get dataset information
   * @return {Object} Dataset information
   */
  getDatasetInfo() {
    if (!this.reportData) return null;
    
    return this.dataExtractor.getDatasetInfo(this.reportData);
  }
  
  /**
   * Get model metadata
   * @return {Object} Model metadata
   */
  getModelMetadata() {
    if (!this.reportData) return null;
    
    return this.dataExtractor.getModelMetadata(this.reportData);
  }
  
  /**
   * Calculate expected calibration error (ECE) from calibration data
   * @return {number} Expected calibration error
   */
  calculateCalibrationError() {
    const calibrationData = this.getCalibrationData();
    if (!calibrationData) return null;
    
    return this.uncertaintyHandler.calculateCalibrationError(calibrationData);
  }
  
  /**
   * Calculate average coverage across alpha levels
   * @return {number} Average coverage
   */
  calculateAverageCoverage() {
    const coverageData = this.getCoverageData();
    if (!coverageData) return null;
    
    return this.uncertaintyHandler.calculateAverageCoverage(coverageData);
  }
  
  /**
   * Calculate average interval width
   * @return {number} Average interval width
   */
  calculateAverageIntervalWidth() {
    const intervalWidthData = this.getIntervalWidthData();
    if (!intervalWidthData) return null;
    
    return this.uncertaintyHandler.calculateAverageIntervalWidth(intervalWidthData);
  }
}

export default DataModel;