/**
 * Data Model
 * 
 * Combines data handling components to provide a unified interface for
 * accessing and manipulating robustness report data.
 */
import DataExtractor from './DataExtractor.js';
import PerturbationData from './PerturbationData.js';
import FeatureData from './FeatureData.js';
import StatsAnalyzer from './StatsAnalyzer.js';

class DataModel {
  /**
   * Initialize the data model
   * @param {Object} reportData - The full report data object
   */
  constructor(reportData = null) {
    // Initialize components
    this.dataExtractor = new DataExtractor();
    this.perturbationHandler = new PerturbationData();
    this.featureHandler = new FeatureData();
    this.statsAnalyzer = new StatsAnalyzer();
    
    // Store report data
    this.reportData = reportData;
    
    // Cache for extracted data
    this.cache = {
      perturbationData: null,
      featureData: null,
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
      perturbationData: null,
      featureData: null,
      summaryMetrics: null,
      modelComparisonData: null
    };
  }
  
  /**
   * Get perturbation data
   * @return {Object} Perturbation data
   */
  getPerturbationData() {
    if (!this.reportData) return null;
    
    // Use cached data if available
    if (this.cache.perturbationData) {
      return this.cache.perturbationData;
    }
    
    // Extract perturbation data
    const perturbationData = this.dataExtractor.getPerturbationData(this.reportData);
    
    // Cache and return
    this.cache.perturbationData = perturbationData;
    return perturbationData;
  }
  
  /**
   * Get feature importance data
   * @return {Object} Feature importance data
   */
  getFeatureData() {
    if (!this.reportData) return null;
    
    // Use cached data if available
    if (this.cache.featureData) {
      return this.cache.featureData;
    }
    
    // Extract feature data
    const featureData = this.dataExtractor.getFeatureImportanceData(this.reportData);
    
    // Cache and return
    this.cache.featureData = featureData;
    return featureData;
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
   * Get feature importance rankings
   * @return {Object} Feature importance rankings for model and robustness
   */
  getFeatureRankings() {
    const featureData = this.getFeatureData();
    if (!featureData) return null;
    
    // Prepare rankings
    const rankings = {
      modelRanking: [],
      robustnessRanking: [],
      combinedRanking: []
    };
    
    // Create model ranking
    if (featureData.modelFeatureImportance) {
      rankings.modelRanking = Object.entries(featureData.modelFeatureImportance)
        .map(([feature, importance]) => ({
          feature,
          importance,
          inSubset: featureData.featureSubset.includes(feature)
        }))
        .sort((a, b) => b.importance - a.importance);
    }
    
    // Create robustness ranking
    if (featureData.featureImportance) {
      rankings.robustnessRanking = Object.entries(featureData.featureImportance)
        .map(([feature, importance]) => ({
          feature,
          importance,
          inSubset: featureData.featureSubset.includes(feature)
        }))
        .sort((a, b) => b.importance - a.importance);
    }
    
    // Create combined ranking
    if (featureData.modelFeatureImportance || featureData.featureImportance) {
      const combinedImportance = {};
      
      // Add model importance
      Object.entries(featureData.modelFeatureImportance || {}).forEach(([feature, importance]) => {
        combinedImportance[feature] = {
          feature,
          modelImportance: importance,
          robustnessImportance: 0,
          combinedScore: importance * 0.5, // 50% weight
          inSubset: featureData.featureSubset.includes(feature)
        };
      });
      
      // Add robustness importance
      Object.entries(featureData.featureImportance || {}).forEach(([feature, importance]) => {
        if (combinedImportance[feature]) {
          combinedImportance[feature].robustnessImportance = importance;
          combinedImportance[feature].combinedScore += importance * 0.5; // 50% weight
        } else {
          combinedImportance[feature] = {
            feature,
            modelImportance: 0,
            robustnessImportance: importance,
            combinedScore: importance * 0.5, // 50% weight
            inSubset: featureData.featureSubset.includes(feature)
          };
        }
      });
      
      // Sort by combined score
      rankings.combinedRanking = Object.values(combinedImportance)
        .sort((a, b) => b.combinedScore - a.combinedScore);
    }
    
    return rankings;
  }
  
  /**
   * Get feature correlation analysis
   * @param {number} topFeatureCount - Number of top features to include
   * @return {Object} Feature correlation analysis
   */
  getFeatureCorrelations(topFeatureCount = 10) {
    const featureData = this.getFeatureData();
    if (!featureData) return null;
    
    return this.featureHandler.calculateFeatureCorrelations(featureData, topFeatureCount);
  }
  
  /**
   * Get feature contribution analysis
   * @return {Object} Feature contribution analysis
   */
  getFeatureContributions() {
    const featureData = this.getFeatureData();
    if (!featureData) return null;
    
    return this.featureHandler.calculateFeatureContributions(featureData);
  }
  
  /**
   * Get performance distribution data
   * @return {Object} Performance distribution data
   */
  getDistributionData() {
    const perturbationData = this.getPerturbationData();
    if (!perturbationData) return null;
    
    return this.perturbationHandler.generateDistributionData(perturbationData);
  }
  
  /**
   * Get comparative metrics for model comparison
   * @return {Object} Comparative metrics
   */
  getComparativeMetrics() {
    const modelComparisonData = this.getModelComparisonData();
    if (!modelComparisonData) return null;
    
    const models = [
      modelComparisonData.primaryModel,
      ...modelComparisonData.alternativeModels
    ];
    
    return this.statsAnalyzer.calculateComparativeMetrics(models);
  }
  
  /**
   * Get level-wise performance metrics
   * @return {Array} Level metrics
   */
  getLevelMetrics() {
    const perturbationData = this.getPerturbationData();
    if (!perturbationData) return null;
    
    return this.statsAnalyzer.calculateLevelMetrics(perturbationData);
  }
}

export default DataModel;