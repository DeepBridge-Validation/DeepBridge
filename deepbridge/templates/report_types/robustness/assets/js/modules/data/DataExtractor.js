/**
 * Data Extractor
 * 
 * Utility class to extract and normalize data from robustness report structure
 * for use in charts, tables, and visualizations.
 */
import PerturbationData from './PerturbationData.js';
import FeatureData from './FeatureData.js';

class DataExtractor {
  /**
   * Initialize the data extractor
   */
  constructor() {
    // Initialize specialized data handlers
    this.perturbationHandler = new PerturbationData();
    this.featureHandler = new FeatureData();
    
    // Default values for fallbacks
    this.defaultLevels = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
  }
  
  /**
   * Extract perturbation data from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted and normalized perturbation data, or null if not available
   */
  getPerturbationData(reportData) {
    if (!reportData) return null;
    
    return this.perturbationHandler.extract(reportData);
  }
  
  /**
   * Extract feature importance data from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted and normalized feature importance data, or null if not available
   */
  getFeatureImportanceData(reportData) {
    if (!reportData) return null;
    
    return this.featureHandler.extract(reportData);
  }
  
  /**
   * Extract model comparison data from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted and normalized model comparison data, or null if not available
   */
  getModelComparisonData(reportData) {
    if (!reportData) return null;
    
    // Get perturbation data first
    const perturbationData = this.getPerturbationData(reportData);
    if (!perturbationData || !perturbationData.alternativeModels) {
      return null;
    }
    
    // Format data for model comparison
    return {
      primaryModel: {
        name: perturbationData.modelName,
        baseScore: perturbationData.baseScore,
        levels: perturbationData.levels,
        scores: perturbationData.scores,
        worstScores: perturbationData.worstScores,
        metric: perturbationData.metric
      },
      alternativeModels: Object.entries(perturbationData.alternativeModels).map(([name, model]) => ({
        name: name,
        baseScore: model.baseScore,
        levels: perturbationData.levels,
        scores: model.scores,
        worstScores: model.worstScores
      }))
    };
  }
  
  /**
   * Get robustness score from report data
   * @param {Object} reportData - The full report data object
   * @return {number} Robustness score (0-1), or null if not available
   */
  getRobustnessScore(reportData) {
    if (!reportData) return null;
    
    // Check if robustness score is directly in reportData
    if (reportData.robustness_score !== undefined) {
      return reportData.robustness_score;
    }
    
    // Check if robustness score is in primary_model
    if (reportData.primary_model && reportData.primary_model.robustness_score !== undefined) {
      return reportData.primary_model.robustness_score;
    }
    
    // Try to calculate from perturbation data
    const perturbationData = this.getPerturbationData(reportData);
    if (perturbationData) {
      return perturbationData.robustnessScore;
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
    
    // Get perturbation data first
    const perturbationData = this.getPerturbationData(reportData);
    
    // Base metrics
    const metrics = {
      robustnessScore: this.getRobustnessScore(reportData) || 
                      (perturbationData ? perturbationData.robustnessScore : null),
      baseScore: reportData.base_score || 
                (reportData.primary_model ? reportData.primary_model.base_score : null) ||
                (perturbationData ? perturbationData.baseScore : null),
      modelName: reportData.model_name || 
                (reportData.primary_model ? reportData.primary_model.model_name : null) ||
                (perturbationData ? perturbationData.modelName : 'Unknown Model'),
      metric: reportData.metric || 
              (reportData.primary_model ? reportData.primary_model.metric : null) ||
              (perturbationData ? perturbationData.metric : 'Score')
    };
    
    // Add raw and quantile impact if available
    metrics.rawImpact = this.calculateRawImpact(reportData, perturbationData);
    metrics.quantileImpact = this.calculateQuantileImpact(reportData, perturbationData);
    
    return metrics;
  }
  
  /**
   * Calculate raw impact from report data
   * @param {Object} reportData - The full report data object
   * @param {Object} perturbationData - Pre-extracted perturbation data (optional)
   * @return {number} Raw impact value (0-1)
   */
  calculateRawImpact(reportData, perturbationData = null) {
    // Use explicit raw impact if available
    if (reportData.raw_impact !== undefined) {
      return reportData.raw_impact;
    } else if (reportData.primary_model && reportData.primary_model.raw_impact !== undefined) {
      return reportData.primary_model.raw_impact;
    } 
    
    // Calculate from perturbation data if available
    const pData = perturbationData || this.getPerturbationData(reportData);
    if (pData && pData.scores && pData.scores.length > 0) {
      // Estimate raw impact as average percentage decrease in score
      const avgScore = pData.scores.reduce((sum, score) => sum + score, 0) / pData.scores.length;
      return (pData.baseScore - avgScore) / pData.baseScore;
    }
    
    return 0;
  }
  
  /**
   * Calculate quantile impact from report data
   * @param {Object} reportData - The full report data object
   * @param {Object} perturbationData - Pre-extracted perturbation data (optional)
   * @return {number} Quantile impact value (0-1)
   */
  calculateQuantileImpact(reportData, perturbationData = null) {
    // Use explicit quantile impact if available
    if (reportData.quantile_impact !== undefined) {
      return reportData.quantile_impact;
    } else if (reportData.primary_model && reportData.primary_model.quantile_impact !== undefined) {
      return reportData.primary_model.quantile_impact;
    } 
    
    // If not explicitly available, simulate using raw impact
    return this.calculateRawImpact(reportData, perturbationData) * 0.9;
  }
  
  /**
   * Get dataset metadata
   * @param {Object} reportData - The full report data object
   * @return {Object} Dataset metadata
   */
  getDatasetMetadata(reportData) {
    if (!reportData) return {};
    
    const metadata = {};
    
    // Extract dataset information if available
    if (reportData.dataset) {
      metadata.name = reportData.dataset.name;
      metadata.size = reportData.dataset.size;
      metadata.features = reportData.dataset.features;
      metadata.target = reportData.dataset.target;
    } else if (reportData.primary_model && reportData.primary_model.dataset) {
      metadata.name = reportData.primary_model.dataset.name;
      metadata.size = reportData.primary_model.dataset.size;
      metadata.features = reportData.primary_model.dataset.features;
      metadata.target = reportData.primary_model.dataset.target;
    }
    
    return metadata;
  }
  
  /**
   * Get model metadata
   * @param {Object} reportData - The full report data object
   * @return {Object} Model metadata
   */
  getModelMetadata(reportData) {
    if (!reportData) return {};
    
    const metadata = {};
    
    // Extract model information
    if (reportData.model_type) {
      metadata.type = reportData.model_type;
      metadata.name = reportData.model_name;
      metadata.metric = reportData.metric;
    } else if (reportData.primary_model) {
      metadata.type = reportData.primary_model.model_type;
      metadata.name = reportData.primary_model.model_name;
      metadata.metric = reportData.primary_model.metric;
    }
    
    // Extract other model metadata if available
    if (reportData.model_parameters) {
      metadata.parameters = reportData.model_parameters;
    } else if (reportData.primary_model && reportData.primary_model.model_parameters) {
      metadata.parameters = reportData.primary_model.model_parameters;
    }
    
    return metadata;
  }
}

export default DataExtractor;