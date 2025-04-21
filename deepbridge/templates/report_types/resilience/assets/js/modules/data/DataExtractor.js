// DataExtractor.js - placeholder
/**
 * Data Extractor
 * 
 * Utility class to extract and normalize data from resilience report structure
 * for use in charts, tables, and visualizations.
 */
import ResilienceData from './ResilienceData.js';
import DistributionData from './DistributionData.js';
import ShiftAnalyzer from './ShiftAnalyzer.js';

class DataExtractor {
  /**
   * Initialize the data extractor
   */
  constructor() {
    // Initialize specialized data handlers
    this.resilienceHandler = new ResilienceData();
    this.distributionHandler = new DistributionData();
    this.shiftAnalyzer = new ShiftAnalyzer();
    
    // Default values for fallbacks
    this.defaultShiftTypes = ["covariate", "concept", "prediction"];
  }
  
  /**
   * Extract resilience data from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted and normalized resilience data, or null if not available
   */
  getResilienceData(reportData) {
    if (!reportData) return null;
    
    return this.resilienceHandler.extract(reportData);
  }
  
  /**
   * Extract distribution data from the report structure
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Extracted and normalized distribution data, or null if not available
   */
  getDistributionData(reportData) {
    if (!reportData) return null;
    
    return this.distributionHandler.extract(reportData);
  }
  
  /**
   * Get individual shift details by type
   * @param {Object} reportData - The full report data object
   * @param {string} shiftType - Type of shift to extract
   * @return {Object|null} Shift details or null if not available
   */
  getShiftDetails(reportData, shiftType) {
    if (!reportData) return null;
    
    const resilienceData = this.getResilienceData(reportData);
    if (!resilienceData || !resilienceData.shifts) return null;
    
    // Find the shift with the specified type
    const shift = resilienceData.shifts.find(s => s.type === shiftType);
    if (!shift) return null;
    
    // Get detailed distribution data for this shift if available
    const distributionData = this.getDistributionData(reportData);
    if (distributionData && distributionData.shifts) {
      const distributionShift = distributionData.shifts.find(s => s.type === shiftType);
      if (distributionShift) {
        // Merge resilience shift data with distribution shift data
        return {
          ...shift,
          distributions: distributionShift.distributions,
          metrics: distributionShift.metrics
        };
      }
    }
    
    return shift;
  }
  
  /**
   * Get feature impact data for resilience
   * @param {Object} reportData - The full report data object
   * @return {Object|null} Feature impact data or null if not available
   */
  getFeatureImpactData(reportData) {
    if (!reportData) return null;
    
    // Extract resilience data first
    const resilienceData = this.getResilienceData(reportData);
    if (!resilienceData) return null;
    
    // Return feature impact data if available
    return resilienceData.featureImpact || null;
  }
  
  /**
   * Get resilience score
   * @param {Object} reportData - The full report data object
   * @return {number} Resilience score (0-1), or null if not available
   */
  getResilienceScore(reportData) {
    if (!reportData) return null;
    
    // Check if resilience score is directly in reportData
    if (reportData.resilience_score !== undefined) {
      return reportData.resilience_score;
    }
    
    // Check if resilience score is in primary_model
    if (reportData.primary_model && reportData.primary_model.resilience_score !== undefined) {
      return reportData.primary_model.resilience_score;
    }
    
    // Try to calculate from resilience data
    const resilienceData = this.getResilienceData(reportData);
    if (resilienceData) {
      return resilienceData.resilienceScore;
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
    
    // Get resilience data first
    const resilienceData = this.getResilienceData(reportData);
    
    // Base metrics
    const metrics = {
      resilienceScore: this.getResilienceScore(reportData) || 
                      (resilienceData ? resilienceData.resilienceScore : null),
      baseScore: reportData.base_score || 
                (reportData.primary_model ? reportData.primary_model.base_score : null) ||
                (resilienceData ? resilienceData.baseScore : null),
      modelName: reportData.model_name || 
                (reportData.primary_model ? reportData.primary_model.model_name : null) ||
                (resilienceData ? resilienceData.modelName : 'Unknown Model'),
      metric: reportData.metric || 
              (reportData.primary_model ? reportData.primary_model.metric : null) ||
              (resilienceData ? resilienceData.metric : 'Score')
    };
    
    // Add average and worst shift impact if available
    metrics.avgShiftImpact = this.calculateAverageShiftImpact(reportData, resilienceData);
    metrics.worstShiftImpact = this.calculateWorstShiftImpact(reportData, resilienceData);
    metrics.worstShiftType = this.getWorstShiftType(reportData, resilienceData);
    
    return metrics;
  }
  
  /**
   * Calculate average shift impact
   * @param {Object} reportData - The full report data object
   * @param {Object} resilienceData - Pre-extracted resilience data (optional)
   * @return {number} Average shift impact (0-1)
   */
  calculateAverageShiftImpact(reportData, resilienceData = null) {
    // Use explicit average impact if available
    if (reportData.avg_shift_impact !== undefined) {
      return reportData.avg_shift_impact;
    } else if (reportData.primary_model && reportData.primary_model.avg_shift_impact !== undefined) {
      return reportData.primary_model.avg_shift_impact;
    } 
    
    // Calculate from resilience data if available
    const rData = resilienceData || this.getResilienceData(reportData);
    if (rData && rData.shifts && rData.shifts.length > 0) {
      const baseScore = rData.baseScore || 0;
      if (baseScore === 0) return 0;
      
      // Calculate average impact as percentage decrease from base score
      const totalImpact = rData.shifts.reduce((sum, shift) => {
        return sum + ((baseScore - shift.score) / baseScore);
      }, 0);
      
      return totalImpact / rData.shifts.length;
    }
    
    return 0;
  }
  
  /**
   * Calculate worst shift impact
   * @param {Object} reportData - The full report data object
   * @param {Object} resilienceData - Pre-extracted resilience data (optional)
   * @return {number} Worst shift impact (0-1)
   */
  calculateWorstShiftImpact(reportData, resilienceData = null) {
    // Use explicit worst impact if available
    if (reportData.worst_shift_impact !== undefined) {
      return reportData.worst_shift_impact;
    } else if (reportData.primary_model && reportData.primary_model.worst_shift_impact !== undefined) {
      return reportData.primary_model.worst_shift_impact;
    } 
    
    // Calculate from resilience data if available
    const rData = resilienceData || this.getResilienceData(reportData);
    if (rData && rData.shifts && rData.shifts.length > 0) {
      const baseScore = rData.baseScore || 0;
      if (baseScore === 0) return 0;
      
      // Calculate worst impact as maximum percentage decrease from base score
      const impacts = rData.shifts.map(shift => {
        return (baseScore - shift.score) / baseScore;
      });
      
      return Math.max(...impacts);
    }
    
    return 0;
  }
  
  /**
   * Get worst shift type
   * @param {Object} reportData - The full report data object
   * @param {Object} resilienceData - Pre-extracted resilience data (optional)
   * @return {string} Worst shift type
   */
  getWorstShiftType(reportData, resilienceData = null) {
    // Use explicit worst shift type if available
    if (reportData.worst_shift_type !== undefined) {
      return reportData.worst_shift_type;
    } else if (reportData.primary_model && reportData.primary_model.worst_shift_type !== undefined) {
      return reportData.primary_model.worst_shift_type;
    } 
    
    // Calculate from resilience data if available
    const rData = resilienceData || this.getResilienceData(reportData);
    if (rData && rData.shifts && rData.shifts.length > 0) {
      const baseScore = rData.baseScore || 0;
      if (baseScore === 0) return 'unknown';
      
      // Find shift with maximum impact
      let worstShift = rData.shifts[0];
      let worstImpact = (baseScore - worstShift.score) / baseScore;
      
      rData.shifts.forEach(shift => {
        const impact = (baseScore - shift.score) / baseScore;
        if (impact > worstImpact) {
          worstImpact = impact;
          worstShift = shift;
        }
      });
      
      return worstShift.type;
    }
    
    return 'unknown';
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