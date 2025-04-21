// ResilienceData.js - placeholder
/**
 * Resilience Data Handler
 * 
 * Specialized handler for extracting and processing resilience data
 * from the resilience report structure.
 */

class ResilienceData {
    /**
     * Initialize the resilience data handler
     */
    constructor() {
      // Default shift types for fallback
      this.defaultShiftTypes = ["covariate", "concept", "prediction"];
    }
    
    /**
     * Extract resilience data from report data
     * @param {Object} reportData - The full report data object
     * @return {Object|null} Extracted resilience data or null if not available
     */
    extract(reportData) {
      if (!reportData) return null;
      
      let resilienceData = null;
      
      // Check if resilience data is directly in reportData
      if (reportData.shifts) {
        resilienceData = this.extractResilienceData(reportData);
      } 
      // Check if resilience data is in primary_model
      else if (reportData.primary_model && reportData.primary_model.shifts) {
        resilienceData = this.extractResilienceData(reportData.primary_model);
        
        // Add alternative models if available
        if (reportData.alternative_models) {
          resilienceData.alternativeModels = {};
          
          Object.entries(reportData.alternative_models).forEach(([name, modelData]) => {
            if (modelData && modelData.shifts) {
              resilienceData.alternativeModels[name] = this.extractResilienceDataForAlternativeModel(
                modelData, 
                resilienceData.shifts.map(shift => shift.type)
              );
            }
          });
        }
      }
      // No resilience data found, try to create from base data if available
      else if (reportData.base_score) {
        resilienceData = this.generateDefaultResilienceData(reportData);
      }
      
      return resilienceData;
    }
    
    /**
     * Extract resilience data from model data
     * @param {Object} modelData - Model data containing resilience results
     * @return {Object} Extracted and normalized resilience data
     */
    extractResilienceData(modelData) {
      const baseScore = modelData.base_score || 0;
      const metric = modelData.metric || 'Score';
      const modelName = modelData.model_name || 'Primary Model';
      
      // Extract shifts
      const shifts = [];
      
      if (modelData.shifts && Array.isArray(modelData.shifts)) {
        // Shifts is already an array
        modelData.shifts.forEach(shift => {
          shifts.push({
            type: shift.type,
            score: shift.score,
            intensity: shift.intensity || 0.5
          });
        });
      } else if (modelData.shifts && typeof modelData.shifts === 'object') {
        // Shifts is an object with shift types as keys
        Object.entries(modelData.shifts).forEach(([shiftType, shiftData]) => {
          shifts.push({
            type: shiftType,
            score: shiftData.score || 0,
            intensity: shiftData.intensity || 0.5
          });
        });
      }
      
      // If no shifts found, create defaults
      if (shifts.length === 0) {
        this.defaultShiftTypes.forEach(shiftType => {
          shifts.push({
            type: shiftType,
            score: baseScore * (0.7 + Math.random() * 0.2), // Random score between 70-90% of base
            intensity: 0.5
          });
        });
      }
      
      // Extract feature impact
      let featureImpact = null;
      
      if (modelData.feature_impact) {
        featureImpact = modelData.feature_impact;
      }
      
      // Calculate resilience score as average of normalized scores
      const resilienceScore = shifts.reduce((sum, shift) => sum + (shift.score / baseScore), 0) / shifts.length;
      
      return {
        shifts,
        featureImpact,
        baseScore,
        metric,
        modelName,
        resilienceScore
      };
    }
    
    /**
     * Extract resilience data for an alternative model
     * @param {Object} modelData - Alternative model data
     * @param {Array} shiftTypes - Shift types from the primary model
     * @return {Object} Extracted resilience data for alternative model
     */
    extractResilienceDataForAlternativeModel(modelData, shiftTypes) {
      const baseScore = modelData.base_score || 0;
      
      // Extract shifts, matching primary model shift types
      const shifts = [];
      
      if (modelData.shifts && Array.isArray(modelData.shifts)) {
        // Extract shifts that match the primary model shift types
        shiftTypes.forEach(shiftType => {
          const shift = modelData.shifts.find(s => s.type === shiftType);
          
          if (shift) {
            shifts.push({
              type: shift.type,
              score: shift.score,
              intensity: shift.intensity || 0.5
            });
          } else {
            // Create default if not found
            shifts.push({
              type: shiftType,
              score: baseScore * (0.7 + Math.random() * 0.2), // Random score
              intensity: 0.5
            });
          }
        });
      } else if (modelData.shifts && typeof modelData.shifts === 'object') {
        // Shifts is an object with shift types as keys
        shiftTypes.forEach(shiftType => {
          const shiftData = modelData.shifts[shiftType];
          
          if (shiftData) {
            shifts.push({
              type: shiftType,
              score: shiftData.score || 0,
              intensity: shiftData.intensity || 0.5
            });
          } else {
            // Create default if not found
            shifts.push({
              type: shiftType,
              score: baseScore * (0.7 + Math.random() * 0.2), // Random score
              intensity: 0.5
            });
          }
        });
      }
      
      // Calculate resilience score
      const resilienceScore = shifts.reduce((sum, shift) => sum + (shift.score / baseScore), 0) / shifts.length;
      
      return {
        baseScore,
        shifts,
        resilienceScore
      };
    }
    
    /**
     * Generate default resilience data if no real data is available
     * @param {Object} reportData - Base report data
     * @return {Object} Generated resilience data
     */
    generateDefaultResilienceData(reportData) {
      const baseScore = reportData.base_score || 0.8;
      const metric = reportData.metric || 'Score';
      const modelName = reportData.model_name || 'Primary Model';
      
      // Generate default shifts
      const shifts = this.defaultShiftTypes.map(shiftType => {
        // Generate random score between 70-90% of base score
        const scoreReduction = (0.1 + Math.random() * 0.2);
        return {
          type: shiftType,
          score: baseScore * (1 - scoreReduction),
          intensity: 0.5
        };
      });
      
      // Generate random feature impact
      const featureCount = 10;
      const featureImpact = {};
      
      for (let i = 1; i <= featureCount; i++) {
        featureImpact[`feature_${i}`] = Math.random() * 0.5;
      }
      
      // Calculate resilience score
      const resilienceScore = shifts.reduce((sum, shift) => sum + (shift.score / baseScore), 0) / shifts.length;
      
      return {
        shifts,
        featureImpact,
        baseScore,
        metric,
        modelName,
        resilienceScore
      };
    }
    
    /**
     * Calculate performance metrics across shifts
     * @param {Object} resilienceData - Resilience data
     * @return {Object} Performance metrics
     */
    calculatePerformanceMetrics(resilienceData) {
      if (!resilienceData || !resilienceData.shifts || resilienceData.shifts.length === 0) {
        return {
          avgScore: null,
          minScore: null,
          maxScore: null,
          avgImpact: null,
          maxImpact: null,
          resilienceScore: null
        };
      }
      
      const baseScore = resilienceData.baseScore || 1.0;
      const shifts = resilienceData.shifts;
      
      // Calculate metrics
      const avgScore = shifts.reduce((sum, shift) => sum + shift.score, 0) / shifts.length;
      const minScore = Math.min(...shifts.map(shift => shift.score));
      const maxScore = Math.max(...shifts.map(shift => shift.score));
      
      // Calculate impacts
      const impacts = shifts.map(shift => (baseScore - shift.score) / baseScore);
      const avgImpact = impacts.reduce((sum, impact) => sum + impact, 0) / impacts.length;
      const maxImpact = Math.max(...impacts);
      
      // Calculate resilience score if not already available
      const resilienceScore = resilienceData.resilienceScore || 
                             (1 - avgImpact); // Fallback: 1 - average impact
      
      return {
        avgScore,
        minScore,
        maxScore,
        avgImpact,
        maxImpact,
        resilienceScore
      };
    }
    
    /**
     * Get worst shift type
     * @param {Object} resilienceData - Resilience data
     * @return {string} Worst shift type
     */
    getWorstShiftType(resilienceData) {
      if (!resilienceData || !resilienceData.shifts || resilienceData.shifts.length === 0) {
        return 'unknown';
      }
      
      // Find shift with lowest score
      let worstShift = resilienceData.shifts[0];
      let worstScore = worstShift.score;
      
      resilienceData.shifts.forEach(shift => {
        if (shift.score < worstScore) {
          worstScore = shift.score;
          worstShift = shift;
        }
      });
      
      return worstShift.type;
    }
  }
  
  export default ResilienceData;