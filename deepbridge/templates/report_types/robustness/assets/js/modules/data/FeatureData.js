/**
 * Feature Data Handler
 * 
 * Specialized handler for extracting and processing feature importance data
 * from the robustness report structure.
 */

class FeatureData {
    /**
     * Initialize the feature data handler
     */
    constructor() {
      // Default optimal subset size
      this.defaultSubsetSize = 10;
    }
    
    /**
     * Extract feature importance data from report data
     * @param {Object} reportData - The full report data object
     * @return {Object|null} Extracted feature data or null if not available
     */
    extract(reportData) {
      if (!reportData) return null;
      
      const featureData = {
        featureImportance: {},
        modelFeatureImportance: {},
        featureSubset: []
      };
      
      // Handle different possible locations of feature importance data
      
      // Check if feature importance is directly in reportData
      if (reportData.feature_importance) {
        featureData.featureImportance = reportData.feature_importance;
      } 
      // Check if feature importance is in a primary_model property
      else if (reportData.primary_model && reportData.primary_model.feature_importance) {
        featureData.featureImportance = reportData.primary_model.feature_importance;
      }
      
      // Check if model feature importance is directly in reportData
      if (reportData.model_feature_importance) {
        featureData.modelFeatureImportance = reportData.model_feature_importance;
      } 
      // Check if model feature importance is in a primary_model property
      else if (reportData.primary_model && reportData.primary_model.model_feature_importance) {
        featureData.modelFeatureImportance = reportData.primary_model.model_feature_importance;
      }
      
      // Check if feature subset is directly in reportData
      if (reportData.feature_subset) {
        featureData.featureSubset = reportData.feature_subset;
      } 
      // Check if feature subset is in a primary_model property
      else if (reportData.primary_model && reportData.primary_model.feature_subset) {
        featureData.featureSubset = reportData.primary_model.feature_subset;
      }
      // If no feature subset is provided, compute an optimal subset
      else if (Object.keys(featureData.featureImportance).length > 0 || 
              Object.keys(featureData.modelFeatureImportance).length > 0) {
        featureData.featureSubset = this.findOptimalFeatureSubset(
          featureData.featureImportance,
          featureData.modelFeatureImportance
        );
      }
      
      // Return null if no data was found
      if (Object.keys(featureData.featureImportance).length === 0 && 
          Object.keys(featureData.modelFeatureImportance).length === 0) {
        return null;
      }
      
      return featureData;
    }
    
    /**
     * Find the optimal feature subset based on importance and robustness
     * @param {Object} featureImportance - Feature importance values
     * @param {Object} modelFeatureImportance - Model feature importance values
     * @param {number} subsetSize - Size of the optimal subset
     * @return {Array} Optimal feature subset
     */
    findOptimalFeatureSubset(featureImportance, modelFeatureImportance, subsetSize = this.defaultSubsetSize) {
      // Calculate combined score for each feature
      const featureScores = {};
      
      // Start with model importance
      Object.entries(modelFeatureImportance).forEach(([feature, importance]) => {
        featureScores[feature] = {
          feature,
          modelImportance: importance,
          robustnessImportance: 0,
          combinedScore: importance * 0.5 // 50% weight for model importance initially
        };
      });
      
      // Add robustness importance
      Object.entries(featureImportance).forEach(([feature, importance]) => {
        if (featureScores[feature]) {
          featureScores[feature].robustnessImportance = importance;
          featureScores[feature].combinedScore += importance * 0.5; // 50% weight for robustness importance
        } else {
          featureScores[feature] = {
            feature,
            modelImportance: 0,
            robustnessImportance: importance,
            combinedScore: importance * 0.5
          };
        }
      });
      
      // Sort features by combined score and return top N
      return Object.values(featureScores)
        .sort((a, b) => b.combinedScore - a.combinedScore)
        .slice(0, subsetSize)
        .map(item => item.feature);
    }
    
    /**
     * Check if feature correlation data is available
     * @param {Object} featureData - Feature importance data
     * @param {number} topFeatureCount - Number of top features to include
     * @return {Object|null} Correlation matrix and feature list, or null if data not available
     */
    calculateFeatureCorrelations(featureData, topFeatureCount = 10) {
      if (!featureData || !featureData.featureImportance || !featureData.modelFeatureImportance) {
        console.warn("No feature importance data available for correlation calculation");
        return null;
      }
      
      // Check if feature correlation data is available in the input
      if (featureData.correlationMatrix && featureData.correlationFeatures) {
        // Use actual correlation data from the report
        return {
          features: featureData.correlationFeatures,
          correlationMatrix: featureData.correlationMatrix
        };
      }
      
      // If actual correlation data is not available, return null
      console.warn("No feature correlation data available in the report data");
      return null;
    }
    
    /**
     * Check for feature contribution data or calculate from available metrics
     * @param {Object} featureData - Feature importance data
     * @return {Object|null} Feature contribution data or null if data not available
     */
    calculateFeatureContributions(featureData) {
      if (!featureData || !featureData.featureImportance || !featureData.modelFeatureImportance) {
        console.warn("No feature importance data available for contribution calculation");
        return null;
      }
      
      // Check if actual feature contribution data is present in the input
      if (featureData.featureContributions) {
        return featureData.featureContributions;
      }
      
      // If we have detailed contribution metrics data
      if (featureData.featureMetrics) {
        const featureContributions = {};
        
        Object.keys(featureData.featureMetrics).forEach(feature => {
          const metrics = featureData.featureMetrics[feature];
          const modelImportance = featureData.modelFeatureImportance[feature] || 0;
          const robustnessImportance = featureData.featureImportance[feature] || 0;
          const inSubset = featureData.featureSubset.includes(feature);
          
          // Use actual metrics data if available
          featureContributions[feature] = {
            feature,
            modelImportance,
            robustnessImportance,
            contribution: metrics.contribution || (modelImportance + robustnessImportance) / 2,
            impact: metrics.impact || robustnessImportance,
            stability: metrics.stability || 0.5,
            inSubset
          };
        });
        
        return featureContributions;
      }
      
      // If we don't have detailed contribution data, just return null
      console.warn("No feature contribution metrics available in the report data");
      return null;
    }
    
    /**
     * Calculate discrepancy between model and robustness importance
     * @param {Object} featureData - Feature importance data
     * @return {Array} Features sorted by importance discrepancy
     */
    calculateImportanceDiscrepancy(featureData) {
      if (!featureData || !featureData.featureImportance || !featureData.modelFeatureImportance) {
        return [];
      }
      
      const allFeatures = new Set([
        ...Object.keys(featureData.featureImportance),
        ...Object.keys(featureData.modelFeatureImportance)
      ]);
      
      const discrepancies = [];
      
      allFeatures.forEach(feature => {
        const modelImportance = featureData.modelFeatureImportance[feature] || 0;
        const robustnessImportance = featureData.featureImportance[feature] || 0;
        const inSubset = featureData.featureSubset.includes(feature);
        
        // Calculate absolute discrepancy
        const discrepancy = Math.abs(modelImportance - robustnessImportance);
        
        discrepancies.push({
          feature,
          modelImportance,
          robustnessImportance,
          discrepancy,
          inSubset
        });
      });
      
      // Sort by discrepancy (largest first)
      return discrepancies.sort((a, b) => b.discrepancy - a.discrepancy);
    }
  }
  
  export default FeatureData;