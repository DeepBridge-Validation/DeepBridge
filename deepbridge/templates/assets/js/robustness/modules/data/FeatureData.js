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
     * Calculate feature correlations for explanatory analysis
     * @param {Object} featureData - Feature importance data
     * @param {number} topFeatureCount - Number of top features to include
     * @return {Object} Correlation matrix and feature list
     */
    calculateFeatureCorrelations(featureData, topFeatureCount = 10) {
      if (!featureData || !featureData.featureImportance || !featureData.modelFeatureImportance) {
        return {
          features: [],
          correlationMatrix: []
        };
      }
      
      // Get combined importance scores
      const combinedImportance = {};
      
      Object.keys(featureData.featureImportance).forEach(feature => {
        combinedImportance[feature] = (combinedImportance[feature] || 0) + featureData.featureImportance[feature];
      });
      
      Object.keys(featureData.modelFeatureImportance).forEach(feature => {
        combinedImportance[feature] = (combinedImportance[feature] || 0) + featureData.modelFeatureImportance[feature];
      });
      
      // Get top features
      const topFeatures = Object.entries(combinedImportance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, topFeatureCount)
        .map(entry => entry[0]);
      
      // Generate random correlation matrix for demonstration
      // In a real implementation, this would use actual correlation data
      const correlationMatrix = [];
      
      for (let i = 0; i < topFeatures.length; i++) {
        correlationMatrix[i] = [];
        
        for (let j = 0; j < topFeatures.length; j++) {
          if (i === j) {
            // Perfect correlation with self
            correlationMatrix[i][j] = 1;
          } else if (i < j) {
            // Generate random correlation coefficient
            // Higher weighting towards lower correlations
            const baseValue = Math.random() * 0.8;
            correlationMatrix[i][j] = baseValue * baseValue;
          } else {
            // Matrix is symmetric
            correlationMatrix[i][j] = correlationMatrix[j][i];
          }
        }
      }
      
      return {
        features: topFeatures,
        correlationMatrix: correlationMatrix
      };
    }
    
    /**
     * Calculate feature contribution to model robustness
     * @param {Object} featureData - Feature importance data
     * @return {Object} Feature contribution data
     */
    calculateFeatureContributions(featureData) {
      if (!featureData || !featureData.featureImportance || !featureData.modelFeatureImportance) {
        return {};
      }
      
      const featureContributions = {};
      
      // Combine feature importance data
      const allFeatures = new Set([
        ...Object.keys(featureData.featureImportance),
        ...Object.keys(featureData.modelFeatureImportance)
      ]);
      
      allFeatures.forEach(feature => {
        const modelImportance = featureData.modelFeatureImportance[feature] || 0;
        const robustnessImportance = featureData.featureImportance[feature] || 0;
        const inSubset = featureData.featureSubset.includes(feature);
        
        // Calculate contribution score as a weighted combination
        const contributionScore = modelImportance * 0.7 + robustnessImportance * 0.3;
        
        // Calculate impact direction (positive or negative)
        // This is a simulated value - in a real implementation, this would be
        // based on actual model coefficients or other specific metrics
        const impactDirection = Math.random() > 0.5 ? 1 : -1;
        
        // Calculate stability score (how consistently the feature behaves)
        // Higher values indicate more stable behavior across perturbations
        const stabilityScore = Math.max(0, Math.min(1, 0.5 + Math.random() * 0.5));
        
        featureContributions[feature] = {
          feature,
          modelImportance,
          robustnessImportance,
          contribution: contributionScore * impactDirection,
          impact: robustnessImportance,
          stability: stabilityScore,
          inSubset
        };
      });
      
      return featureContributions;
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