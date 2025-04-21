/**
 * Robustness Metrics
 * 
 * Specialized metrics for evaluating model robustness,
 * including impact measures, stability scores, and comparative analysis.
 */
import StatsCalculator from './StatsCalculator.js';
import Correlation from './Correlation.js';

class RobustnessMetrics {
  /**
   * Initialize the robustness metrics calculator
   */
  constructor() {
    // Initialize dependencies
    this.statsCalculator = new StatsCalculator();
    this.correlation = new Correlation();
  }
  
  /**
   * Calculate robustness score from perturbation data
   * @param {Array} scores - Scores at different perturbation levels
   * @param {number} baseScore - Base score (unperturbed)
   * @return {number} Robustness score (0-1)
   */
  calculateRobustnessScore(scores, baseScore) {
    if (!scores || scores.length === 0 || !baseScore || baseScore === 0) {
      return null;
    }
    
    // Calculate normalized scores
    const normalizedScores = scores.map(score => score / baseScore);
    
    // Robustness score is the average of normalized scores
    return normalizedScores.reduce((sum, score) => sum + score, 0) / normalizedScores.length;
  }
  
  /**
   * Calculate impact metrics for perturbation results
   * @param {Array} scores - Scores at different perturbation levels
   * @param {number} baseScore - Base score (unperturbed)
   * @return {Object} Impact metrics
   */
  calculateImpactMetrics(scores, baseScore) {
    if (!scores || scores.length === 0 || !baseScore) {
      return {
        avgImpact: null,
        maxImpact: null,
        impactVariance: null,
        areaUnderCurve: null
      };
    }
    
    // Calculate impacts (proportional decrease from base score)
    const impacts = scores.map(score => (baseScore - score) / baseScore);
    
    // Calculate average impact
    const avgImpact = impacts.reduce((sum, impact) => sum + impact, 0) / impacts.length;
    
    // Calculate maximum impact
    const maxImpact = Math.max(...impacts);
    
    // Calculate variance of impacts
    const impactVariance = impacts.reduce(
      (sum, impact) => sum + Math.pow(impact - avgImpact, 2), 0
    ) / impacts.length;
    
    // Calculate area under the impact curve (normalized 0-1)
    const areaUnderCurve = impacts.reduce((sum, impact) => sum + impact, 0) / impacts.length;
    
    return {
      avgImpact,
      maxImpact,
      impactVariance,
      impactStdDev: Math.sqrt(impactVariance),
      areaUnderCurve
    };
  }
  
  /**
   * Calculate stability score
   * @param {Array} scores - Scores at different perturbation levels
   * @param {Array} levels - Perturbation levels
   * @return {number} Stability score (0-1)
   */
  calculateStabilityScore(scores, levels) {
    if (!scores || !levels || scores.length !== levels.length || scores.length < 2) {
      return null;
    }
    
    // Calculate regression
    const regression = this.correlation.calculateLinearRegression(levels, scores);
    
    // Calculate predicted values
    const predicted = levels.map(level => regression.slope * level + regression.intercept);
    
    // Calculate R-squared
    const rSquared = this.correlation.calculateRSquared(scores, predicted);
    
    // Calculate stability as a function of R-squared and slope
    // Higher R-squared means more predictable degradation (more stable)
    // Less negative slope means more gradual degradation (more stable)
    
    // Normalize the slope to -1 to 0 range (assuming slope is negative)
    const normalizedSlope = Math.min(0, Math.max(-1, regression.slope));
    const slopeComponent = 1 + normalizedSlope; // 0 for slope of -1, 1 for slope of 0
    
    // Combine with R-squared
    const stabilityScore = (rSquared * 0.5) + (slopeComponent * 0.5);
    
    return stabilityScore;
  }
  
  /**
   * Calculate feature importance for robustness
   * @param {Array} baseScores - Base scores for each feature subset
   * @param {Array} perturbedScores - Perturbed scores for each feature subset
   * @param {Array} features - Feature names
   * @return {Object} Feature importance scores
   */
  calculateFeatureImportance(baseScores, perturbedScores, features) {
    if (!baseScores || !perturbedScores || !features || 
        baseScores.length !== perturbedScores.length || 
        baseScores.length !== features.length) {
      return {};
    }
    
    const importance = {};
    
    // Calculate impact for each feature
    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      const baseScore = baseScores[i];
      const perturbedScore = perturbedScores[i];
      
      // Impact when feature is perturbed
      const impact = (baseScore - perturbedScore) / baseScore;
      
      // Higher impact means higher importance for robustness
      importance[feature] = impact;
    }
    
    return importance;
  }
  
  /**
   * Find optimal feature subset for robustness
   * @param {Object} featureImportance - Feature importance scores
   * @param {Object} modelFeatureImportance - Model feature importance scores
   * @param {number} subsetSize - Size of subset to select
   * @return {Array} Optimal feature subset
   */
  findOptimalFeatureSubset(featureImportance, modelFeatureImportance, subsetSize = 10) {
    if (!featureImportance || !modelFeatureImportance) {
      return [];
    }
    
    // Get all unique features
    const allFeatures = new Set([
      ...Object.keys(featureImportance),
      ...Object.keys(modelFeatureImportance)
    ]);
    
    // Calculate combined score for each feature
    const featureScores = [];
    
    allFeatures.forEach(feature => {
      const robustnessImportance = featureImportance[feature] || 0;
      const modelImportance = modelFeatureImportance[feature] || 0;
      
      // Combined score with weightings
      const combinedScore = (robustnessImportance * 0.5) + (modelImportance * 0.5);
      
      featureScores.push({
        feature,
        robustnessImportance,
        modelImportance,
        combinedScore
      });
    });
    
    // Sort by combined score (highest first)
    featureScores.sort((a, b) => b.combinedScore - a.combinedScore);
    
    // Select top features
    return featureScores.slice(0, subsetSize).map(item => item.feature);
  }
  
  /**
   * Calculate comparative metrics for model comparison
   * @param {Array} models - Array of model data objects
   * @return {Object} Comparative metrics
   */
  calculateComparativeMetrics(models) {
    if (!models || models.length === 0) {
      return {
        bestModel: null,
        avgRobustness: null,
        bestRobustness: null,
        worstRobustness: null,
        robustnessSpread: null
      };
    }
    
    // Calculate robustness scores
    const robustnessScores = models.map(model => {
      const baseScore = model.baseScore || 1;
      const avgScore = model.scores.reduce((sum, score) => sum + (score || 0), 0) / 
                      model.scores.filter(score => score !== null).length;
      
      return (avgScore / baseScore);
    });
    
    // Find best and worst models
    const bestIndex = robustnessScores.indexOf(Math.max(...robustnessScores));
    const worstIndex = robustnessScores.indexOf(Math.min(...robustnessScores));
    
    return {
      bestModel: models[bestIndex].name,
      avgRobustness: robustnessScores.reduce((sum, score) => sum + score, 0) / robustnessScores.length,
      bestRobustness: robustnessScores[bestIndex],
      worstRobustness: robustnessScores[worstIndex],
      robustnessSpread: robustnessScores[bestIndex] - robustnessScores[worstIndex]
    };
  }
  
  /**
   * Calculate level metrics for different perturbation levels
   * @param {Object} perturbationData - Perturbation data object
   * @return {Array} Metrics for each level
   */
  calculateLevelMetrics(perturbationData) {
    if (!perturbationData || !perturbationData.levels || !perturbationData.scores) {
      return [];
    }
    
    const { levels, scores, worstScores, baseScore } = perturbationData;
    const metrics = [];
    
    // Calculate metrics for each level
    for (let i = 0; i < levels.length; i++) {
      const level = levels[i];
      const score = scores[i];
      const worstScore = worstScores[i];
      
      // Skip if score is missing
      if (score === null) continue;
      
      // Calculate impact (proportional decrease from base score)
      const impact = (baseScore - score) / baseScore;
      
      // Calculate robustness (1 - impact)
      const robustness = 1 - impact;
      
      // Calculate relative variability (difference between mean and worst score)
      const variability = score - worstScore;
      const relativeVariability = variability / score;
      
      // Calculate stability (1 - normalized variability)
      const stability = 1 - Math.min(1, relativeVariability);
      
      metrics.push({
        level,
        score,
        worstScore,
        impact,
        robustness,
        variability,
        relativeVariability,
        stability
      });
    }
    
    return metrics;
  }
  
  /**
   * Evaluate model response to different perturbation types
   * @param {Object} rawData - Raw perturbation data
   * @param {Object} quantileData - Quantile perturbation data
   * @return {Object} Comparative analysis of perturbation types
   */
  comparePerturbationTypes(rawData, quantileData) {
    if (!rawData || !quantileData) {
      return {
        moreRobustTo: null,
        rawImpact: null,
        quantileImpact: null,
        differenceRatio: null
      };
    }
    
    // Calculate average impacts
    const rawImpact = this.calculateImpactMetrics(rawData.scores, rawData.baseScore).avgImpact;
    const quantileImpact = this.calculateImpactMetrics(quantileData.scores, quantileData.baseScore).avgImpact;
    
    // Calculate difference ratio
    const differenceRatio = rawImpact > 0 ? quantileImpact / rawImpact : 1;
    
    // Determine which type the model is more robust to
    const moreRobustTo = rawImpact < quantileImpact ? 'raw' : 'quantile';
    
    return {
      moreRobustTo,
      rawImpact,
      quantileImpact,
      differenceRatio
    };
  }
}

export default RobustnessMetrics;