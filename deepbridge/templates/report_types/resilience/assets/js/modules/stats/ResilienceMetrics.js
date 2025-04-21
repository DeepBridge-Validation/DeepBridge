// ResilienceMetrics.js - placeholder
/**
 * Resilience Metrics
 * 
 * Provides metrics for measuring model resilience to distribution shifts,
 * including impact assessment and resilience scoring.
 */

import StatsCalculator from './StatsCalculator.js';

class ResilienceMetrics {
  /**
   * Initialize the resilience metrics calculator
   */
  constructor() {
    this.statsCalculator = new StatsCalculator();
  }
  
  /**
   * Calculate performance impact metrics
   * @param {Array} originalPerf - Performance metrics on original distribution
   * @param {Array} shiftedPerf - Performance metrics on shifted distribution
   * @return {Object} Impact metrics
   */
  calculatePerformanceImpact(originalPerf, shiftedPerf) {
    if (!originalPerf || !shiftedPerf || 
        originalPerf.length === 0 || shiftedPerf.length === 0) {
      return {
        originalMean: null,
        shiftedMean: null,
        absoluteChange: null,
        relativeChange: null,
        percentageChange: null,
        impactScore: null
      };
    }
    
    // Calculate average performance
    const originalMean = this.statsCalculator.calculateMean(originalPerf);
    const shiftedMean = this.statsCalculator.calculateMean(shiftedPerf);
    
    // Calculate impact metrics
    const absoluteChange = shiftedMean - originalMean;
    const relativeChange = absoluteChange / originalMean;
    const percentageChange = relativeChange * 100;
    
    // Calculate impact score (0-1 scale, higher means more impact)
    // Normalized to [0,1] range with sigmoid function
    const impactScore = 2 / (1 + Math.exp(-Math.abs(relativeChange) * 5)) - 1;
    
    return {
      originalMean,
      shiftedMean,
      absoluteChange,
      relativeChange,
      percentageChange,
      impactScore
    };
  }
  
  /**
   * Calculate resilience score based on performance impact and shift magnitude
   * @param {Object} performanceImpact - Performance impact metrics
   * @param {Object} distanceMetrics - Distribution distance metrics
   * @return {number} Resilience score (0-1, higher is better)
   */
  calculateResilienceScore(performanceImpact, distanceMetrics) {
    if (!performanceImpact || !distanceMetrics || 
        performanceImpact.impactScore === null || 
        (distanceMetrics.js_divergence === null && 
         distanceMetrics.wasserstein === null)) {
      return null;
    }
    
    // Normalize distance metrics to [0,1] scale
    let normalizedDistance;
    
    // Prefer JS divergence, fallback to Wasserstein
    if (distanceMetrics.js_divergence !== null) {
      // JS is already in [0,1]
      normalizedDistance = distanceMetrics.js_divergence;
    } else {
      // Normalize Wasserstein with sigmoid
      normalizedDistance = 2 / (1 + Math.exp(-distanceMetrics.wasserstein * 5)) - 1;
    }
    
    // Avoid very small distances to prevent division issues
    if (normalizedDistance < 0.01) {
      return 1.0; // Perfect resilience for negligible shifts
    }
    
    // Calculate resilience as inverse of (impact / distance)
    // Lower impact relative to shift magnitude means better resilience
    const rawResilience = 1 - (performanceImpact.impactScore / normalizedDistance);
    
    // Ensure result is in [0,1] range
    return Math.min(1, Math.max(0, rawResilience));
  }
  
  /**
   * Calculate feature-wise resilience metrics
   * @param {Object} featureShifts - Feature shift information
   * @param {Object} featurePerformanceImpacts - Feature performance impacts
   * @return {Object} Feature resilience metrics
   */
  calculateFeatureResilience(featureShifts, featurePerformanceImpacts) {
    if (!featureShifts || !featurePerformanceImpacts) {
      return {};
    }
    
    const featureResilience = {};
    
    // Calculate resilience for each feature
    Object.keys(featureShifts).forEach(feature => {
      const shiftInfo = featureShifts[feature];
      const impactInfo = featurePerformanceImpacts[feature];
      
      if (shiftInfo && impactInfo) {
        // Calculate feature resilience score
        const resilience = this.calculateResilienceScore(impactInfo, shiftInfo.distanceMetrics);
        
        featureResilience[feature] = {
          shiftMagnitude: shiftInfo.shiftMagnitude,
          performanceImpact: impactInfo.impactScore,
          resilience: resilience
        };
      }
    });
    
    return featureResilience;
  }
  
  /**
   * Calculate overall model resilience
   * @param {Object} performanceImpacts - Performance impacts at different shift levels
   * @param {Object} distanceMetrics - Distance metrics at different shift levels
   * @return {Object} Overall resilience metrics
   */
  calculateModelResilience(performanceImpacts, distanceMetrics) {
    if (!performanceImpacts || !distanceMetrics) {
      return {
        overallResilience: null,
        resilientThreshold: null,
        criticalShiftLevel: null
      };
    }
    
    // Calculate resilience at each shift level
    const resilienceScores = {};
    let sumResilience = 0;
    let countValidScores = 0;
    
    Object.keys(performanceImpacts).forEach(level => {
      const impact = performanceImpacts[level];
      const distance = distanceMetrics[level];
      
      if (impact && distance) {
        const resilience = this.calculateResilienceScore(impact, distance);
        
        if (resilience !== null) {
          resilienceScores[level] = resilience;
          sumResilience += resilience;
          countValidScores++;
        }
      }
    });
    
    // Calculate overall resilience (average across shift levels)
    const overallResilience = countValidScores > 0 ? 
      sumResilience / countValidScores : null;
    
    // Find critical shift level (where resilience drops below 0.5)
    let criticalShiftLevel = null;
    
    const sortedLevels = Object.keys(resilienceScores)
      .map(level => parseFloat(level))
      .sort((a, b) => a - b);
    
    for (const level of sortedLevels) {
      if (resilienceScores[level] < 0.5) {
        criticalShiftLevel = level;
        break;
      }
    }
    
    // Calculate resilient threshold (max shift level with resilience > 0.7)
    let resilientThreshold = null;
    
    for (let i = sortedLevels.length - 1; i >= 0; i--) {
      const level = sortedLevels[i];
      if (resilienceScores[level] >= 0.7) {
        resilientThreshold = level;
        break;
      }
    }
    
    return {
      overallResilience,
      resilientThreshold,
      criticalShiftLevel,
      resilienceByLevel: resilienceScores
    };
  }
  
  /**
   * Get qualitative resilience rating
   * @param {number} resilienceScore - Resilience score (0-1)
   * @return {string} Qualitative rating
   */
  getResilienceRating(resilienceScore) {
    if (resilienceScore === null || resilienceScore === undefined) {
      return 'Unknown';
    }
    
    if (resilienceScore >= 0.9) {
      return 'Excellent';
    } else if (resilienceScore >= 0.75) {
      return 'Good';
    } else if (resilienceScore >= 0.6) {
      return 'Moderate';
    } else if (resilienceScore >= 0.4) {
      return 'Fair';
    } else if (resilienceScore >= 0.2) {
      return 'Poor';
    } else {
      return 'Critical';
    }
  }
  
  /**
   * Generate recommendations based on resilience analysis
   * @param {Object} resilienceData - Complete resilience analysis data
   * @return {Array} List of recommendations
   */
  generateRecommendations(resilienceData) {
    if (!resilienceData || !resilienceData.overallResilience) {
      return [];
    }
    
    const recommendations = [];
    const featureResilience = resilienceData.featureResilience || {};
    
    // Overall resilience recommendations
    if (resilienceData.overallResilience < 0.6) {
      recommendations.push({
        type: 'critical',
        text: 'Model shows low overall resilience to distribution shifts. Consider model retraining with augmented data.'
      });
    }
    
    // Find most vulnerable features
    const vulnerableFeatures = Object.entries(featureResilience)
      .filter(([_, metrics]) => metrics.resilience < 0.5)
      .map(([feature, _]) => feature);
    
    if (vulnerableFeatures.length > 0) {
      recommendations.push({
        type: 'warning',
        text: `The following features have low resilience and are potential weak points: ${vulnerableFeatures.join(', ')}.`
      });
    }
    
    // Find most robust features
    const robustFeatures = Object.entries(featureResilience)
      .filter(([_, metrics]) => metrics.resilience > 0.8)
      .map(([feature, _]) => feature);
    
    if (robustFeatures.length > 0) {
      recommendations.push({
        type: 'positive',
        text: `These features show high resilience and should be emphasized: ${robustFeatures.join(', ')}.`
      });
    }
    
    // Shift level recommendations
    if (resilienceData.criticalShiftLevel !== null) {
      recommendations.push({
        type: 'info',
        text: `Critical distribution shift threshold occurs at level ${resilienceData.criticalShiftLevel}. Monitor for shifts approaching this magnitude.`
      });
    }
    
    return recommendations;
  }
}

export default ResilienceMetrics;