// DataModel.js
/**
 * Data model for the resilience report
 * 
 * This class processes and provides structured access to the report data
 */
class DataModel {
    /**
     * Initialize the data model
     * @param {Object} reportData - The raw resilience report data
     */
    constructor(reportData) {
        this.reportData = reportData || {};
        this.distributionShiftResults = this.extractDistributionShiftResults();
        this.features = this.extractFeatures();
        this.featureDistances = this.extractFeatureDistances();
    }
    
    /**
     * Extract distribution shift results from the report data
     * @returns {Array} - Array of distribution shift results
     */
    extractDistributionShiftResults() {
        const results = [];
        
        try {
            // Try to get from standardized location
            if (this.reportData.distribution_shift_results) {
                return this.reportData.distribution_shift_results;
            }
            
            // Try to get from distribution_shift path
            if (this.reportData.distribution_shift && this.reportData.distribution_shift.all_results) {
                return this.reportData.distribution_shift.all_results;
            }
            
            // Try to extract from nested results
            if (this.reportData.results && this.reportData.results.distribution_shift) {
                const distShift = this.reportData.results.distribution_shift;
                if (distShift.all_results) {
                    return distShift.all_results;
                }
            }
        } catch (error) {
            console.error('Error extracting distribution shift results:', error);
        }
        
        return results;
    }
    
    /**
     * Extract features from the report data
     * @returns {Array} - Array of feature names
     */
    extractFeatures() {
        const features = new Set();
        
        try {
            // Try to get from feature subset
            if (this.reportData.feature_subset && Array.isArray(this.reportData.feature_subset)) {
                this.reportData.feature_subset.forEach(f => features.add(f));
            }
            
            // Try to get from feature_distances
            if (this.featureDistances) {
                Object.keys(this.featureDistances).forEach(f => features.add(f));
            }
        } catch (error) {
            console.error('Error extracting features:', error);
        }
        
        return Array.from(features);
    }
    
    /**
     * Extract feature distances from the report data
     * @returns {Object} - Feature distances by feature name
     */
    extractFeatureDistances() {
        try {
            // Try to get from standardized location
            if (this.reportData.feature_distances) {
                return this.reportData.feature_distances;
            }
            
            // Try to get from distribution_shift
            if (this.reportData.distribution_shift && 
                this.reportData.distribution_shift.by_distance_metric) {
                
                const distances = {};
                const metrics = this.reportData.distribution_shift.by_distance_metric;
                
                // Combine distances from all metrics
                Object.values(metrics).forEach(metricData => {
                    if (metricData.avg_feature_distances) {
                        Object.entries(metricData.avg_feature_distances).forEach(([feature, value]) => {
                            if (!distances[feature]) {
                                distances[feature] = {};
                            }
                            distances[feature] = value;
                        });
                    }
                });
                
                return distances;
            }
        } catch (error) {
            console.error('Error extracting feature distances:', error);
        }
        
        return {};
    }
    
    /**
     * Get distribution shift results
     * @returns {Array} - Array of distribution shift results
     */
    getDistributionShiftResults() {
        return this.distributionShiftResults;
    }
    
    /**
     * Get list of features
     * @returns {Array} - Array of feature names
     */
    getFeatures() {
        return this.features;
    }
    
    /**
     * Get baseline distribution for a specific feature
     * This is a placeholder - in a real implementation, this would extract 
     * actual distribution data from the results
     * @param {string} feature - The feature name
     * @returns {Array} - Array of values or null if not available
     */
    getBaselineDistribution(feature) {
        // Placeholder - would need real data extraction
        return this.generateRandomDistribution();
    }
    
    /**
     * Get target distribution for a specific feature
     * This is a placeholder - in a real implementation, this would extract 
     * actual distribution data from the results
     * @param {string} feature - The feature name
     * @returns {Array} - Array of values or null if not available
     */
    getTargetDistribution(feature) {
        // Placeholder - would need real data extraction
        return this.generateRandomDistribution(0.2);
    }
    
    /**
     * Helper to generate random distribution for demo purposes
     * @param {number} shift - Amount to shift the distribution
     * @returns {Array} - Array of random values
     */
    generateRandomDistribution(shift = 0) {
        const values = [];
        const count = 100;
        const mean = 0.5 + shift;
        const stdDev = 0.15;
        
        for (let i = 0; i < count; i++) {
            // Generate normally distributed random values
            let u = 0, v = 0;
            while (u === 0) u = Math.random();
            while (v === 0) v = Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            values.push(mean + z * stdDev);
        }
        
        return values;
    }
    
    /**
     * Get distribution metrics for a specific feature
     * This is a placeholder - in a real implementation, this would extract 
     * actual metrics from the results
     * @param {string} feature - The feature name
     * @returns {Object} - Distribution metrics or null if not available
     */
    getDistributionMetrics(feature) {
        // In a real implementation, this would return actual metrics from the results
        return {
            klDivergence: Math.random() * 0.5,
            jsDistance: Math.random() * 0.3,
            wasserstein: Math.random() * 0.2,
            hellinger: Math.random() * 0.4,
            
            baselineMean: 0.5,
            targetMean: 0.6,
            meanChange: 0.1,
            
            baselineMedian: 0.5,
            targetMedian: 0.6,
            medianChange: 0.1,
            
            baselineStd: 0.15,
            targetStd: 0.18,
            stdChange: 0.03,
            
            baselineIQR: 0.2,
            targetIQR: 0.25,
            iqrChange: 0.05
        };
    }
    
    /**
     * Get feature shift magnitudes
     * @returns {Object} - Map of feature names to shift magnitudes
     */
    getFeatureShiftMagnitudes() {
        const magnitudes = {};
        
        try {
            if (this.reportData.distribution_shift && 
                this.reportData.distribution_shift.by_distance_metric) {
                
                const metrics = this.reportData.distribution_shift.by_distance_metric;
                
                // Use PSI as default if available
                if (metrics.PSI && metrics.PSI.top_features) {
                    return metrics.PSI.top_features;
                }
                
                // Use any available metric
                for (const metric in metrics) {
                    if (metrics[metric].top_features) {
                        return metrics[metric].top_features;
                    }
                }
            }
            
            // If no structured data, use feature distances
            if (this.featureDistances) {
                // Just copy the values
                Object.keys(this.featureDistances).forEach(feature => {
                    magnitudes[feature] = this.featureDistances[feature];
                });
                return magnitudes;
            }
        } catch (error) {
            console.error('Error getting feature shift magnitudes:', error);
        }
        
        // If no data available, provide a small set of random data
        if (Object.keys(magnitudes).length === 0 && this.features.length > 0) {
            this.features.slice(0, 10).forEach(feature => {
                magnitudes[feature] = Math.random() * 0.5;
            });
        }
        
        return magnitudes;
    }
    
    /**
     * Get all feature distance metrics
     * @returns {Array} - Array of feature metrics
     */
    getAllFeatureDistanceMetrics() {
        const metrics = [];
        
        try {
            // Get available distance metrics
            const distanceMetrics = this.reportData.distance_metrics || ['PSI', 'KS', 'WD1'];
            
            // Construct metrics for each feature
            this.features.forEach(feature => {
                const featureMetrics = {
                    feature: feature,
                    impact: Math.random() * 0.3 // Placeholder - would use real data
                };
                
                // Add metrics for each distance type
                distanceMetrics.forEach(metricType => {
                    if (this.reportData.distribution_shift && 
                        this.reportData.distribution_shift.by_distance_metric && 
                        this.reportData.distribution_shift.by_distance_metric[metricType] && 
                        this.reportData.distribution_shift.by_distance_metric[metricType].avg_feature_distances && 
                        this.reportData.distribution_shift.by_distance_metric[metricType].avg_feature_distances[feature]) {
                        
                        featureMetrics[metricType] = this.reportData.distribution_shift.by_distance_metric[metricType].avg_feature_distances[feature];
                    } else {
                        featureMetrics[metricType] = null;
                    }
                });
                
                metrics.push(featureMetrics);
            });
        } catch (error) {
            console.error('Error getting all feature distance metrics:', error);
        }
        
        return metrics;
    }
}