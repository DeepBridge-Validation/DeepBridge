class ChartRenderer {
    constructor(chartFactory) {
      this.chartFactory = chartFactory;
    }
    
    renderPrimaryModelBoxplot(container, perturbationData) {
      if (!container || !perturbationData || !perturbationData.distribution) {
        return;
      }
      
      const distribution = perturbationData.distribution;
      const primaryModel = distribution.primaryModel;
      
      // Prepare data for boxplot chart
      const boxplotSeries = [{
        name: primaryModel.name,
        data: primaryModel.distributions.map(dist => ({
          x: dist.level,
          low: dist.min,
          q1: dist.q1,
          median: dist.median,
          q3: dist.q3,
          high: dist.max,
          mean: dist.mean,
          outliers: dist.outliers
        })),
        color: '#1b78de'
      }];
      
      // Create line series for the mean scores
      const lineSeries = [{
        name: 'Mean Score',
        type: 'line',
        data: primaryModel.distributions.map(dist => ({
          x: dist.level,
          y: dist.mean
        })),
        color: '#ff7300',
        marker: {
          enabled: true,
          radius: 4
        }
      }];
      
      // Add reference line for base score
      const referenceSeries = [{
        name: 'Base Score',
        type: 'line',
        data: [
          { x: distribution.levels[0], y: perturbationData.baseScore },
          { x: distribution.levels[distribution.levels.length - 1], y: perturbationData.baseScore }
        ],
        color: '#2ecc71',
        marker: { enabled: false },
        enableMouseTracking: false,
        dashStyle: 'dash'
      }];
      
      const chartConfig = {
        title: `${primaryModel.name} Performance Distribution (${perturbationData.metric || 'Score'})`,
        xAxis: {
          title: 'Perturbation Level',
          categories: distribution.levels
        },
        yAxis: {
          title: perturbationData.metric || 'Score',
          min: Math.max(0, Math.min(...primaryModel.distributions.map(d => d.min)) - 0.05),
          max: Math.min(1, Math.max(perturbationData.baseScore, ...primaryModel.distributions.map(d => d.max)) + 0.05)
        },
        boxplotSeries: boxplotSeries,
        lineSeries: lineSeries,
        referenceSeries: referenceSeries
      };
      
      // Create the chart
      this.chartFactory.createBoxplotChart(container, chartConfig);
    }
    
    // Other chart rendering methods
  }
  
  export default ChartRenderer;