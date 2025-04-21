class TableRenderer {
    constructor() {
      // Initialize any needed properties
    }
    
    formatNumber(value, precision = 4) {
      if (value === null || value === undefined) {
        return 'N/A';
      }
      return value.toFixed(precision);
    }
    
    renderStatisticalSummary(tableBody, perturbationData) {
      if (!tableBody || !perturbationData || !perturbationData.distribution) {
        return;
      }
      
      const distribution = perturbationData.distribution;
      const primaryModel = distribution.primaryModel;
      
      let html = '';
      
      // Add a row for each perturbation level
      distribution.levels.forEach((level, index) => {
        const dist = primaryModel.distributions[index];
        if (!dist) return;
        
        // Calculate standard deviation (synthetic)
        const stdDev = (dist.q3 - dist.q1) / 1.35; // Approximation based on normal distribution
        const iqr = dist.q3 - dist.q1;
        
        html += `
          <tr>
            <td>${level}</td>
            <td>${this.formatNumber(dist.mean, 4)}</td>
            <td>${this.formatNumber(dist.median, 4)}</td>
            <td>${this.formatNumber(stdDev, 4)}</td>
            <td>${this.formatNumber(dist.min, 4)}</td>
            <td>${this.formatNumber(dist.max, 4)}</td>
            <td>${this.formatNumber(iqr, 4)}</td>
          </tr>
        `;
      });
      
      // If no data, show a message
      if (html === '') {
        html = `<tr><td colspan="7">No statistical data available</td></tr>`;
      }
      
      tableBody.innerHTML = html;
    }
    
    // Other table rendering methods
  }
  
  export default TableRenderer;