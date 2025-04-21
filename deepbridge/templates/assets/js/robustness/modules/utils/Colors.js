// Color utility functions
const Colors = {
    palette: {
        primary: "#1b78de",
        secondary: "#2c3e50",
        success: "#2ecc71",
        warning: "#f39c12",
        danger: "#e74c3c",
        info: "#3498db",
        light: "#f8f9fa",
        dark: "#343a40"
    },
    
    getChartColors: function(count) {
        const baseColors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ];
        
        if (count <= baseColors.length) {
            return baseColors.slice(0, count);
        }
        
        // If more colors needed, generate them
        const result = [...baseColors];
        for (let i = baseColors.length; i < count; i++) {
            const hue = (i * 137.5) % 360;
            result.push(`hsl(${hue}, 70%, 50%)`);
        }
        
        return result;
    }
};
