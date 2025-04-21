// Formatter utility functions
const Formatters = {
    percent: function(value, decimals = 1) {
        return (value * 100).toFixed(decimals) + '%';
    },
    
    score: function(value, decimals = 3) {
        return value.toFixed(decimals);
    },
    
    impact: function(value, decimals = 2) {
        return (value * 100).toFixed(decimals) + '%';
    }
};
EOF < /dev/null
