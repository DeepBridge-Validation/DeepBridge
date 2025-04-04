document.addEventListener('DOMContentLoaded', function() {
  // Initialize tablesort on all tables with the 'sortable' class
  const tables = document.querySelectorAll('table.sortable');
  tables.forEach(function(table) {
    new Tablesort(table);
  });
  
  // Add sortable class to API tables and data tables automatically
  const apiTables = document.querySelectorAll('.md-typeset table:not(.no-sort)');
  apiTables.forEach(function(table) {
    table.classList.add('sortable');
    new Tablesort(table);
  });

  // Add custom sort types for specific data formats
  
  // Sort for semantic versioning
  Tablesort.extend('semver', function(item) {
    return /^\d+\.\d+\.\d+(-.*)?$/.test(item);
  }, function(a, b) {
    const aComponents = a.split('.');
    const bComponents = b.split('.');
    
    // Compare major version
    if (parseInt(aComponents[0]) !== parseInt(bComponents[0])) {
      return parseInt(aComponents[0]) - parseInt(bComponents[0]);
    }
    
    // Compare minor version
    if (parseInt(aComponents[1]) !== parseInt(bComponents[1])) {
      return parseInt(aComponents[1]) - parseInt(bComponents[1]);
    }
    
    // Compare patch version (handling potential suffixes like -alpha, -beta)
    const aPatch = aComponents[2].split('-')[0];
    const bPatch = bComponents[2].split('-')[0];
    
    if (parseInt(aPatch) !== parseInt(bPatch)) {
      return parseInt(aPatch) - parseInt(bPatch);
    }
    
    // Compare suffixes if they exist
    const aSuffix = aComponents[2].includes('-') ? aComponents[2].split('-')[1] : '';
    const bSuffix = bComponents[2].includes('-') ? bComponents[2].split('-')[1] : '';
    
    if (aSuffix === '' && bSuffix !== '') return 1;
    if (aSuffix !== '' && bSuffix === '') return -1;
    
    return aSuffix.localeCompare(bSuffix);
  });
  
  // Sort for percentages
  Tablesort.extend('percentage', function(item) {
    return /^\d+(\.\d+)?%$/.test(item);
  }, function(a, b) {
    return parseFloat(a) - parseFloat(b);
  });
  
  // Sort for file sizes (KB, MB, GB)
  Tablesort.extend('filesize', function(item) {
    return /^\d+(\.\d+)?\s*(B|KB|MB|GB|TB)$/.test(item);
  }, function(a, b) {
    const units = {
      'B': 1,
      'KB': 1024,
      'MB': 1024 * 1024,
      'GB': 1024 * 1024 * 1024,
      'TB': 1024 * 1024 * 1024 * 1024
    };
    
    function parseFileSize(size) {
      const parts = size.trim().split(/\s+/);
      const value = parseFloat(parts[0]);
      const unit = parts[1];
      return value * units[unit];
    }
    
    return parseFileSize(a) - parseFileSize(b);
  });
  
  // Add custom styling to sorted columns
  document.querySelectorAll('th').forEach(function(th) {
    th.addEventListener('click', function() {
      // Reset all th styling
      document.querySelectorAll('th').forEach(function(header) {
        header.classList.remove('sort-active');
      });
      
      // Add active class to current sorted header
      this.classList.add('sort-active');
    });
  });
});