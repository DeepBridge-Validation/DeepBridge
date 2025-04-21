/**
 * Utilitários para renderização de componentes UI nos relatórios de robustez
 * 
 * Este módulo concentra funções para manipular o DOM, renderizar componentes
 * e lidar com erros de forma consistente.
 */

const RenderingUtils = {
    /**
     * Renderiza um gráfico Plotly de forma segura, com tratamento de erros
     */
    safelyRenderPlot: function(containerId, traces, layout, config = {responsive: true}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Elemento container com ID '${containerId}' não encontrado`);
            return;
        }
        
        try {
            Plotly.newPlot(containerId, traces, layout, config);
        } catch (error) {
            console.error(`Erro ao renderizar gráfico em '${containerId}':`, error);
            this.showErrorMessage(containerId, 'Erro ao renderizar gráfico', error.message);
        }
    },
    
    /**
     * Mostra uma mensagem de erro padronizada em um container
     */
    showErrorMessage: function(containerId, title, message) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = `
            <div class="alert alert-warning">
                <strong>${title}</strong><br>
                ${message}
            </div>
        `;
    },
    
    /**
     * Mostra uma mensagem padronizada de "dados não disponíveis"
     */
    showNoDataMessage: function(containerId, title, message) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = `
            <div class="alert alert-info">
                <strong>${title}</strong><br>
                ${message}
            </div>
        `;
    },
    
    /**
     * Renderiza uma tabela de dados com cabeçalhos e linhas fornecidos
     */
    renderDataTable: function(containerId, headers, data, rowFormatter) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        if (!data || data.length === 0) {
            this.showNoDataMessage(containerId, 'Dados não disponíveis', 'Não há dados para exibir nesta tabela.');
            return;
        }
        
        let html = `
            <table class="data-table">
                <thead>
                    <tr>
                        ${headers.map(header => `<th>${header}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
        `;
        
        data.forEach((row, index) => {
            const formattedRow = rowFormatter(row, index);
            html += `
                <tr>
                    ${formattedRow.map(cell => `<td>${cell}</td>`).join('')}
                </tr>
            `;
        });
        
        html += `
                </tbody>
            </table>
        `;
        
        container.innerHTML = html;
    },
    
    /**
     * Verifica se o Plotly está disponível, se não, carrega-o
     */
    ensurePlotlyLoaded: function(callback) {
        if (typeof Plotly !== 'undefined') {
            callback();
            return;
        }
        
        console.log('Carregando biblioteca Plotly...');
        
        const script = document.createElement('script');
        script.src = 'https://cdn.plot.ly/plotly-2.29.1.min.js';
        script.onload = function() {
            console.log('Plotly carregado com sucesso');
            callback();
        };
        
        script.onerror = function() {
            console.error('Erro ao carregar Plotly');
        };
        
        document.head.appendChild(script);
    },
    
    /**
     * Gera um ID único para elementos
     */
    generateUniqueId: function(prefix = 'element') {
        return `${prefix}-${Math.random().toString(36).substr(2, 9)}`;
    },
    
    /**
     * Formata um número para exibição com precisão especificada
     */
    formatNumber: function(value, precision = 4) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        
        return value.toFixed(precision);
    },
    
    /**
     * Formatadores para diferentes tipos de valores
     */
    formatters: {
        // Formata um impacto como porcentagem
        impactPercent: function(impact) {
            if (impact === null || impact === undefined || isNaN(impact)) {
                return 'N/A';
            }
            
            return (impact * 100).toFixed(2) + '%';
        },
        
        // Adiciona ícone para indicar presença em subconjunto
        subset: function(inSubset) {
            return inSubset ? '✓' : '—';
        }
    }
};