# Geração de Relatórios de Qualidade

Este exemplo demonstra como gerar relatórios detalhados para avaliar a qualidade dos dados sintéticos.

```python
import pandas as pd
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from deepbridge.synthetic.synthesizer import Synthesize

# Carregar um conjunto de dados de habitação da Califórnia
housing = fetch_california_housing()
data = pd.DataFrame(
    data=np.c_[housing['data'], housing['target']],
    columns=housing['feature_names'] + ['target']
)

print(f"Conjunto de dados da habitação: {data.shape}")
print(data.head())

# Gerando dados sintéticos
synthetic_data = Synthesize(
    dataset=data,
    method='gaussian_copula',      # Método que geralmente produz boa qualidade
    num_samples=1000,              # Gerar 1000 amostras sintéticas
    random_state=42,
    return_quality_metrics=True,   # Calcular métricas detalhadas
    print_metrics=True,            # Imprimir métricas durante a geração
    verbose=True,
    # Não gerar relatório automaticamente ainda
    generate_report=False
)

# Criar diretório para relatórios
output_dir = 'synthetic_reports'
os.makedirs(output_dir, exist_ok=True)

# Método 1: Gerar relatório usando o método save_report
report_path1 = os.path.join(output_dir, 'housing_quality_report.html')
report_file1 = synthetic_data.save_report(
    output_path=report_path1,
    include_data_samples=True,        # Incluir exemplos dos dados
    include_visualizations=True,      # Incluir visualizações
    report_title="Relatório de Qualidade - Dados de Habitação da Califórnia",
    # Parâmetros opcionais adicionais
    max_samples_to_show=10,           # Limitar número de amostras mostradas
    max_columns_to_visualize=6        # Limitar número de colunas para visualizar
)

print(f"\nRelatório gerado em: {report_file1}")
print("Abra este arquivo HTML em um navegador para visualizar o relatório completo")

# Método 2: Gerar relatório durante a criação dos dados sintéticos
synthetic_data2 = Synthesize(
    dataset=data,
    method='gaussian_copula',
    num_samples=1000,
    random_state=42,
    return_quality_metrics=True,
    print_metrics=False,
    # Gerar relatório automaticamente
    generate_report=True,
    report_path=os.path.join(output_dir, 'auto_generated_report.html')
)

print(f"\nRelatório gerado automaticamente em: {os.path.join(output_dir, 'auto_generated_report.html')}")

# Método 3: Comparar dois métodos diferentes em um relatório
# Primeiro, geramos dados usando outro método
synthetic_data_gmm = Synthesize(
    dataset=data,
    method='gmm',                  # Gaussian Mixture Model
    num_samples=1000,
    random_state=42,
    return_quality_metrics=True,
    print_metrics=False,
    verbose=False
)

# Agora podemos gerar um relatório comparativo
report_path3 = os.path.join(output_dir, 'comparison_report.html')

# Precisaríamos implementar esta função (não incluída por padrão no DeepBridge)
# Aqui está uma implementação simples de exemplo
def generate_comparison_report(real_data, synthetic_data1, synthetic_data2, method1, method2, output_path):
    """
    Gera um relatório comparativo entre dois métodos de geração de dados sintéticos.
    
    Args:
        real_data: DataFrame com dados reais
        synthetic_data1: DataFrame com dados sintéticos do método 1
        synthetic_data2: DataFrame com dados sintéticos do método 2
        method1: Nome do método 1
        method2: Nome do método 2
        output_path: Caminho para salvar o relatório HTML
    
    Returns:
        Caminho do relatório gerado
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    
    # Criar um relatório simples em HTML
    html_content = []
    html_content.append("<html><head><title>Comparação de Métodos de Geração Sintética</title>")
    html_content.append("<style>body { font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px; }")
    html_content.append("table { border-collapse: collapse; width: 100%; }")
    html_content.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html_content.append("th { background-color: #f2f2f2; }")
    html_content.append("</style></head><body>")
    
    # Título e descrição
    html_content.append(f"<h1>Comparação de Métodos de Geração Sintética</h1>")
    html_content.append(f"<p>Comparação entre os métodos <b>{method1}</b> e <b>{method2}</b></p>")
    
    # Tabela de estatísticas básicas
    html_content.append("<h2>Estatísticas Comparativas</h2>")
    html_content.append("<table>")
    html_content.append("<tr><th>Métrica</th><th>Dados Reais</th><th>Método: " + method1 + "</th><th>Método: " + method2 + "</th></tr>")
    
    # Adicionar algumas métricas básicas
    html_content.append(f"<tr><td>Número de Linhas</td><td>{len(real_data)}</td><td>{len(synthetic_data1)}</td><td>{len(synthetic_data2)}</td></tr>")
    
    # Adicionar métricas de qualidade se disponíveis
    if hasattr(synthetic_data, 'metrics') and hasattr(synthetic_data_gmm, 'metrics'):
        metrics1 = synthetic_data.metrics
        metrics2 = synthetic_data_gmm.metrics
        
        # Adicionar algumas métricas importantes
        for metric_name in ['wasserstein_distance', 'ml_efficacy', 'privacy_score', 'utility_score']:
            if metric_name in metrics1 and metric_name in metrics2:
                html_content.append(f"<tr><td>{metric_name}</td><td>N/A</td><td>{metrics1[metric_name]:.4f}</td><td>{metrics2[metric_name]:.4f}</td></tr>")
    
    html_content.append("</table>")
    
    # Adicionar visualizações comparativas para algumas colunas
    html_content.append("<h2>Comparações Visuais</h2>")
    
    # Selecionar algumas colunas para visualizar
    vis_columns = list(real_data.columns[:4])
    
    # Gerar e salvar visualizações
    for i, col in enumerate(vis_columns):
        plt.figure(figsize=(10, 6))
        
        # Plotar histogramas
        plt.hist(real_data[col], alpha=0.5, bins=20, label='Real', color='blue')
        plt.hist(synthetic_data1[col], alpha=0.5, bins=20, label=f'Sintético ({method1})', color='orange')
        plt.hist(synthetic_data2[col], alpha=0.5, bins=20, label=f'Sintético ({method2})', color='green')
        
        plt.title(f'Distribuição de {col}')
        plt.legend()
        
        # Salvar visualização como imagem
        img_path = os.path.join(os.path.dirname(output_path), f'comparison_{col}.png')
        plt.savefig(img_path)
        plt.close()
        
        # Adicionar imagem ao relatório
        html_content.append(f"<h3>Comparação para {col}</h3>")
        html_content.append(f"<img src='comparison_{col}.png' style='max-width:100%;'>")
    
    # Finalizar HTML
    html_content.append("</body></html>")
    
    # Salvar o relatório HTML
    with open(output_path, 'w') as f:
        f.write("\n".join(html_content))
    
    return output_path

# Gerar relatório comparativo
comparison_report = generate_comparison_report(
    real_data=data,
    synthetic_data1=synthetic_data.data,
    synthetic_data2=synthetic_data_gmm.data,
    method1='gaussian_copula',
    method2='gmm',
    output_path=report_path3
)

print(f"\nRelatório comparativo gerado em: {comparison_report}")
```

## Conteúdo do Relatório de Qualidade

Os relatórios de qualidade gerados pelo DeepBridge incluem:

### 1. Resumo Geral
- Informações sobre o conjunto de dados (tamanho, colunas, etc.)
- Método de geração utilizado e seus parâmetros
- Pontuação geral de qualidade

### 2. Métricas de Qualidade Detalhadas
- Estatísticas por coluna
- Similaridade de distribuições
- Eficácia para machine learning
- Métricas de privacidade

### 3. Visualizações
- Histogramas comparativos
- Gráficos de dispersão para correlações
- Comparações de distribuições

### 4. Amostras de Dados
- Exemplos dos dados reais
- Exemplos dos dados sintéticos gerados

## Pontos-Chave

- Relatórios HTML permitem uma análise interativa e visual da qualidade dos dados
- É possível gerar relatórios durante a criação dos dados ou posteriormente
- Comparar diferentes métodos ajuda a escolher a melhor abordagem para seu caso de uso
- Os relatórios facilitam a identificação de problemas específicos nas distribuições
- A documentação visual é importante para comunicar a qualidade dos dados sintéticos com stakeholders