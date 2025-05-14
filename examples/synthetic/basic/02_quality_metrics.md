# Métricas de Qualidade de Dados Sintéticos

Este exemplo demonstra como avaliar a qualidade dos dados sintéticos gerados usando diferentes métricas disponíveis no DeepBridge.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from deepbridge.synthetic.synthesizer import Synthesize

# Carregando um conjunto de dados mais complexo (Diabetes dataset)
diabetes = load_diabetes()
data = pd.DataFrame(
    data=np.c_[diabetes['data'], diabetes['target']],
    columns=diabetes['feature_names'] + ['target']
)

print(f"Conjunto de dados original: {data.shape}")
print(data.describe().round(2))

# Gerando dados sintéticos com métricas de qualidade detalhadas
synthetic_data = Synthesize(
    dataset=data,
    method='gaussian_copula',    # Método baseado em cópulas gaussianas
    num_samples=len(data),       # Mesmo número de amostras que o original
    random_state=42,
    return_quality_metrics=True, # Calcular métricas detalhadas
    print_metrics=False,         # Não imprimir métricas automaticamente
    verbose=True
)

# Examinando métricas de qualidade específicas
print("\nMétricas de qualidade por categoria:")

# 1. Métricas estatísticas (similaridade de distribuições)
print("\n1. Métricas Estatísticas:")
stat_metrics = {k: v for k, v in synthetic_data.metrics.items() 
               if k.startswith('stat_') or k in ['wasserstein_distance', 'ks_test_mean']}
for name, value in stat_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {name}: {value:.4f}")

# 2. Métricas de utilidade (usabilidade para modelagem de machine learning)
print("\n2. Métricas de Utilidade:")
utility_metrics = {k: v for k, v in synthetic_data.metrics.items() 
                  if k.startswith('utility_') or k in ['ml_efficacy', 'prediction_similarity']}
for name, value in utility_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {name}: {value:.4f}")

# 3. Métricas de privacidade
print("\n3. Métricas de Privacidade:")
privacy_metrics = {k: v for k, v in synthetic_data.metrics.items() 
                  if k.startswith('privacy_') or k in ['nearest_neighbor_distance']}
for name, value in privacy_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {name}: {value:.4f}")

# Visualizando a comparação das distribuições
# Vamos escolher 4 colunas para visualizar
cols_to_plot = ['age', 'sex', 'bmi', 'target']

# Criar um DataFrame combinado com uma coluna de origem
real_data = data[cols_to_plot].copy()
real_data['source'] = 'Real'

synth_data = synthetic_data.data[cols_to_plot].copy()
synth_data['source'] = 'Synthetic'

combined_data = pd.concat([real_data, synth_data], ignore_index=True)

# Plotar histogramas para comparar distribuições
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    ax = axes[i]
    for source, color in zip(['Real', 'Synthetic'], ['blue', 'orange']):
        subset = combined_data[combined_data['source'] == source]
        ax.hist(subset[col], alpha=0.5, bins=20, label=source, color=color)
    ax.set_title(f'Distribuição de {col}')
    ax.legend()

plt.tight_layout()
plt.show()

# Obtendo a pontuação geral de qualidade
overall_quality = synthetic_data.overall_quality()
print(f"\nPontuação Geral de Qualidade: {overall_quality:.4f} (0-1, quanto maior melhor)")

# Métricas por coluna (apenas algumas como exemplo)
print("\nSimilaridade por coluna (distância de Wasserstein):")
if 'column_similarities' in synthetic_data.metrics:
    column_metrics = synthetic_data.metrics['column_similarities']
    for col, score in sorted(column_metrics.items(), key=lambda x: x[1]):
        print(f"  {col}: {score:.4f}")
```

## Tipos de Métricas de Qualidade

### 1. Métricas Estatísticas
Estas métricas avaliam quão bem os dados sintéticos preservam as distribuições estatísticas dos dados originais.

- **wasserstein_distance**: Mede a distância entre as distribuições multivariadas
- **ks_test_mean**: Resultado médio do teste Kolmogorov-Smirnov para cada coluna
- **stat_correlation_difference**: Diferença nas matrizes de correlação
- **stat_distribution_similarity**: Similaridade geral das distribuições

### 2. Métricas de Utilidade
Estas métricas avaliam se os dados sintéticos podem ser usados para os mesmos propósitos que os dados originais.

- **ml_efficacy**: Quão bem modelos treinados em dados sintéticos funcionam em dados reais
- **prediction_similarity**: Similaridade entre predições de modelos treinados em dados reais vs. sintéticos
- **utility_score**: Pontuação geral de utilidade

### 3. Métricas de Privacidade
Estas métricas avaliam o risco de divulgação de informações sensíveis.

- **nearest_neighbor_distance**: Distância média ao vizinho mais próximo (valores altos indicam melhor privacidade)
- **privacy_score**: Pontuação geral de privacidade (quanto maior, melhor)

## Pontos-Chave

- As métricas de qualidade são organizadas em três categorias principais: estatísticas, utilidade e privacidade
- A pontuação geral de qualidade (`overall_quality()`) combina métricas dessas categorias
- Métricas por coluna mostram quais variáveis foram melhor preservadas nos dados sintéticos
- A visualização das distribuições é uma forma importante de avaliar a qualidade dos dados sintéticos
- Um bom conjunto de dados sintéticos deve equilibrar a preservação estatística com a privacidade