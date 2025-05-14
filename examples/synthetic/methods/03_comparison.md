# Comparação entre Diferentes Métodos de Geração de Dados Sintéticos

Este exemplo demonstra como comparar diferentes métodos de geração de dados sintéticos disponíveis no DeepBridge, para escolher o mais adequado para seu caso de uso.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from deepbridge.synthetic.synthesizer import Synthesize

# Criar um conjunto de dados para classificação binária
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=3,
    random_state=42
)

# Adicionar algumas correlações mais fortes entre features
X[:, 0] = X[:, 1] + np.random.normal(0, 0.5, X.shape[0])
X[:, 2] = X[:, 3] * 0.8 + np.random.normal(0, 0.5, X.shape[0])

# Criar um DataFrame
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

print(f"Conjunto de dados original: {data.shape}")
print(data.head())

# Dividir em treino e teste para avaliação posterior
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1),
    data['target'],
    test_size=0.2,
    random_state=42
)

# Treinar um modelo no conjunto original para referência
original_model = RandomForestClassifier(random_state=42)
original_model.fit(X_train, y_train)
y_pred_original = original_model.predict(X_test)
original_accuracy = accuracy_score(y_test, y_pred_original)
original_auc = roc_auc_score(y_test, original_model.predict_proba(X_test)[:, 1])

print(f"\nDesempenho do modelo com dados originais:")
print(f"Acurácia: {original_accuracy:.4f}")
print(f"AUC-ROC: {original_auc:.4f}")

# Métodos a serem comparados
methods = ['gaussian', 'gmm', 'kde', 'gaussian_copula']

# Estruturas para armazenar resultados
synthetic_datasets = {}
quality_metrics = {}
ml_metrics = {}

# Gerar datasets sintéticos com cada método
for method in methods:
    print(f"\n\n===== Gerando dados sintéticos com método: {method} =====")
    
    synthetic = Synthesize(
        dataset=data,
        method=method,
        num_samples=1000,
        random_state=42,
        return_quality_metrics=True,
        print_metrics=False,
        verbose=True
    )
    
    # Armazenar o dataset sintético
    synthetic_datasets[method] = synthetic.data
    
    # Armazenar métricas de qualidade
    if hasattr(synthetic, 'metrics'):
        quality_metrics[method] = synthetic.metrics
    
    # Treinar modelo no conjunto sintético
    synth_data = synthetic.data
    
    # Dividir atributos e target
    X_synth = synth_data.drop('target', axis=1)
    y_synth = synth_data['target']
    
    # Treinar modelo no dataset sintético
    synth_model = RandomForestClassifier(random_state=42)
    synth_model.fit(X_synth, y_synth)
    
    # Avaliar no conjunto de teste original
    y_pred_synth = synth_model.predict(X_test)
    synth_accuracy = accuracy_score(y_test, y_pred_synth)
    synth_auc = roc_auc_score(y_test, synth_model.predict_proba(X_test)[:, 1])
    
    # Armazenar métricas de ML
    ml_metrics[method] = {
        'accuracy': synth_accuracy,
        'auc': synth_auc,
        'accuracy_ratio': synth_accuracy / original_accuracy,
        'auc_ratio': synth_auc / original_auc
    }
    
    print(f"Modelo treinado em dados sintéticos ({method}):")
    print(f"Acurácia: {synth_accuracy:.4f} ({100 * synth_accuracy / original_accuracy:.1f}% do original)")
    print(f"AUC-ROC: {synth_auc:.4f} ({100 * synth_auc / original_auc:.1f}% do original)")

# 1. Comparar métricas de qualidade
# Selecionar algumas métricas importantes
key_metrics = [
    'wasserstein_distance',
    'stat_correlation_difference',
    'utility_score',
    'stat_distribution_similarity',
    'privacy_score'
]

# Criar DataFrame de métricas
metrics_df = pd.DataFrame(index=methods)

for metric in key_metrics:
    values = []
    for method in methods:
        metrics = quality_metrics.get(method, {})
        value = metrics.get(metric, np.nan)
        
        # Garantir que o valor seja um número
        if isinstance(value, (int, float)):
            values.append(value)
        else:
            values.append(np.nan)
    
    metrics_df[metric] = values

print("\n\n===== Comparação de Métricas de Qualidade =====")
print(metrics_df.round(4))

# 2. Comparar preservação de correlações
original_corr = data.corr()

plt.figure(figsize=(15, 12))

# Original correlation matrix
plt.subplot(2, 3, 1)
sns.heatmap(original_corr, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
plt.title('Correlações Originais')

for i, method in enumerate(methods):
    synth_data = synthetic_datasets[method]
    synth_corr = synth_data.corr()
    
    plt.subplot(2, 3, i+2)
    sns.heatmap(synth_corr, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
    plt.title(f'Correlações: {method}')

plt.tight_layout()
plt.show()

# 3. Comparar distribuições de features importantes
# Escolher algumas features para visualizar
features_to_plot = ['feature_0', 'feature_1', 'feature_8', 'target']

plt.figure(figsize=(15, 12))

for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 2, i+1)
    
    # Plot da distribuição original
    plt.hist(data[feature], alpha=0.5, bins=20, label='Original', color='black')
    
    # Plot das distribuições sintéticas
    for method, color in zip(methods, ['blue', 'green', 'red', 'purple']):
        plt.hist(synthetic_datasets[method][feature], alpha=0.3, bins=20, label=method, color=color)
    
    plt.title(f'Distribuição de {feature}')
    plt.legend()

plt.tight_layout()
plt.show()

# 4. Comparar desempenho de ML
ml_df = pd.DataFrame({
    'accuracy': [ml_metrics[method]['accuracy'] for method in methods],
    'auc': [ml_metrics[method]['auc'] for method in methods],
    'accuracy_ratio': [ml_metrics[method]['accuracy_ratio'] for method in methods],
    'auc_ratio': [ml_metrics[method]['auc_ratio'] for method in methods]
}, index=methods)

print("\n\n===== Desempenho de Machine Learning =====")
print(ml_df.round(4))

# Visualizar performance de ML
plt.figure(figsize=(12, 6))

# Plot de acurácia
plt.subplot(1, 2, 1)
bars = plt.bar(methods, ml_df['accuracy_ratio'] * 100)
plt.axhline(y=100, color='r', linestyle='-', alpha=0.3)
plt.title('Acurácia Relativa (%)')
plt.ylabel('% do desempenho original')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom')

# Plot de AUC
plt.subplot(1, 2, 2)
bars = plt.bar(methods, ml_df['auc_ratio'] * 100)
plt.axhline(y=100, color='r', linestyle='-', alpha=0.3)
plt.title('AUC-ROC Relativo (%)')
plt.ylabel('% do desempenho original')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 5. Resumo comparativo - pontuação geral por método
# Criar um sistema de pontuação ponderado
weights = {
    'wasserstein_distance': -1.0,  # Menor é melhor, por isso o sinal negativo
    'stat_correlation_difference': -1.0,
    'utility_score': 2.0,
    'stat_distribution_similarity': 1.5,
    'privacy_score': 1.0,
    'accuracy_ratio': 2.0,
    'auc_ratio': 2.0
}

# Calcular pontuação para cada método
scores = {}
for method in methods:
    score = 0
    
    # Adicionar pontuação de métricas de qualidade
    for metric, weight in weights.items():
        if metric in metrics_df.columns and not pd.isna(metrics_df.loc[method, metric]):
            score += metrics_df.loc[method, metric] * weight
        elif metric in ml_df.columns and not pd.isna(ml_df.loc[method, metric]):
            score += ml_df.loc[method, metric] * weight
    
    scores[method] = score

# Ordenar métodos por pontuação
sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("\n\n===== Classificação Geral dos Métodos =====")
for method, score in sorted_methods:
    print(f"{method}: {score:.4f} pontos")

print("\nMelhor método geral:", sorted_methods[0][0])

# Recomendação baseada no caso de uso
print("\n===== Recomendações por Caso de Uso =====")
print("Para preservação de correlações:", 
      methods[np.argmin([metrics_df.loc[m, 'stat_correlation_difference'] for m in methods])])
print("Para utilidade em machine learning:", 
      methods[np.argmax([ml_df.loc[m, 'accuracy_ratio'] for m in methods])])
print("Para privacidade:", 
      methods[np.argmax([metrics_df.loc[m, 'privacy_score'] for m in methods])])
print("Para similaridade de distribuição:", 
      methods[np.argmax([metrics_df.loc[m, 'stat_distribution_similarity'] for m in methods])])
```

## Critérios de Comparação

### 1. Métricas de Qualidade Sintética
- **wasserstein_distance**: Mede a distância entre distribuições (menor é melhor)
- **stat_correlation_difference**: Diferença nas matrizes de correlação (menor é melhor)
- **utility_score**: Pontuação geral de utilidade (maior é melhor)
- **stat_distribution_similarity**: Similaridade das distribuições (maior é melhor)
- **privacy_score**: Pontuação de privacidade (maior é melhor)

### 2. Desempenho de Machine Learning
- **accuracy_ratio**: Razão entre a acurácia do modelo treinado com dados sintéticos e o original
- **auc_ratio**: Razão entre o AUC-ROC do modelo treinado com dados sintéticos e o original

### 3. Análise Visual
- **Preservação de correlações**: Comparação de matrizes de correlação
- **Distribuições de features**: Comparação das distribuições de características importantes

## Características dos Métodos

### 1. Gaussian
- **Pontos fortes**: Simples, rápido, bom para dados normalmente distribuídos
- **Pontos fracos**: Pode não capturar bem distribuições complexas

### 2. GMM (Gaussian Mixture Model)
- **Pontos fortes**: Bom para dados multimodais, pode capturar clusters
- **Pontos fracos**: Mais complexo, requer ajuste do número de componentes

### 3. KDE (Kernel Density Estimation)
- **Pontos fortes**: Não-paramétrico, flexível para diferentes distribuições
- **Pontos fracos**: Computacionalmente mais intensivo, pode ter problemas com alta dimensionalidade

### 4. Gaussian Copula
- **Pontos fortes**: Excelente para preservar correlações e dependências entre variáveis
- **Pontos fracos**: Mais complexo, pode ser computacionalmente intensivo para grandes conjuntos de dados

## Pontos-Chave

- Diferentes métodos têm diferentes vantagens e desvantagens
- A escolha do melhor método depende do caso de uso específico:
  - Para preservação de correlações: Gaussian Copula geralmente é melhor
  - Para machine learning: O desempenho varia, mas métodos que preservam correlações tendem a ser melhores
  - Para privacidade: Métodos como GMM podem oferecer melhor privacidade
- Uma avaliação abrangente deve considerar múltiplos critérios
- Os métodos podem ser ajustados com diferentes parâmetros para otimizar aspectos específicos
- A comparação visual é importante para validar a qualidade dos dados sintéticos