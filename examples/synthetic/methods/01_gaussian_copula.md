# Geração de Dados Sintéticos com Gaussian Copula

Este exemplo demonstra como usar o gerador de dados sintéticos baseado em Cópulas Gaussianas, que é especialmente eficaz para preservar correlações entre variáveis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from deepbridge.synthetic.methods.gaussian_copula import GaussianCopulaGenerator

# Carregando o conjunto de dados de vinhos
wine = load_wine()
data = pd.DataFrame(
    data=wine['data'],
    columns=wine['feature_names']
)
# Adicionar a coluna target
data['target'] = wine['target']

print(f"Conjunto de dados original: {data.shape}")
print(data.head())

# Verificar correlações no conjunto original
correlation_matrix = data.corr()
print("\nMatriz de correlação (primeiras 5 colunas):")
print(correlation_matrix.iloc[:5, :5].round(2))

# Criar e configurar o gerador de Gaussian Copula
copula_generator = GaussianCopulaGenerator(
    random_state=42,
    verbose=True,
    preserve_dtypes=True,
    preserve_constraints=True,
    # Parâmetros específicos do Gaussian Copula
    fit_sample_size=1000,            # Número máximo de amostras para ajuste
    use_dask=False,                  # Para datasets pequenos, Dask não é necessário
    post_process_method='enhanced'   # Usar pós-processamento aprimorado
)

# Ajustar o gerador aos dados
copula_generator.fit(
    data=data,
    target_column='target',          # Especificar a coluna alvo
    # Estes parâmetros são opcionais - o método os inferirá automaticamente
    # categorical_columns=['target'],
    # numerical_columns=wine['feature_names']
)

# Gerar dados sintéticos 
synthetic_data = copula_generator.generate(
    num_samples=200,                 # Número de amostras a gerar
    chunk_size=100,                  # Tamanho dos chunks para processamento
    # Parâmetros adicionais de geração
    dynamic_chunk_sizing=True,       # Ajustar tamanho do chunk automaticamente
    noise_level=0.05                 # Adicionar um pouco de ruído (0-1)
)

print(f"\nDados sintéticos gerados: {synthetic_data.shape}")
print(synthetic_data.head())

# Verificar correlações nos dados sintéticos
synth_correlation = synthetic_data.corr()
print("\nMatriz de correlação sintética (primeiras 5 colunas):")
print(synth_correlation.iloc[:5, :5].round(2))

# Comparar as distribuições para algumas variáveis
features_to_plot = ['alcohol', 'malic_acid', 'ash', 'target']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    # Plotar histogramas sobrepostos
    axes[i].hist(data[feature], alpha=0.5, bins=20, label='Original', color='blue')
    axes[i].hist(synthetic_data[feature], alpha=0.5, bins=20, label='Sintético', color='orange')
    axes[i].set_title(f'Distribuição de {feature}')
    axes[i].legend()

plt.tight_layout()
plt.show()

# Visualizar a preservação de correlações usando um heatmap da diferença
plt.figure(figsize=(10, 8))
correlation_diff = np.abs(correlation_matrix - synth_correlation)
plt.imshow(correlation_diff, cmap='Blues_r', vmin=0, vmax=0.3)
plt.colorbar(label='Diferença absoluta')
plt.title('Diferença nas correlações (menor = melhor)')
plt.xticks(range(len(data.columns)), data.columns, rotation=90)
plt.yticks(range(len(data.columns)), data.columns)
plt.tight_layout()
plt.show()

# Avaliar a qualidade dos dados sintéticos
quality_metrics = copula_generator.evaluate_quality(
    real_data=data,
    synthetic_data=synthetic_data
)

print("\nMétricas de Qualidade:")
for metric, value in quality_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.4f}")

# Salvar e carregar o modelo de Gaussian Copula
# Salvar o modelo
model_path = 'wine_copula_model.pkl'
copula_generator.save_model(model_path)
print(f"\nModelo salvo em: {model_path}")

# Carregar o modelo salvo
loaded_generator = GaussianCopulaGenerator(random_state=42)
loaded_generator.load_model(model_path)
print("Modelo carregado com sucesso")

# Gerar novos dados com o modelo carregado
new_synthetic_data = loaded_generator.generate(num_samples=100)
print(f"Novos dados gerados com o modelo carregado: {new_synthetic_data.shape}")
```

## Como Funciona o Gaussian Copula

A abordagem de Cópulas Gaussianas é um método poderoso para modelar dependências multivariadas. Ela funciona da seguinte maneira:

1. **Modelagem Marginal**: Ajusta distribuições marginais para cada variável individualmente
2. **Modelagem de Dependência**: Usa uma cópula gaussiana para modelar a estrutura de dependência entre as variáveis
3. **Geração de Dados**: Gera amostras da cópula e as transforma de volta ao espaço original das variáveis

### Vantagens do Gaussian Copula:

- **Preservação de Correlações**: Excelente em preservar estruturas de dependência entre variáveis
- **Flexibilidade**: Funciona bem com diferentes tipos de dados (contínuos e categóricos)
- **Escalabilidade**: Pode ser aplicado a conjuntos de dados grandes com otimizações de memória

## Parâmetros Importantes

### Para o Ajuste (fit):
- **random_state**: Garante reprodutibilidade
- **preserve_dtypes**: Mantém os tipos de dados originais
- **preserve_constraints**: Garante que as restrições (min/max, valores categóricos válidos) sejam respeitadas
- **fit_sample_size**: Número máximo de amostras usadas para ajustar o modelo
- **use_dask**: Ativar processamento distribuído com Dask para grandes conjuntos de dados

### Para a Geração (generate):
- **num_samples**: Número de amostras sintéticas a gerar
- **chunk_size**: Tamanho dos blocos para processamento em memória eficiente
- **dynamic_chunk_sizing**: Ajustar automaticamente o tamanho dos chunks com base na memória disponível
- **noise_level**: Adicionar ruído aleatório (0-1) para aumentar a diversidade

## Pontos-Chave

- O método de Cópulas Gaussianas é especialmente eficaz para preservar correlações entre variáveis
- O pós-processamento aprimorado (`enhanced`) ajuda a melhorar a qualidade dos dados sintéticos
- É possível salvar e carregar modelos para reutilização, o que é útil para gerar dados consistentes ao longo do tempo
- A avaliação de qualidade permite verificar a fidelidade dos dados sintéticos em relação aos originais
- O processamento em chunks e o suporte a Dask possibilitam a aplicação em conjuntos de dados muito grandes