# Uso Básico da Classe Synthesize

Este exemplo demonstra o uso básico da classe `Synthesize` para geração de dados sintéticos.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from deepbridge.synthetic.synthesizer import Synthesize

# Carregando um conjunto de dados de exemplo (Iris dataset)
iris = load_iris()
data = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)

print(f"Conjunto de dados original: {data.shape}")
print(data.head())

# Exemplo básico: gerar dados sintéticos usando o método padrão (gaussian)
# O processo de síntese começa automaticamente durante a inicialização da classe
synthetic_data = Synthesize(
    dataset=data,                 # Dataset original
    method='gaussian',            # Método de geração (padrão: 'gaussian')
    num_samples=150,              # Número de amostras a serem geradas
    random_state=42,              # Semente aleatória para reprodutibilidade
    print_metrics=True,           # Imprimir métricas de qualidade
    return_quality_metrics=True,  # Calcular e retornar métricas de qualidade
    verbose=True                  # Mostrar informações detalhadas durante o processo
)

# Os dados sintéticos estão disponíveis no atributo 'data'
print(f"\nDados sintéticos gerados: {synthetic_data.data.shape}")
print(synthetic_data.data.head())

# Você pode acessar as métricas de qualidade calculadas
if hasattr(synthetic_data, 'metrics'):
    print("\nMétricas de qualidade:")
    for metric_name, metric_value in synthetic_data.metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"  {metric_name}: {metric_value:.4f}")

# Gerando outro conjunto de dados com um método diferente (gmm - Gaussian Mixture Model)
synthetic_data_gmm = Synthesize(
    dataset=data,
    method='gmm',                # Usando Gaussian Mixture Model
    num_samples=150,
    random_state=42,
    print_metrics=True
)

print(f"\nDados sintéticos (GMM) gerados: {synthetic_data_gmm.data.shape}")
print(synthetic_data_gmm.data.head())

# Você também pode gerar novas amostras sem retreinar o modelo
# usando o método resample
new_samples = synthetic_data.resample(num_samples=50)
print(f"\nNovas amostras geradas via resample: {new_samples.shape}")
print(new_samples.head())

# Obter uma pontuação geral de qualidade
overall_quality = synthetic_data.overall_quality()
print(f"\nPontuação geral de qualidade: {overall_quality:.4f} (0-1, quanto maior melhor)")
```

## Pontos-Chave

- A classe `Synthesize` é a interface principal para geração de dados sintéticos
- O processamento começa automaticamente durante a inicialização da classe
- Métodos disponíveis incluem:
  - `'gaussian'`: Método padrão usando distribuição gaussiana
  - `'gmm'`: Gaussian Mixture Model
  - `'kde'`: Kernel Density Estimation
  - `'bootstrap'`: Amostragem com substituição do conjunto original
  - `'gaussian_copula'`: Método baseado em cópulas gaussianas
- A qualidade dos dados sintéticos pode ser avaliada através do atributo `metrics`
- O método `resample()` permite gerar novas amostras sem retreinar o modelo
- A função `overall_quality()` retorna uma pontuação geral de qualidade entre 0 e 1