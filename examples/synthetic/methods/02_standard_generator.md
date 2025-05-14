# Gerando Dados Sintéticos com o StandardGenerator

Este exemplo demonstra o uso da classe `StandardGenerator`, que oferece diversos métodos estatísticos para geração de dados sintéticos.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from deepbridge.synthetic.standard_generator import StandardGenerator
from sklearn.preprocessing import StandardScaler

# Aviso: O conjunto de dados Boston Housing foi removido do scikit-learn por conter linguagem ofensiva
# Esta é apenas uma demonstração. Em projetos reais, prefira outro conjunto de dados
# Carregando o conjunto de dados com um aviso ignorado
import warnings
warnings.filterwarnings("ignore")
try:
    boston = load_boston()
    data = pd.DataFrame(
        data=boston['data'],
        columns=boston['feature_names']
    )
    data['PRICE'] = boston['target']
except:
    # Caso o dataset não esteja disponível, criar dados de exemplo
    np.random.seed(42)
    n_samples = 500
    data = pd.DataFrame({
        'CRIM': np.random.exponential(0.5, n_samples),
        'ZN': np.random.normal(10, 10, n_samples),
        'INDUS': np.random.normal(11, 7, n_samples),
        'CHAS': np.random.binomial(1, 0.07, n_samples),
        'NOX': np.random.normal(0.55, 0.15, n_samples),
        'RM': np.random.normal(6.3, 0.7, n_samples),
        'AGE': np.random.normal(70, 30, n_samples),
        'PRICE': np.random.normal(22, 9, n_samples)
    })

print(f"Conjunto de dados original: {data.shape}")
print(data.head())

# Demonstrando os 4 métodos disponíveis no StandardGenerator
methods = ['gaussian', 'gmm', 'kde', 'bootstrap']
synthetic_datasets = {}

for method in methods:
    # Criar e configurar o gerador
    generator = StandardGenerator(
        random_state=42,
        verbose=True,
        preserve_dtypes=True,
        # Parâmetros específicos do método
        method=method,
        n_components=5 if method == 'gmm' else None,
        preserve_correlations=True,
        outlier_rate=0.01  # 1% de outliers
    )
    
    # Identificar colunas categóricas (no Boston dataset, apenas CHAS é binária)
    categorical_columns = ['CHAS'] if 'CHAS' in data.columns else []
    
    # Ajustar o gerador
    generator.fit(
        data=data,
        target_column='PRICE',
        categorical_columns=categorical_columns
    )
    
    # Gerar dados sintéticos
    synthetic_data = generator.generate(
        num_samples=200,
        noise_level=0.05  # 5% de ruído
    )
    
    print(f"\nDados sintéticos ({method}) gerados: {synthetic_data.shape}")
    print(synthetic_data.head())
    
    # Armazenar para comparação
    synthetic_datasets[method] = synthetic_data

# Comparando os diferentes métodos
# Vamos escolher 2 variáveis para visualizar
features_to_plot = ['RM', 'PRICE']  # Rooms e Price são importantes

# Criar plots de dispersão para comparar as relações entre as variáveis
plt.figure(figsize=(15, 10))

# Plot original
plt.subplot(2, 3, 1)
plt.scatter(data[features_to_plot[0]], data[features_to_plot[1]], alpha=0.5)
plt.title('Dados Originais')
plt.xlabel(features_to_plot[0])
plt.ylabel(features_to_plot[1])

# Plots para cada método
for i, method in enumerate(methods):
    plt.subplot(2, 3, i+2)
    plt.scatter(
        synthetic_datasets[method][features_to_plot[0]], 
        synthetic_datasets[method][features_to_plot[1]], 
        alpha=0.5
    )
    plt.title(f'Método: {method}')
    plt.xlabel(features_to_plot[0])
    plt.ylabel(features_to_plot[1])

plt.tight_layout()
plt.show()

# Comparar as distribuições das variáveis individuais
fig, axes = plt.subplots(len(features_to_plot), len(methods) + 1, figsize=(15, 8))

for i, feature in enumerate(features_to_plot):
    # Plot para dados originais
    axes[i, 0].hist(data[feature], bins=20, alpha=0.7)
    axes[i, 0].set_title(f'Original: {feature}')
    
    # Plots para cada método
    for j, method in enumerate(methods):
        axes[i, j+1].hist(synthetic_datasets[method][feature], bins=20, alpha=0.7)
        axes[i, j+1].set_title(f'{method}: {feature}')

plt.tight_layout()
plt.show()

# Avaliando a preservação de correlações
print("\nCorrelações entre RM e PRICE:")
print(f"Original: {data[features_to_plot].corr().iloc[0, 1]:.4f}")

for method in methods:
    synth_data = synthetic_datasets[method]
    corr = synth_data[features_to_plot].corr().iloc[0, 1]
    print(f"{method}: {corr:.4f}")

# Demonstrando a configuração do parâmetro 'outlier_rate'
# Vamos criar dados com diferentes taxas de outliers
outlier_rates = [0.0, 0.05, 0.1]
outlier_datasets = {}

for rate in outlier_rates:
    generator = StandardGenerator(
        random_state=42,
        method='gaussian',
        preserve_correlations=True,
        outlier_rate=rate
    )
    
    generator.fit(data=data, target_column='PRICE')
    
    synth_data = generator.generate(num_samples=300)
    outlier_datasets[rate] = synth_data

# Visualizar o efeito de diferentes taxas de outliers
plt.figure(figsize=(15, 5))

for i, rate in enumerate(outlier_rates):
    plt.subplot(1, 3, i+1)
    plt.scatter(
        outlier_datasets[rate]['RM'], 
        outlier_datasets[rate]['PRICE'], 
        alpha=0.5
    )
    plt.title(f'Outlier Rate: {rate*100}%')
    plt.xlabel('RM')
    plt.ylabel('PRICE')

plt.tight_layout()
plt.show()
```

## Métodos Disponíveis no StandardGenerator

### 1. 'gaussian'
Utiliza distribuições gaussianas (normais) para modelar cada variável, com a opção de preservar correlações.
- **Prós**: Simples, eficiente e funciona bem para dados que seguem distribuição normal
- **Contras**: Pode não capturar bem distribuições multimodais ou fortemente assimétricas

### 2. 'gmm' (Gaussian Mixture Model)
Utiliza uma mistura de distribuições gaussianas para modelar o conjunto de dados.
- **Prós**: Pode capturar distribuições multimodais e clusters
- **Contras**: Requer ajuste do número de componentes (n_components)

### 3. 'kde' (Kernel Density Estimation)
Utiliza estimativa de densidade por kernel para modelar a distribuição dos dados.
- **Prós**: Não assume forma paramétrica específica, flexível para distribuições complexas
- **Contras**: Menos eficiente para grandes conjuntos de dados

### 4. 'bootstrap'
Amostragem com substituição do conjunto de dados original, com adição de ruído.
- **Prós**: Preserva muito bem as características dos dados originais
- **Contras**: Pode "vazar" valores reais se a privacidade for uma preocupação

## Parâmetros Importantes

### Parâmetros Globais
- **random_state**: Semente para reprodutibilidade
- **verbose**: Exibir informações detalhadas durante o processo
- **preserve_dtypes**: Manter os tipos de dados originais
- **preserve_correlations**: Preservar correlações entre variáveis

### Parâmetros Específicos
- **method**: O método a ser utilizado ('gaussian', 'gmm', 'kde', 'bootstrap')
- **n_components**: Número de componentes para o método 'gmm'
- **outlier_rate**: Taxa de outliers a serem gerados (0.0 a 0.2)
- **noise_level**: Nível de ruído para adicionar aos dados (0.0 a 1.0)

## Pontos-Chave

- O StandardGenerator oferece uma variedade de métodos estatísticos para geração de dados sintéticos
- Cada método tem suas vantagens e desvantagens, adequadas para diferentes tipos de dados
- A preservação de correlações é um recurso importante para manter as relações entre variáveis
- A geração de outliers pode ser controlada para simular dados mais realistas
- O método 'bootstrap' é excelente para preservar características dos dados, mas pode comprometer a privacidade