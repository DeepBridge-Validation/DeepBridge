# Configuração Básica de um Experimento

Este exemplo demonstra como configurar e inicializar um experimento usando a classe `Experiment` do módulo `core.experiment`.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Importar classes do DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# 1. Preparar dados e modelo
# --------------------------
# Carregar um conjunto de dados para classificação binária
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar um modelo de classificação
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Criar um DBDataset
# ---------------------
# Combinar os dados em DataFrames para treino e teste
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Criar o DBDataset com os dados e o modelo
dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=model,
    dataset_name="Breast Cancer"  # Nome para facilitar identificação
)

print(f"DBDataset criado com {len(dataset.features)} features")

# 3. Configurar e inicializar o Experimento
# -----------------------------------------
# Configuração básica de um experimento para classificação binária
experiment = Experiment(
    dataset=dataset,                     # O DBDataset contendo dados e modelo
    experiment_type="binary_classification",  # Tipo do experimento (classificação binária)
    random_state=42,                     # Semente aleatória para reprodutibilidade
    # Lista de testes a serem executados (executados apenas quando run_tests() for chamado)
    tests=["robustness", "uncertainty"],
    # Configuração opcional para controlar comportamento
    config={
        'verbose': True,                 # Mostrar informações detalhadas durante a execução
    }
)

print("\nExperimento inicializado")

# 4. Acessar informações iniciais do experimento
# ---------------------------------------------
# Obtém resultados iniciais (métricas básicas)
initial_results = experiment.initial_results

# Mostrar métricas do modelo principal
print("\nMétricas iniciais do modelo principal:")
if 'models' in initial_results and 'primary_model' in initial_results['models']:
    metrics = initial_results['models']['primary_model']['metrics']
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# 5. Verificar modelos alternativos criados
# ----------------------------------------
if 'models' in initial_results:
    alt_models = [name for name in initial_results['models'].keys() 
                 if name != 'primary_model']
    print(f"\nModelos alternativos criados: {len(alt_models)}")
    for model_name in alt_models:
        model_metrics = initial_results['models'][model_name]['metrics']
        print(f"  {model_name}: accuracy={model_metrics.get('accuracy', 'N/A')}")

# 6. Acessar componentes do experimento
# ------------------------------------
# Dados de treino e teste
print(f"\nDados de treino: {experiment.X_train.shape}")
print(f"Dados de teste: {experiment.X_test.shape}")

# Acessar o modelo principal
if experiment.model is not None:
    print(f"\nModelo principal: {type(experiment.model).__name__}")

# Ver quais testes estão configurados
print(f"\nTestes configurados: {experiment.tests}")

# 7. Verificar informações básicas sobre o experimento
# --------------------------------------------------
experiment_info = experiment.get_comprehensive_results()
print("\nInformações do experimento:")
for section, data in experiment_info.items():
    if isinstance(data, dict):
        print(f"  {section}:")
        for key, value in data.items():
            print(f"    {key}: {value}")
    else:
        print(f"  {section}: {data}")

# 8. Importância das features (se disponível)
# -----------------------------------------
try:
    feature_importance = experiment.get_feature_importance()
    print("\nImportância das features (top 5):")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
except ValueError as e:
    print(f"\nImportância das features não disponível: {str(e)}")
```

## Configurações do Experimento

Ao inicializar um `Experiment`, você pode configurar diversos parâmetros:

### Parâmetros Obrigatórios
- **dataset**: Um objeto `DBDataset` contendo os dados e o modelo a ser avaliado
- **experiment_type**: Tipo de experimento ("binary_classification", "regression", "forecasting")

### Parâmetros Opcionais
- **test_size**: Proporção dos dados para teste (padrão: 0.2)
- **random_state**: Semente aleatória para reprodutibilidade (padrão: 42)
- **config**: Dicionário de configuração com opções adicionais
- **auto_fit**: Se deve ajustar automaticamente um modelo (se não for fornecido)
- **tests**: Lista de testes a preparar para o modelo
- **feature_subset**: Lista de features para focar durante os testes

## Tipos de Experimentos

O DeepBridge suporta diferentes tipos de experimentos:

- **binary_classification**: Para modelos de classificação binária
- **regression**: Para modelos de regressão
- **forecasting**: Para modelos de previsão de séries temporais

## Tipos de Testes Disponíveis

Você pode configurar um ou mais dos seguintes testes:

- **robustness**: Avalia a robustez do modelo a perturbações nas features
- **uncertainty**: Avalia a calibração e a qualidade das estimativas de incerteza do modelo
- **resilience**: Avalia a resiliência do modelo a mudanças na distribuição dos dados
- **hyperparameters**: Avalia a importância dos hiperparâmetros no desempenho do modelo

## Pontos-Chave

- A classe `Experiment` serve como ponto central para coordenar diferentes componentes de validação
- A inicialização do experimento **não** executa os testes completos automaticamente
- Métricas básicas são calculadas durante a inicialização e disponíveis em `initial_results`
- Modelos alternativos são criados automaticamente para comparação
- Os testes completos são executados explicitamente chamando `experiment.run_tests()`
- O experimento fornece acesso fácil aos dados de treino/teste e aos modelos