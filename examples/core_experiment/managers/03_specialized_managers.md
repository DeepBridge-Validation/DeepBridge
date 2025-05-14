# Usando Gerenciadores Especializados

O DeepBridge oferece vários gerenciadores especializados para diferentes tipos de testes de modelos. Cada gerenciador se concentra em um aspecto específico de avaliação, como robustez, resiliência, incerteza e hiperparâmetros. Vamos explorar como usar cada um deles.

## 1. Gerenciador de Robustez (RobustnessManager)

O `RobustnessManager` avalia a estabilidade do modelo sob perturbações nos dados de entrada.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.managers.robustness_manager import RobustnessManager

# Criar dados de exemplo
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})

# Preparar dataset e modelo
model = RandomForestClassifier(random_state=42)
dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

# Treinar o modelo
X = dataset.X
y = dataset.target
model.fit(X, y)

# Criar um gerenciador de robustez
robustness_manager = RobustnessManager(dataset=dataset)

# Executar testes de robustez com configuração rápida
results = robustness_manager.run_tests(config_name='quick')

print("Resultados dos testes de robustez:")
for method, result in results.items():
    print(f"Método {method}: Robustez geral: {result.get('general_robustness', 'N/A')}")
    print(f"  Características mais sensíveis: {result.get('most_sensitive_features', [])}")
```

### Comparando a Robustez de Vários Modelos

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.managers.robustness_manager import RobustnessManager

# Preparar dados
df = pd.DataFrame({
    'feature1': np.random.rand(200),
    'feature2': np.random.rand(200),
    'feature3': np.random.rand(200),
    'target': np.random.randint(0, 2, 200)
})

# Criar e treinar modelos alternativos
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
lr_model = LogisticRegression(random_state=42)

X = df.drop('target', axis=1)
y = df['target']

rf_model.fit(X, y)
gb_model.fit(X, y)
lr_model.fit(X, y)

# Criar dataset com modelo principal
dataset = DBDataset(
    data=df,
    target_column='target',
    model=rf_model
)

# Configurar modelos alternativos
alternative_models = {
    'gradient_boost': gb_model,
    'logistic_regression': lr_model
}

# Criar gerenciador e comparar modelos
robustness_manager = RobustnessManager(
    dataset=dataset,
    alternative_models=alternative_models
)

# Comparar a robustez dos modelos
comparison_results = robustness_manager.compare_models(config_name='quick')

print("Comparação de robustez entre modelos:")
for model_name, results in comparison_results.items():
    print(f"\nModelo: {model_name}")
    for method, scores in results.items():
        if method != 'raw_results':
            print(f"  Método {method}: {scores.get('general_robustness', 'N/A')}")
```

## 2. Gerenciador de Resiliência (ResilienceManager)

O `ResilienceManager` avalia o desempenho do modelo sob mudanças de distribuição dos dados.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.managers.resilience_manager import ResilienceManager

# Criar dados de exemplo com alguma estrutura
np.random.seed(42)
n_samples = 300
feature1 = np.random.normal(0, 1, n_samples)
feature2 = np.random.normal(0, 1, n_samples)
# Criar target com dependência nas features
target = (0.7 * feature1 + 0.3 * feature2 > 0).astype(int)

df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'target': target
})

# Preparar dataset e modelo
model = RandomForestClassifier(random_state=42)
dataset = DBDataset(
    data=df, 
    target_column='target',
    model=model
)

# Treinar o modelo
X = dataset.X
y = dataset.target
model.fit(X, y)

# Criar gerenciador de resiliência
resilience_manager = ResilienceManager(dataset=dataset)

# Executar testes de resiliência
results = resilience_manager.run_tests(config_name='quick')

print("Resultados dos testes de resiliência:")
for drift_type, metrics in results.items():
    print(f"\nTipo de drift: {drift_type}")
    for intensity, scores in metrics.items():
        print(f"  Intensidade {intensity}: {scores}")
```

### Comparando a Resiliência de Múltiplos Modelos

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.managers.resilience_manager import ResilienceManager

# Preparar dados
np.random.seed(42)
n_samples = 300
feature1 = np.random.normal(0, 1, n_samples)
feature2 = np.random.normal(0, 1, n_samples)
# Target com dependência nas features
target = (0.7 * feature1 + 0.3 * feature2 > 0).astype(int)

df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'target': target
})

# Criar e treinar modelos
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
lr_model = LogisticRegression(random_state=42)

X = df.drop('target', axis=1)
y = df['target']

rf_model.fit(X, y)
gb_model.fit(X, y)
lr_model.fit(X, y)

# Criar dataset com modelo principal
dataset = DBDataset(
    data=df,
    target_column='target',
    model=rf_model
)

# Configurar modelos alternativos
alternative_models = {
    'gradient_boost': gb_model,
    'logistic_regression': lr_model
}

# Criar gerenciador e comparar modelos
resilience_manager = ResilienceManager(
    dataset=dataset,
    alternative_models=alternative_models
)

# Comparar a resiliência dos modelos
comparison_results = resilience_manager.compare_models(config_name='quick')

print("Comparação de resiliência entre modelos:")
for model_name, drift_results in comparison_results.items():
    print(f"\nModelo: {model_name}")
    for drift_type, intensities in drift_results.items():
        print(f"  Drift tipo {drift_type}:")
        for intensity, metrics in intensities.items():
            print(f"    Intensidade {intensity}: {metrics}")
```

## 3. Gerenciador de Incerteza (UncertaintyManager)

O `UncertaintyManager` avalia a capacidade do modelo de expressar incerteza em suas previsões.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.managers.uncertainty_manager import UncertaintyManager

# Criar dados de exemplo
np.random.seed(42)
n_samples = 200
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 1, n_samples)
# Adicionar algum ruído para criar incerteza
noise = np.random.normal(0, 0.5, n_samples)
y = (0.7 * X1 + 0.3 * X2 + noise > 0).astype(int)

df = pd.DataFrame({
    'feature1': X1,
    'feature2': X2,
    'target': y
})

# Preparar dataset e modelo
model = RandomForestClassifier(random_state=42, n_estimators=100, oob_score=True)
dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

# Treinar o modelo
X = dataset.X
y = dataset.target
model.fit(X, y)

# Criar gerenciador de incerteza
uncertainty_manager = UncertaintyManager(dataset=dataset)

# Executar testes de incerteza
results = uncertainty_manager.run_tests(config_name='quick')

print("Resultados dos testes de incerteza:")
for method, test_results in results.items():
    print(f"\nMétodo: {method}")
    for alpha, metrics in test_results.items():
        print(f"  Alpha {alpha}:")
        for metric_name, value in metrics.items():
            if metric_name != 'raw_results':
                print(f"    {metric_name}: {value}")
```

### Comparando a Quantificação de Incerteza Entre Modelos

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.managers.uncertainty_manager import UncertaintyManager

# Preparar dados
np.random.seed(42)
n_samples = 200
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 1, n_samples)
noise = np.random.normal(0, 0.5, n_samples)
y = (0.7 * X1 + 0.3 * X2 + noise > 0).astype(int)

df = pd.DataFrame({
    'feature1': X1,
    'feature2': X2,
    'target': y
})

# Criar e treinar modelos
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, oob_score=True)
gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)

X = df.drop('target', axis=1)
y = df['target']

rf_model.fit(X, y)
gb_model.fit(X, y)

# Criar dataset com modelo principal
dataset = DBDataset(
    data=df,
    target_column='target',
    model=rf_model
)

# Configurar modelos alternativos
alternative_models = {
    'gradient_boost': gb_model
}

# Criar gerenciador e comparar modelos
uncertainty_manager = UncertaintyManager(
    dataset=dataset,
    alternative_models=alternative_models
)

# Comparar a quantificação de incerteza dos modelos
comparison_results = uncertainty_manager.compare_models(config_name='quick')

print("Comparação de quantificação de incerteza entre modelos:")
for model_name, method_results in comparison_results.items():
    print(f"\nModelo: {model_name}")
    for method, alpha_results in method_results.items():
        print(f"  Método: {method}")
        for alpha, metrics in alpha_results.items():
            print(f"    Alpha {alpha}:")
            for metric_name, value in metrics.items():
                if metric_name != 'raw_results':
                    print(f"      {metric_name}: {value}")
```

## 4. Gerenciador de Hiperparâmetros (HyperparameterManager)

O `HyperparameterManager` avalia a importância dos hiperparâmetros no desempenho do modelo.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.managers.hyperparameter_manager import HyperparameterManager

# Criar dados de exemplo
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 5)
y = (X[:, 0] > 0.5).astype(int)

df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(X.shape[1])])
df['target'] = y

# Definir o espaço de hiperparâmetros
hyperparameter_space = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Preparar dataset e modelo
model = RandomForestClassifier(random_state=42)
dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

# Criar gerenciador de hiperparâmetros
hyperparameter_manager = HyperparameterManager(
    dataset=dataset,
    param_grid=hyperparameter_space
)

# Executar testes de importância de hiperparâmetros
results = hyperparameter_manager.run_tests(config_name='quick')

print("Resultados da análise de hiperparâmetros:")
for param_name, importance in results.get('importance_scores', {}).items():
    print(f"Hiperparâmetro: {param_name}, Importância: {importance}")

print("\nMelhores combinações de hiperparâmetros:")
for i, config in enumerate(results.get('top_configs', [])[:3]):
    print(f"Top {i+1}: {config}")
```

### Comparando a Importância de Hiperparâmetros Entre Modelos

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.managers.hyperparameter_manager import HyperparameterManager

# Preparar dados
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 5)
y = (X[:, 0] > 0.5).astype(int)

df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(X.shape[1])])
df['target'] = y

# Definir espaços de hiperparâmetros para cada tipo de modelo
rf_param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

gb_param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10]
}

# Preparar modelos
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

# Treinar os modelos com configurações padrão
X_data = df.drop('target', axis=1)
y_data = df['target']
rf_model.fit(X_data, y_data)
gb_model.fit(X_data, y_data)

# Criar dataset com modelo principal
dataset = DBDataset(
    data=df,
    target_column='target',
    model=rf_model
)

# Configurar modelos alternativos
alternative_models = {
    'gradient_boost': gb_model
}

# Criar gerenciador de hiperparâmetros com ambos os espaços de parâmetros
param_grids = {
    'primary': rf_param_grid,
    'gradient_boost': gb_param_grid
}

hyperparameter_manager = HyperparameterManager(
    dataset=dataset,
    alternative_models=alternative_models,
    param_grids=param_grids
)

# Comparar a importância de hiperparâmetros entre modelos
comparison_results = hyperparameter_manager.compare_models(config_name='quick')

print("Comparação de importância de hiperparâmetros entre modelos:")
for model_name, results in comparison_results.items():
    print(f"\nModelo: {model_name}")
    print("Importância de hiperparâmetros:")
    for param_name, importance in results.get('importance_scores', {}).items():
        print(f"  {param_name}: {importance}")
    
    print("\nMelhores configurações:")
    for i, config in enumerate(results.get('top_configs', [])[:2]):
        print(f"  Top {i+1}: {config}")
```

## 5. Uso Integrado dos Gerenciadores em Experimentos

Os gerenciadores especializados podem ser usados diretamente dentro da classe `Experiment` para uma experiência integrada:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# Preparar dados de exemplo
np.random.seed(42)
n_samples = 300
X = np.random.rand(n_samples, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(X.shape[1])])
df['target'] = y

# Criar modelo e dataset
model = RandomForestClassifier(random_state=42)
model.fit(df.drop('target', axis=1), df['target'])

dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

# Criar modelos alternativos
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

alternative_models = {
    'logistic': LogisticRegression(random_state=42).fit(df.drop('target', axis=1), df['target']),
    'svm': SVC(probability=True, random_state=42).fit(df.drop('target', axis=1), df['target'])
}

# Criar e configurar experimento
experiment = Experiment(
    name="comprehensive_evaluation",
    dataset=dataset,
    alternative_models=alternative_models,
    test_size=0.3,
    random_state=42
)

# Executar todos os tipos de testes
robustness_results = experiment.run_robustness_tests(config_name='quick')
resilience_results = experiment.run_resilience_tests(config_name='quick')
uncertainty_results = experiment.run_uncertainty_tests(config_name='quick')

# Comparar modelos em todos os aspectos
rob_comparison = experiment.compare_models_robustness(config_name='quick')
res_comparison = experiment.compare_models_resilience(config_name='quick')
unc_comparison = experiment.compare_models_uncertainty(config_name='quick')

# Gerar relatório completo
experiment.generate_report(
    output_dir="./experiment_report",
    report_type="comprehensive",
    include_sections=[
        "robustness", 
        "resilience", 
        "uncertainty"
    ]
)

print("Experimento completo executado com sucesso. Relatório gerado em ./experiment_report")
```

Os gerenciadores especializados do DeepBridge fornecem uma forma estruturada e consistente de avaliar modelos em diferentes dimensões de desempenho, permitindo uma análise abrangente da qualidade e confiabilidade de modelos de machine learning.