# Gerenciamento de Modelos com ModelManager

Este exemplo demonstra como usar a classe `ModelManager` do módulo `core.experiment` para gerenciar, criar e otimizar modelos.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Importar componentes do DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.managers.model_manager import ModelManager
from deepbridge.utils.model_registry import ModelType

# 1. Preparar dados
# ---------------
# Carregar dados
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar DataFrames para DBDataset
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Criar DBDataset (sem modelo inicial)
dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    dataset_name="Breast Cancer"
)

# 2. Criar e configurar o ModelManager
# ----------------------------------
# Inicializar o gerenciador de modelos
model_manager = ModelManager(
    dataset=dataset,
    experiment_type="binary_classification",
    verbose=True
)

print("Gerenciador de modelos inicializado")

# 3. Criar modelos alternativos
# ---------------------------
# O método create_alternative_models cria automaticamente vários modelos alternativos
alternative_models = model_manager.create_alternative_models(X_train, y_train)

print(f"\nModelos alternativos criados: {len(alternative_models)}")
for name, model in alternative_models.items():
    print(f"  {name}: {type(model).__name__}")

# 4. Avaliar os modelos alternativos
# --------------------------------
# Avaliar cada modelo no conjunto de teste
results = {}
for name, model in alternative_models.items():
    # Calcular acurácia
    accuracy = model.score(X_test, y_test)
    
    # Calcular AUC se o modelo suportar predict_proba
    auc = None
    if hasattr(model, 'predict_proba'):
        from sklearn.metrics import roc_auc_score
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            pass
    
    results[name] = {'accuracy': accuracy, 'auc': auc}

# Mostrar resultados
print("\nDesempenho dos modelos alternativos:")
for name, metrics in results.items():
    acc = metrics['accuracy']
    auc = metrics['auc']
    auc_str = f", AUC: {auc:.4f}" if auc is not None else ""
    print(f"  {name}: Acurácia: {acc:.4f}{auc_str}")

# 5. Obter o tipo de modelo padrão
# ------------------------------
default_model_type = model_manager.get_default_model_type()
print(f"\nTipo de modelo padrão para classificação binária: {default_model_type}")

# 6. Criar um modelo de destilação
# ------------------------------
print("\nCriando modelo de destilação...")
distillation_model = model_manager.create_distillation_model(
    distillation_method="surrogate",
    student_model_type=ModelType.LOGISTIC_REGRESSION,
    student_params=None,  # Parâmetros padrão
    temperature=1.0,
    alpha=0.5,
    use_probabilities=True,
    n_trials=20,  # Número de trials para otimização
    validation_split=0.2
)

print(f"Modelo de destilação criado: {type(distillation_model).__name__}")

# 7. Treinar o modelo de destilação
# -------------------------------
# Treinar o modelo de destilação com os dados de treino
distillation_model.fit(X_train, y_train)

# Avaliar no conjunto de teste
distillation_accuracy = distillation_model.score(X_test, y_test)
distillation_auc = None
if hasattr(distillation_model, 'predict_proba'):
    distillation_auc = roc_auc_score(y_test, distillation_model.predict_proba(X_test)[:, 1])

print(f"\nDesempenho do modelo de destilação:")
print(f"  Acurácia: {distillation_accuracy:.4f}")
if distillation_auc is not None:
    print(f"  AUC: {distillation_auc:.4f}")

# 8. Comparar com o melhor modelo alternativo
# -----------------------------------------
# Identificar o melhor modelo alternativo
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = alternative_models[best_model_name]
best_model_accuracy = results[best_model_name]['accuracy']
best_model_auc = results[best_model_name]['auc']

print(f"\nComparação com o melhor modelo alternativo ({best_model_name}):")
print(f"  Melhor modelo alternativo - Acurácia: {best_model_accuracy:.4f}, AUC: {best_model_auc:.4f if best_model_auc else 'N/A'}")
print(f"  Modelo de destilação - Acurácia: {distillation_accuracy:.4f}, AUC: {distillation_auc:.4f if distillation_auc else 'N/A'}")

diff_acc = distillation_accuracy - best_model_accuracy
print(f"  Diferença de acurácia: {diff_acc:.4f} ({diff_acc*100:.2f}%)")

if distillation_auc is not None and best_model_auc is not None:
    diff_auc = distillation_auc - best_model_auc
    print(f"  Diferença de AUC: {diff_auc:.4f} ({diff_auc*100:.2f}%)")

# 9. Visualizar feature importance
# -----------------------------
# Extrair importância das features do modelo de destilação
feature_importance = None
if hasattr(distillation_model, 'coef_'):
    # Para modelos lineares como LogisticRegression
    coefficients = distillation_model.coef_[0] if distillation_model.coef_.ndim > 1 else distillation_model.coef_
    feature_importance = {feature: abs(coef) for feature, coef in zip(X.columns, coefficients)}
elif hasattr(distillation_model, 'feature_importances_'):
    # Para modelos baseados em árvores
    feature_importance = {feature: imp for feature, imp in zip(X.columns, distillation_model.feature_importances_)}

if feature_importance:
    # Ordenar por importância
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nImportância das features no modelo de destilação (top 10):")
    for feature, importance in sorted_importance[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    # Visualizar
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    features = [x[0] for x in sorted_importance[:10]]
    importances = [x[1] for x in sorted_importance[:10]]
    
    plt.barh(features, importances)
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.title('Importância das Features no Modelo de Destilação')
    plt.tight_layout()
    plt.show()

# 10. Criação avançada de modelos com configuração personalizada
# -----------------------------------------------------------
# Criar um modelo personalizado usando o ModelManager
from deepbridge.utils.model_registry import ModelType

# Definir parâmetros personalizados para um RandomForest
custom_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'class_weight': 'balanced'
}

# Criar o modelo personalizado
custom_model = model_manager.create_model(
    model_type=ModelType.RANDOM_FOREST,
    model_params=custom_params
)

# Treinar o modelo
custom_model.fit(X_train, y_train)

# Avaliar no conjunto de teste
custom_accuracy = custom_model.score(X_test, y_test)
custom_auc = roc_auc_score(y_test, custom_model.predict_proba(X_test)[:, 1])

print(f"\nDesempenho do modelo personalizado:")
print(f"  Acurácia: {custom_accuracy:.4f}")
print(f"  AUC: {custom_auc:.4f}")

# Comparar com os demais modelos
print("\nComparação final de todos os modelos:")
print(f"  Modelo personalizado - Acurácia: {custom_accuracy:.4f}, AUC: {custom_auc:.4f}")
print(f"  Modelo de destilação - Acurácia: {distillation_accuracy:.4f}, AUC: {distillation_auc:.4f if distillation_auc else 'N/A'}")
print(f"  Melhor modelo alternativo ({best_model_name}) - Acurácia: {best_model_accuracy:.4f}, AUC: {best_model_auc:.4f if best_model_auc else 'N/A'}")
```

## Funções do ModelManager

A classe `ModelManager` é responsável por gerenciar diversos aspectos relacionados a modelos no DeepBridge:

### 1. Criação de Modelos Alternativos

O método `create_alternative_models` cria automaticamente diversos modelos alternativos para comparação:

- **DummyClassifier**: Modelo baseline simples
- **LogisticRegression**: Modelo linear
- **DecisionTree**: Árvore de decisão simples
- **RandomForest**: Ensemble de árvores
- **GradientBoosting**: Algoritmo de boosting
- **XGBoost**: Implementação otimizada de boosting (se disponível)

### 2. Criação de Modelos de Destilação

O método `create_distillation_model` cria modelos usando técnicas de destilação:

- **surrogate**: Treina um modelo simples para imitar um modelo complexo
- **knowledge_distillation**: Treina um modelo usando combinação de rótulos verdadeiros e previsões de um modelo complexo

Parâmetros importantes:
- **student_model_type**: Tipo do modelo a ser treinado
- **temperature**: Controla a "suavidade" das probabilidades durante a destilação
- **alpha**: Peso entre rótulos verdadeiros e probabilidades do teacher
- **n_trials**: Número de tentativas para otimização de hiperparâmetros

### 3. Criação de Modelos Personalizados

O método `create_model` permite criar modelos com configurações personalizadas:

```python
model = model_manager.create_model(
    model_type=ModelType.RANDOM_FOREST,
    model_params={'n_estimators': 200, 'max_depth': 5}
)
```

## Tipos de Modelos Suportados

Os tipos de modelos são definidos na enumeração `ModelType` no módulo `utils.model_registry`:

- **DUMMY**: DummyClassifier (baseline)
- **LOGISTIC_REGRESSION**: Regressão Logística
- **DECISION_TREE**: Árvore de Decisão
- **RANDOM_FOREST**: Random Forest
- **GRADIENT_BOOSTING**: Gradient Boosting
- **XGB**: XGBoost (se disponível)
- **NEURAL_NETWORK**: Redes Neurais (se disponível via TensorFlow/Keras)
- **SVM**: Support Vector Machine
- **KNN**: K-Nearest Neighbors

## Cenários de Uso

### Comparação de Modelos

O `ModelManager` facilita a criação e comparação de diferentes tipos de modelos, permitindo identificar a abordagem mais adequada para seu problema.

### Destilação de Modelos

A destilação é útil quando:
- Você tem um modelo complexo e precisa de um modelo mais simples para produção
- Deseja um modelo interpretável que mantém o desempenho de um modelo "caixa-preta"
- Precisa reduzir o tamanho ou complexidade do modelo para implantação

### Criação de Modelos Personalizados

Quando você tem requisitos específicos para os hiperparâmetros do modelo, o `ModelManager` oferece uma interface padronizada para criar e configurar modelos.

## Pontos-Chave

- O `ModelManager` centraliza a criação e gerenciamento de modelos no DeepBridge
- Os modelos alternativos permitem comparar diferentes abordagens
- A destilação de modelos permite obter modelos mais simples e interpretáveis
- A criação de modelos personalizados oferece flexibilidade para configurações específicas
- O gerenciador abstrai detalhes de implementação, permitindo focar na lógica de negócio