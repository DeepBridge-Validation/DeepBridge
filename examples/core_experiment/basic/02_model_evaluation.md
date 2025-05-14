# Avaliação de Modelos

Este exemplo demonstra como usar o componente de avaliação de modelos do módulo `core.experiment` para avaliar e comparar diferentes modelos.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Importar componentes do DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.model_evaluation import ModelEvaluation
from deepbridge.metrics.classification import Classification

# 1. Preparar dados e modelos
# ---------------------------
# Carregar dados de classificação binária
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar diferentes modelos para comparação
models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'logistic_regression': LogisticRegression(random_state=42)
}

# Treinar cada modelo
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Modelo {name} treinado")

# 2. Usar ModelEvaluation diretamente
# ----------------------------------
# Criar o avaliador de modelos
metrics_calculator = Classification()  # Para classificação binária
model_evaluator = ModelEvaluation("binary_classification", metrics_calculator)

print("\nAvaliação direta dos modelos:")
model_metrics = {}

for name, model in models.items():
    # Fazer predições
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas usando o avaliador
    metrics = model_evaluator.calculate_metrics(y_test, y_pred, y_prob)
    model_metrics[name] = metrics
    
    print(f"\nMétricas para {name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# 3. Comparar os modelos
# ---------------------
# Criar uma tabela comparativa de métricas
comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
comparison_df = pd.DataFrame(index=models.keys(), columns=comparison_metrics)

for model_name, metrics in model_metrics.items():
    for metric in comparison_metrics:
        comparison_df.loc[model_name, metric] = metrics.get(metric, np.nan)

print("\nComparação de modelos:")
print(comparison_df.round(4))

# 4. Uso dentro do framework de Experimento
# ----------------------------------------
# Criar um DBDataset com o melhor modelo (por exemplo, o Random Forest)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=models['random_forest'],
    dataset_name="Breast Cancer"
)

# Criar o experimento
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    config={'verbose': True}
)

# 5. Comparar todos os modelos (primário e alternativos)
# -----------------------------------------------------
# O Experiment cria automaticamente modelos alternativos para comparação
all_models_comparison = experiment.compare_all_models(dataset='test')

print("\nComparação automática de todos os modelos no Experiment:")
models_df = pd.DataFrame(all_models_comparison).T
print(models_df.round(4))

# 6. Avaliação da importância das features
# --------------------------------------
feature_importance = experiment.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': list(feature_importance.keys()),
    'Importance': list(feature_importance.values())
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("\nTop 10 features mais importantes:")
print(importance_df.head(10))

# 7. Avaliação de modelo de destilação
# ----------------------------------
# Criar um modelo de destilação (modelo student treinado com as previsões do teacher)
print("\nTreinando modelo destilado...")
experiment.fit(
    student_model_type='logistic_regression',
    temperature=1.0,
    alpha=0.5,  # Peso para combinar rótulos verdadeiros e probabilidades do teacher
    use_probabilities=True,
    verbose=False
)

# Avaliar o modelo destilado
student_predictions = experiment.get_student_predictions(dataset='test')
print("\nMétricas do modelo destilado:")
for metric, value in experiment._results_data['test'].items():
    print(f"  {metric}: {value:.4f}")

# 8. Comparação final incluindo o modelo destilado
# ----------------------------------------------
final_comparison = experiment.compare_all_models(dataset='test')
final_df = pd.DataFrame(final_comparison).T
print("\nComparação final incluindo modelo destilado:")
print(final_df.round(4))

# 9. Criando uma matriz de confusão para o melhor modelo
# ----------------------------------------------------
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Maligno', 'Benigno'],
                yticklabels=['Maligno', 'Benigno'])
    plt.xlabel('Predição')
    plt.ylabel('Real')
    plt.title(title)
    plt.show()

# Obter predições do melhor modelo
best_model_name = comparison_df['accuracy'].idxmax()
best_model = models[best_model_name]
best_preds = best_model.predict(X_test)

# Plotar matriz de confusão
plot_confusion_matrix(y_test, best_preds, f'Matriz de Confusão - {best_model_name}')

# 10. Avaliação de curvas ROC
# -------------------------
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

# Adicionar modelo destilado se disponível
if hasattr(experiment, 'distillation_model') and experiment.distillation_model is not None:
    y_prob_dist = experiment.distillation_model.predict_proba(X_test)[:, 1]
    fpr_dist, tpr_dist, _ = roc_curve(y_test, y_prob_dist)
    roc_auc_dist = auc(fpr_dist, tpr_dist)
    
    plt.plot(fpr_dist, tpr_dist, lw=2, linestyle='--', 
             label=f'Modelo Destilado (AUC = {roc_auc_dist:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curvas ROC - Comparação de Modelos')
plt.legend(loc="lower right")
plt.show()
```

## Métricas de Avaliação

O módulo de avaliação de modelos no DeepBridge calcula diversas métricas conforme o tipo de experimento:

### Para Classificação Binária
- **accuracy**: Porcentagem de previsões corretas
- **precision**: Precisão (Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos))
- **recall**: Recall (Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos))
- **f1_score**: Média harmônica entre precisão e recall
- **roc_auc**: Área sob a curva ROC
- **log_loss**: Perda logarítmica (menor é melhor)
- **brier_score**: Pontuação de Brier (menor é melhor)

### Para Regressão
- **mse**: Erro Quadrático Médio
- **rmse**: Raiz do Erro Quadrático Médio
- **mae**: Erro Absoluto Médio
- **r2**: Coeficiente de Determinação (R²)
- **explained_variance**: Variância Explicada

## Componentes Principais

### ModelEvaluation
A classe `ModelEvaluation` é responsável por:
- Calcular métricas apropriadas para o tipo de experimento
- Comparar diferentes modelos
- Avaliar modelos destilados

### Métodos Importantes
- **calculate_metrics()**: Calcula métricas para um modelo
- **evaluate_distillation()**: Avalia um modelo destilado
- **compare_all_models()**: Compara todos os modelos disponíveis
- **get_predictions()**: Obtém previsões de um modelo

## Classes de Métricas

O DeepBridge fornece calculadoras de métricas específicas para diferentes tipos de modelos:

- **Classification**: Para modelos de classificação binária e multiclasse
- **Regression**: Para modelos de regressão
- **TimeSeries**: Para modelos de séries temporais

## Pontos-Chave

- A avaliação detalhada de modelos é uma parte fundamental do framework
- É possível avaliar modelos individualmente ou comparar vários modelos
- As métricas calculadas dependem do tipo de experimento
- A classe `Experiment` usa `ModelEvaluation` internamente, mas você também pode usá-la diretamente
- A comparação de modelos ajuda a identificar o melhor modelo para seu caso de uso
- A importância das features ajuda a entender quais variáveis são mais relevantes
- Os modelos destilados podem ser comparados com os modelos originais para avaliar a qualidade da destilação