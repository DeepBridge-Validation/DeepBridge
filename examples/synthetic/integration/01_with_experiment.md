# Integração entre Synthetic Data e Experiment/DBDataset

Este exemplo demonstra como integrar o módulo `synthetic` com as classes `Experiment` e `DBDataset` do DeepBridge para criar um fluxo de trabalho completo.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Importar classes do DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.synthetic.synthesizer import Synthesize

# Carregar um conjunto de dados para classificação binária
data = load_breast_cancer()
df = pd.DataFrame(
    data=data['data'],
    columns=data['feature_names']
)
df['target'] = data['target']

print(f"Conjunto de dados original: {df.shape}")
print(df.head())

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1),
    df['target'],
    test_size=0.2,
    random_state=42
)

# Criar DataFrames de treino e teste
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Treinar um modelo de referência
orig_model = RandomForestClassifier(random_state=42)
orig_model.fit(X_train, y_train)

# Passo 1: Criar DBDataset com dados reais e modelo
original_dbdataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=orig_model,
    dataset_name="Breast Cancer (Original)"
)

print("\nDBDataset original criado com sucesso")

# Passo 2: Gerar dados sintéticos com o módulo synthetic
synthetic_data = Synthesize(
    dataset=train_df,  # Usar apenas os dados de treino para gerar sintéticos
    method='gaussian_copula',
    num_samples=len(train_df),
    random_state=42,
    print_metrics=False,
    verbose=True
)

print(f"\nDados sintéticos gerados: {synthetic_data.data.shape}")
print(synthetic_data.data.head())

# Passo 3: Criar um DBDataset com dados sintéticos
# Vamos manter o mesmo conjunto de teste real para avaliação justa
synthetic_train = synthetic_data.data

# Treinar um modelo nos dados sintéticos
synth_model = RandomForestClassifier(random_state=42)
synth_model.fit(
    synthetic_train.drop('target', axis=1),
    synthetic_train['target']
)

# Criar DBDataset com dados sintéticos
synthetic_dbdataset = DBDataset(
    train_data=synthetic_train,
    test_data=test_df,  # Mesmo conjunto de teste real
    target_column='target',
    model=synth_model,
    dataset_name="Breast Cancer (Synthetic)"
)

print("\nDBDataset sintético criado com sucesso")

# Passo 4: Criar experimentos com os dois DBDatasets
# Experimento com dados originais
original_experiment = Experiment(
    dataset=original_dbdataset,
    experiment_type="binary_classification",
    tests=["robustness", "uncertainty"],
    random_state=42
)

# Experimento com dados sintéticos
synthetic_experiment = Experiment(
    dataset=synthetic_dbdataset,
    experiment_type="binary_classification",
    tests=["robustness", "uncertainty"],
    random_state=42
)

# Passo 5: Executar testes em ambos os experimentos
print("\nExecutando testes no experimento com dados originais...")
original_results = original_experiment.run_tests(config_name="quick")

print("\nExecutando testes no experimento com dados sintéticos...")
synthetic_results = synthetic_experiment.run_tests(config_name="quick")

# Passo 6: Comparar os resultados iniciais
print("\n===== Comparação de Métricas Iniciais =====")
orig_metrics = original_results.results['initial_results']['models']['primary_model']['metrics']
synth_metrics = synthetic_results.results['initial_results']['models']['primary_model']['metrics']

# Métricas para comparar
metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

print(f"{'Métrica':<15} {'Original':<10} {'Sintético':<10} {'Diferença %':<15}")
print("-" * 50)
for metric in metrics_to_compare:
    if metric in orig_metrics and metric in synth_metrics:
        orig_value = orig_metrics[metric]
        synth_value = synth_metrics[metric]
        diff_pct = (synth_value / orig_value - 1) * 100
        print(f"{metric:<15} {orig_value:.4f}     {synth_value:.4f}     {diff_pct:+.2f}%")

# Passo 7: Comparar resultados de testes de robustez
if 'robustness' in original_results.results and 'robustness' in synthetic_results.results:
    print("\n===== Comparação de Robustez =====")
    orig_robust = original_results.results['robustness'].get('overall_robustness_score')
    synth_robust = synthetic_results.results['robustness'].get('overall_robustness_score')
    
    if orig_robust is not None and synth_robust is not None:
        diff_pct = (synth_robust / orig_robust - 1) * 100
        print(f"Pontuação de Robustez Original: {orig_robust:.4f}")
        print(f"Pontuação de Robustez Sintético: {synth_robust:.4f}")
        print(f"Diferença: {diff_pct:+.2f}%")

# Passo 8: Comparar resultados de testes de incerteza
if 'uncertainty' in original_results.results and 'uncertainty' in synthetic_results.results:
    print("\n===== Comparação de Incerteza =====")
    
    metrics_to_compare = [
        'calibration_error', 
        'coverage_error',
        'expected_calibration_error'
    ]
    
    for metric in metrics_to_compare:
        orig_value = original_results.results['uncertainty'].get(metric)
        synth_value = synthetic_results.results['uncertainty'].get(metric)
        
        if orig_value is not None and synth_value is not None:
            if orig_value != 0:
                diff_pct = (synth_value / orig_value - 1) * 100
                print(f"{metric}: Original={orig_value:.4f}, Sintético={synth_value:.4f}, Diferença={diff_pct:+.2f}%")
            else:
                print(f"{metric}: Original={orig_value:.4f}, Sintético={synth_value:.4f}")

# Passo 9: Gerar relatórios para ambos os experimentos
# Definir diretório para relatórios
import os
output_dir = "reports"
os.makedirs(output_dir, exist_ok=True)

# Gerar relatórios de robustez
original_report = original_experiment.save_html(
    test_type="robustness",
    file_path=os.path.join(output_dir, "original_robustness_report.html"),
    model_name="Modelo Original"
)

synthetic_report = synthetic_experiment.save_html(
    test_type="robustness",
    file_path=os.path.join(output_dir, "synthetic_robustness_report.html"),
    model_name="Modelo Treinado em Dados Sintéticos"
)

print(f"\nRelatório de robustez original: {original_report}")
print(f"Relatório de robustez sintético: {synthetic_report}")

# Passo 10: Demonstração de como usar os dados sintéticos para data augmentation
# Combinar dados originais e sintéticos para treinar um modelo melhorado
combined_train = pd.concat([train_df, synthetic_train], ignore_index=True)
print(f"\nConjunto de treino combinado: {combined_train.shape} (Original + Sintético)")

# Treinar um modelo nos dados combinados
combined_model = RandomForestClassifier(random_state=42)
combined_model.fit(
    combined_train.drop('target', axis=1),
    combined_train['target']
)

# Avaliar o modelo combinado no conjunto de teste
combined_preds = combined_model.predict(X_test)
from sklearn.metrics import accuracy_score, roc_auc_score

# Calcular métricas para o modelo combinado
combined_acc = accuracy_score(y_test, combined_preds)
combined_auc = roc_auc_score(y_test, combined_model.predict_proba(X_test)[:, 1])

# Comparar com os modelos original e sintético
orig_acc = accuracy_score(y_test, orig_model.predict(X_test))
orig_auc = roc_auc_score(y_test, orig_model.predict_proba(X_test)[:, 1])

synth_acc = accuracy_score(y_test, synth_model.predict(X_test))
synth_auc = roc_auc_score(y_test, synth_model.predict_proba(X_test)[:, 1])

print("\n===== Comparação Final dos Modelos =====")
print(f"{'Métrica':<10} {'Original':<10} {'Sintético':<10} {'Combinado':<10}")
print("-" * 40)
print(f"{'Acurácia':<10} {orig_acc:.4f}     {synth_acc:.4f}     {combined_acc:.4f}")
print(f"{'AUC-ROC':<10} {orig_auc:.4f}     {synth_auc:.4f}     {combined_auc:.4f}")

# Demonstrar como isso poderia ser incorporado em um fluxo de validação cruzada
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

print("\n===== Demonstração de Validação Cruzada com Dados Sintéticos =====")
k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Preparar listas para armazenar resultados
orig_fold_scores = []
synth_fold_scores = []
combined_fold_scores = []

# Executar validação cruzada
fold = 1
for train_idx, val_idx in kf.split(X_train):
    # Dividir dados originais de treino em treino e validação
    fold_train_X, fold_val_X = X_train.iloc[train_idx], X_train.iloc[val_idx]
    fold_train_y, fold_val_y = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    print(f"\nProcessando fold {fold}/{k_folds}...")
    
    # Criar DataFrame de treino para fold atual
    fold_train_df = pd.concat([fold_train_X, fold_train_y], axis=1)
    
    # Gerar dados sintéticos para este fold
    fold_synthetic = Synthesize(
        dataset=fold_train_df,
        method='gaussian_copula',
        num_samples=len(fold_train_df),
        random_state=42 + fold,
        print_metrics=False,
        verbose=False
    ).data
    
    # Treinar modelos para este fold
    # 1. Modelo original
    fold_orig_model = RandomForestClassifier(random_state=42)
    fold_orig_model.fit(fold_train_X, fold_train_y)
    
    # 2. Modelo sintético
    fold_synth_model = RandomForestClassifier(random_state=42)
    fold_synth_model.fit(
        fold_synthetic.drop('target', axis=1),
        fold_synthetic['target']
    )
    
    # 3. Modelo combinado
    fold_combined_train = pd.concat([fold_train_df, fold_synthetic], ignore_index=True)
    fold_combined_model = RandomForestClassifier(random_state=42)
    fold_combined_model.fit(
        fold_combined_train.drop('target', axis=1),
        fold_combined_train['target']
    )
    
    # Avaliar modelos na validação
    fold_orig_auc = roc_auc_score(
        fold_val_y, 
        fold_orig_model.predict_proba(fold_val_X)[:, 1]
    )
    orig_fold_scores.append(fold_orig_auc)
    
    fold_synth_auc = roc_auc_score(
        fold_val_y, 
        fold_synth_model.predict_proba(fold_val_X)[:, 1]
    )
    synth_fold_scores.append(fold_synth_auc)
    
    fold_combined_auc = roc_auc_score(
        fold_val_y, 
        fold_combined_model.predict_proba(fold_val_X)[:, 1]
    )
    combined_fold_scores.append(fold_combined_auc)
    
    print(f"Fold {fold} AUC - Original: {fold_orig_auc:.4f}, Sintético: {fold_synth_auc:.4f}, Combinado: {fold_combined_auc:.4f}")
    
    fold += 1

# Mostrar resultados médios
print("\nResultados médios da validação cruzada (AUC-ROC):")
print(f"Original: {np.mean(orig_fold_scores):.4f} ± {np.std(orig_fold_scores):.4f}")
print(f"Sintético: {np.mean(synth_fold_scores):.4f} ± {np.std(synth_fold_scores):.4f}")
print(f"Combinado: {np.mean(combined_fold_scores):.4f} ± {np.std(combined_fold_scores):.4f}")
```

## Fluxo de Trabalho Integrado

Este exemplo demonstra um fluxo de trabalho completo de integração entre os módulos `synthetic`, `core.db_data` e `core.experiment` do DeepBridge:

### 1. Preparação dos Dados
- Carregar e dividir o conjunto de dados em treino e teste
- Treinar um modelo de referência nos dados originais

### 2. Criação de DBDatasets
- Criar um `DBDataset` com dados originais
- Gerar dados sintéticos usando o módulo `synthetic`
- Criar um `DBDataset` com dados sintéticos

### 3. Experimentos e Testes
- Criar experimentos com os dois `DBDatasets`
- Executar testes de robustez e incerteza
- Comparar resultados dos experimentos

### 4. Geração de Relatórios
- Gerar relatórios HTML para análise visual
- Comparar métricas entre modelos

### 5. Data Augmentation
- Combinar dados originais e sintéticos para melhorar o modelo
- Avaliar os ganhos de desempenho

### 6. Validação Cruzada
- Implementar um fluxo de validação cruzada com dados sintéticos
- Comparar o desempenho dos modelos em cada fold

## Cenários de Uso

### Cenário 1: Avaliação de Impacto de Dados Sintéticos
- Gerar dados sintéticos a partir dos dados originais
- Comparar o desempenho de modelos treinados em dados reais vs. sintéticos
- Avaliar se os dados sintéticos preservam características importantes para modelagem

### Cenário 2: Data Augmentation
- Aumentar o conjunto de treino com dados sintéticos
- Melhorar o desempenho e a robustez do modelo
- Especialmente útil quando o conjunto de dados original é pequeno

### Cenário 3: Privacidade e Compartilhamento de Dados
- Gerar dados sintéticos que preservam características estatísticas, mas não dados individuais
- Compartilhar apenas os dados sintéticos para proteger a privacidade
- Avaliar o impacto na utilidade dos dados

## Pontos-Chave

- A integração entre os módulos permite um fluxo de trabalho completo de modelagem e avaliação
- Os dados sintéticos podem ser usados para:
  - Aumentar o tamanho do conjunto de treino
  - Melhorar a robustez dos modelos
  - Permitir compartilhamento de dados preservando a privacidade
- O conjunto de teste deve permanecer o mesmo para comparações justas
- Modelos treinados em dados sintéticos geralmente têm desempenho ligeiramente inferior aos treinados em dados reais
- A combinação de dados reais e sintéticos frequentemente produz os melhores resultados
- Cada passo do fluxo é configurável, permitindo personalização para diferentes casos de uso