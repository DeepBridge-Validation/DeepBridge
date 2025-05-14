# Gerenciamento de Dados com DataManager

O `DataManager` é uma classe essencial para gerenciar dados dentro de experimentos no DeepBridge. Ela é responsável pela divisão dos conjuntos de dados em treino e teste, além de facilitar o acesso a esses dados e manipular predições de probabilidade.

## Inicialização Básica

```python
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.data_manager import DataManager
import pandas as pd
import numpy as np

# Criar um dataset
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})

# Criar um objeto DBDataset
dataset = DBDataset(
    data=df,
    target_column='target',
    categorical_columns=[],
    numerical_columns=['feature1', 'feature2', 'feature3']
)

# Criar uma instância do DataManager
data_manager = DataManager(
    dataset=dataset,
    test_size=0.3,  # 30% para teste
    random_state=42  # Para reprodutibilidade
)

# Preparar os dados dividindo-os em treino e teste
data_manager.prepare_data()

# Acessar os conjuntos de treino e teste
X_train, y_train, prob_train = data_manager.get_dataset_split('train')
X_test, y_test, prob_test = data_manager.get_dataset_split('test')

print(f"Tamanho do conjunto de treino: {X_train.shape[0]} exemplos")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} exemplos")
```

## Trabalhando com Probabilidades

O `DataManager` também facilita o trabalho com probabilidades, convertendo-as em predições binárias quando necessário:

```python
import pandas as pd
import numpy as np
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.data_manager import DataManager
from sklearn.ensemble import RandomForestClassifier

# Criar e preparar o dataset
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})

dataset = DBDataset(
    data=df,
    target_column='target'
)

# Inicializar o DataManager
data_manager = DataManager(dataset, test_size=0.3, random_state=42)
data_manager.prepare_data()

# Obter dados de treino e teste
X_train, y_train, _ = data_manager.get_dataset_split('train')
X_test, y_test, _ = data_manager.get_dataset_split('test')

# Treinar um modelo que gera probabilidades
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Gerar probabilidades para o conjunto de teste
probabilities = model.predict_proba(X_test)
prob_df = pd.DataFrame(probabilities, columns=[f'class_{i}' for i in range(probabilities.shape[1])])

# Converter probabilidades em predições binárias com diferentes limiares
predictions_default = data_manager.get_binary_predictions(prob_df)  # Limiar padrão de 0.5
predictions_conservative = data_manager.get_binary_predictions(prob_df, threshold=0.7)  # Limiar mais conservador

print("Predições com limiar 0.5:", predictions_default.value_counts())
print("Predições com limiar 0.7:", predictions_conservative.value_counts())
```

## Integração com Experimentos

O `DataManager` é normalmente usado dentro de um objeto `Experiment`, que automatiza muitos dos processos:

```python
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Criar dados de exemplo
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})

# Criar modelo e dataset
model = RandomForestClassifier(random_state=42)
dataset = DBDataset(
    data=df,
    target_column='target',
    model=model  # Associar o modelo ao dataset
)

# Criar experimento (que usa DataManager internamente)
experiment = Experiment(
    name="data_manager_demo",
    dataset=dataset,
    test_size=0.3,
    random_state=42
)

# O experimento usa o DataManager internamente
# Acesso direto aos dados gerenciados pelo DataManager
print(f"Características de treino: {experiment.X_train.shape}")
print(f"Características de teste: {experiment.X_test.shape}")
print(f"Alvos de treino: {experiment.y_train.shape}")
print(f"Alvos de teste: {experiment.y_test.shape}")
```

## Personalização Avançada de Divisão de Dados

Podemos personalizar a forma como o `DataManager` divide os dados integrando-o com a estratificação e técnicas de validação cruzada:

```python
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.data_manager import DataManager
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

# Criar um dataset desbalanceado (com muitos mais exemplos da classe 0)
df = pd.DataFrame({
    'feature1': np.random.rand(1000),
    'feature2': np.random.rand(1000),
    'target': np.random.choice([0, 1], size=1000, p=[0.9, 0.1])  # 90% classe 0, 10% classe 1
})

# Método 1: Usar o DataManager com um DBDataset customizado
# Criar uma classe que estende o DBDataset para divisão estratificada
class StratifiedDBDataset(DBDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratify = True
    
    def split_data(self, test_size, random_state):
        # Sobrescrever o método de divisão para usar estratificação
        if self.stratify and self.target is not None:
            train_idx, test_idx = train_test_split(
                range(len(self.data)), 
                test_size=test_size, 
                random_state=random_state,
                stratify=self.target  # Estratificar pela coluna alvo
            )
        else:
            train_idx, test_idx = super().split_data(test_size, random_state)
        
        return train_idx, test_idx

# Usar a classe personalizada com o DataManager
stratified_dataset = StratifiedDBDataset(
    data=df,
    target_column='target'
)

data_manager = DataManager(stratified_dataset, test_size=0.3, random_state=42)
data_manager.prepare_data()

# Verificar a distribuição da classe minoritária
_, y_train, _ = data_manager.get_dataset_split('train')
_, y_test, _ = data_manager.get_dataset_split('test')

print("Distribuição no conjunto de treino:", y_train.value_counts(normalize=True))
print("Distribuição no conjunto de teste:", y_test.value_counts(normalize=True))

# Método 2: Validação cruzada com DataManager
# Implementamos manualmente a validação cruzada com múltiplos DataManagers
dataset = DBDataset(data=df, target_column='target')

# Configurar a validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lista para armazenar os DataManagers de cada fold
fold_managers = []

# Criar DataManagers para cada fold
for fold_idx, (train_index, test_index) in enumerate(kf.split(df)):
    # Criar um dataset específico para este fold
    fold_data = df.copy()
    fold_dataset = DBDataset(data=fold_data, target_column='target')
    
    # Criar o DataManager
    fold_manager = DataManager(fold_dataset, test_size=len(test_index)/len(df), random_state=42)
    
    # Atribuir índices manualmente
    fold_manager.train_indices = train_index
    fold_manager.test_indices = test_index
    fold_manager.X_train = fold_data.iloc[train_index].drop('target', axis=1)
    fold_manager.y_train = fold_data.iloc[train_index]['target']
    fold_manager.X_test = fold_data.iloc[test_index].drop('target', axis=1)
    fold_manager.y_test = fold_data.iloc[test_index]['target']
    
    fold_managers.append(fold_manager)
    
    print(f"Fold {fold_idx+1}: {len(train_index)} exemplos de treino, {len(test_index)} exemplos de teste")
```

## Manipulação Avançada de Dados

O `DataManager` pode ser usado junto com utilitários de dados para transformar e preparar dados para experimentos:

```python
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.data_manager import DataManager
from deepbridge.utils.feature_manager import FeatureManager  # Classe hipotética para ilustração
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Criar dados com valores faltantes
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': [np.nan if i % 10 == 0 else np.random.rand() for i in range(100)],
    'feature3': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})

# Criar o dataset
dataset = DBDataset(
    data=df,
    target_column='target'
)

# Inicializar o DataManager
data_manager = DataManager(dataset, test_size=0.3, random_state=42)
data_manager.prepare_data()

# Obter dados de treino e teste
X_train, y_train, _ = data_manager.get_dataset_split('train')
X_test, y_test, _ = data_manager.get_dataset_split('test')

print("Antes do pré-processamento:")
print("Valores NaN no treino:", X_train.isna().sum())
print("Valores NaN no teste:", X_test.isna().sum())

# Aplicar pré-processamento apenas no conjunto de treino e depois transformar o teste
# 1. Imputação de valores faltantes
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# 2. Normalização
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_imputed),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_imputed),
    columns=X_test.columns,
    index=X_test.index
)

print("\nDepois do pré-processamento:")
print("Valores NaN no treino:", X_train_scaled.isna().sum())
print("Valores NaN no teste:", X_test_scaled.isna().sum())
print("Média das características no treino:", X_train_scaled.mean())
print("Desvio padrão das características no treino:", X_train_scaled.std())

# Atualizar os dados no DataManager para uso futuro
data_manager.X_train = X_train_scaled
data_manager.X_test = X_test_scaled
```

O `DataManager` é um componente flexível que pode ser integrado com várias técnicas de pré-processamento e manipulação de dados, permitindo um fluxo de trabalho organizado para experimentos de machine learning no DeepBridge.