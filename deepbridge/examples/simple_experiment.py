"""
Exemplo bem simples de como usar a classe Experiment do DeepBridge.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Importando o DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment

def main():
    print("DeepBridge - Exemplo Simples de Experiment")
    print("==========================================")
    
    # 1. Criar dados sintéticos
    X, y = make_classification(
        n_samples=500, 
        n_features=5, 
        n_informative=3,
        random_state=42
    )
    
    # 2. Transformar em DataFrame/Series
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    # 3. Criar o conjunto de teste/treino
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42
    )
    
    # 4. Treinar um modelo simples
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Criar o DBDataset
    dataset = DBDataset(
        feature_data=X_df,  # Usando os dados completos
        target_data=y_series,  # Usando o target completo
        model=model  # Modelo treinado
    )
    
    # 6. Criar e usar a classe Experiment
    try:
        print("\nCriando a instância de Experiment...")
        # Inicializando com apenas um tipo de teste para simplicidade
        experiment = Experiment(
            dataset=dataset,
            experiment_type="binary_classification",
            tests=["robustness"]  # Apenas teste de robustez
        )
        
        print("✅ Experiment criado com sucesso!")
        
        # Executar os testes
        print("\nExecutando testes com configuração 'quick'...")
        results = experiment.run_tests("quick")
        
        # Mostrar resultados básicos
        print("\nResultados básicos:")
        robustness_score = results.get('robustness', {}).get('primary_model', {}).get('robustness_score', 0)
        print(f"Score de robustez: {robustness_score:.4f}")
        
        print("\nExemplo concluído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro ao criar/executar Experiment: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()