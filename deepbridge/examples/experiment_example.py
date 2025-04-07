"""
Este exemplo demonstra como utilizar a classe Experiment do DeepBridge.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Importando os componentes do DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment

def main():
    print("DeepBridge - Exemplo da Classe Experiment")
    print("=======================================")
    
    # Etapa 1: Criar um conjunto de dados sintético
    print("\nCriando conjunto de dados sintético...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Etapa 2: Transformar em DataFrame e Series do pandas
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    # Etapa 3: Dividir em treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42
    )
    
    # Etapa 4: Treinar um modelo
    print("Treinando modelo Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Acurácia do modelo no conjunto de teste: {model.score(X_test, y_test):.4f}")
    
    # Etapa 5: Criar um DBDataset
    print("\nCriando DBDataset...")
    # Criar dataframes completos
    train_df = X_train.copy()
    train_df["target"] = y_train
    test_df = X_test.copy()
    test_df["target"] = y_test
    
    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column="target",
        model=model
    )
    
    # Etapa 6: Criar um objeto Experiment
    print("\nCriando objeto Experiment...")
    try:
        experiment = Experiment(
            dataset=dataset,
            experiment_type="binary_classification",
            tests=["robustness", "uncertainty"]
        )
        print("✅ Experiment criado com sucesso!")
        
        # Etapa 7: Executar testes
        print("\nExecutando testes com configuração 'quick'...")
        results = experiment.run_tests("quick")
        print("✅ Testes executados com sucesso!")
        
        # Etapa 8: Acessar resultados
        print("\nResultados dos testes:")
        if 'robustness' in results:
            robustness_score = results['robustness']['primary_model'].get('robustness_score', 0)
            print(f"  - Pontuação de robustez: {robustness_score:.4f}")
            
        if 'uncertainty' in results:
            calibration_error = results['uncertainty']['primary_model'].get('calibration_error', 0)
            print(f"  - Erro de calibração: {calibration_error:.4f}")
        
        # Etapa 9: Salvar relatório
        print("\nSalvando relatório HTML...")
        report_path = results.save_report("./experiment_report.html")
        print(f"✅ Relatório salvo em: {report_path}")
        
    except Exception as e:
        print(f"❌ Erro ao executar o experimento: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nExemplo concluído.")

if __name__ == "__main__":
    main()