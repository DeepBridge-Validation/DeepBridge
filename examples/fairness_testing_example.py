"""
Exemplo completo de uso do Fairness Testing Module do DeepBridge

Este script demonstra como usar o módulo de testes de fairness para
avaliar se um modelo de ML apresenta discriminação contra grupos protegidos.

Casos de uso:
- Banking: Aprovação de crédito
- Healthcare: Diagnóstico médico
- Insurance: Cálculo de prêmios
- Lending: Empréstimos pessoais
- Employment: Seleção de candidatos
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import DeepBridge
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment


def create_synthetic_lending_dataset(n_samples=1000, seed=42):
    """
    Cria dataset sintético de empréstimos com bias intencional.

    Features:
    - income: Renda anual
    - age: Idade
    - credit_score: Score de crédito (300-850)
    - employment_years: Anos de emprego
    - gender: Gênero (M/F) - PROTECTED
    - race: Raça (White/Black/Hispanic) - PROTECTED

    Target:
    - loan_approved: 1 se empréstimo aprovado, 0 caso contrário

    O dataset tem bias intencional: mulheres e minorias têm menor
    taxa de aprovação mesmo com mesmas qualificações.
    """
    np.random.seed(seed)

    # Generate features
    income = np.random.lognormal(10.5, 0.5, n_samples)  # $30k-$150k
    age = np.random.randint(22, 65, n_samples)
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    employment_years = np.random.poisson(5, n_samples).clip(0, 40)

    # Protected attributes (with bias)
    gender = np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4])
    race = np.random.choice(
        ['White', 'Black', 'Hispanic'],
        n_samples,
        p=[0.6, 0.25, 0.15]
    )

    # Create loan approval with INTENTIONAL BIAS
    # Base approval probability
    approval_prob = (
        0.2 +  # Base rate
        (income - income.min()) / (income.max() - income.min()) * 0.3 +
        (credit_score - 300) / 550 * 0.4 +
        employment_years / 40 * 0.1
    )

    # Add BIAS: reduce approval for women and minorities
    bias_factor = np.ones(n_samples)
    bias_factor[gender == 'F'] *= 0.7  # Women 30% less likely
    bias_factor[race == 'Black'] *= 0.6  # Black 40% less likely
    bias_factor[race == 'Hispanic'] *= 0.75  # Hispanic 25% less likely

    approval_prob *= bias_factor
    approval_prob = approval_prob.clip(0, 1)

    # Generate binary outcomes
    loan_approved = (np.random.random(n_samples) < approval_prob).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'income': income,
        'age': age,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'gender': gender,
        'race': race,
        'loan_approved': loan_approved
    })

    return df


def example_1_basic_fairness_testing():
    """
    Exemplo 1: Teste básico de fairness com Logistic Regression

    Demonstra:
    - Criação de dataset com bias
    - Treinamento de modelo simples
    - Execução de testes de fairness
    - Análise de resultados
    """
    print("="*70)
    print("EXEMPLO 1: Teste Básico de Fairness")
    print("="*70)

    # 1. Create dataset with bias
    print("\n1. Criando dataset sintético de empréstimos (com bias intencional)...")
    df = create_synthetic_lending_dataset(n_samples=1000)

    print(f"   Dataset shape: {df.shape}")
    print(f"   Approval rate geral: {df['loan_approved'].mean():.2%}")
    print(f"   Approval rate por gênero:")
    print(df.groupby('gender')['loan_approved'].mean())
    print(f"   Approval rate por raça:")
    print(df.groupby('race')['loan_approved'].mean())

    # 2. Prepare features and target
    print("\n2. Preparando features e target...")
    feature_cols = ['income', 'age', 'credit_score', 'employment_years', 'gender', 'race']
    X = df[feature_cols].copy()

    # Encode categorical variables
    X['gender'] = (X['gender'] == 'M').astype(int)
    X['race_encoded'] = pd.Categorical(X['race']).codes
    X = X.drop('race', axis=1)

    # Add race back for fairness testing (need original values)
    X['race'] = df['race'].values

    y = df['loan_approved'].values

    # 3. Train model
    print("\n3. Treinando modelo (Logistic Regression)...")
    model = LogisticRegression(random_state=42, max_iter=1000)

    # Train on encoded features (excluding 'race' for fairness)
    X_train_encoded = X[['income', 'age', 'credit_score', 'employment_years', 'gender', 'race_encoded']]
    model.fit(X_train_encoded, y)

    train_score = model.score(X_train_encoded, y)
    print(f"   Train Accuracy: {train_score:.4f}")

    # 4. Create DBDataset
    print("\n4. Criando DBDataset...")
    dataset = DBDataset(
        features=X,
        target=y,
        model=model
    )

    # 5. Run fairness tests
    print("\n5. Executando testes de fairness...")
    print("   Protected attributes: ['gender', 'race']")

    experiment = Experiment(
        dataset=dataset,
        experiment_type="binary_classification",
        tests=["fairness"],
        protected_attributes=['gender', 'race']  # Teste ambos atributos protegidos
    )

    # Run fairness tests
    fairness_results = experiment.run_fairness_tests(config='full')

    # 6. Analyze results
    print("\n" + "="*70)
    print("RESULTADOS DO TESTE DE FAIRNESS")
    print("="*70)

    print(f"\nOverall Fairness Score: {fairness_results.overall_fairness_score:.3f} / 1.000")

    print(f"\nProtected Attributes Tested: {fairness_results.protected_attributes}")

    print(f"\n⚠️  Warnings ({len(fairness_results.warnings)}):")
    for warning in fairness_results.warnings:
        print(f"  - {warning}")

    print(f"\n🚨 Critical Issues ({len(fairness_results.critical_issues)}):")
    for issue in fairness_results.critical_issues:
        print(f"  - {issue}")

    # Detailed metrics per attribute
    print("\n" + "-"*70)
    print("DETALHAMENTO POR ATRIBUTO PROTEGIDO")
    print("-"*70)

    for attr in fairness_results.protected_attributes:
        print(f"\n📊 {attr.upper()}")
        attr_metrics = fairness_results.results['metrics'][attr]

        # Statistical Parity
        if 'statistical_parity' in attr_metrics:
            sp = attr_metrics['statistical_parity']
            print(f"\n  Statistical Parity:")
            print(f"    Ratio: {sp['ratio']:.3f} {'✅' if sp['passes_80_rule'] else '❌'}")
            print(f"    Group rates: {sp['group_rates']}")
            print(f"    Interpretation: {sp['interpretation']}")

        # Disparate Impact
        if 'disparate_impact' in attr_metrics:
            di = attr_metrics['disparate_impact']
            print(f"\n  Disparate Impact:")
            print(f"    Ratio: {di['ratio']:.3f} {'✅' if di['passes_threshold'] else '❌'}")
            print(f"    Threshold: {di['threshold']}")
            print(f"    Interpretation: {di['interpretation']}")

    print("\n" + "="*70)


def example_2_comparing_models():
    """
    Exemplo 2: Comparar fairness de diferentes modelos

    Demonstra:
    - Testar múltiplos modelos
    - Comparar fairness scores
    - Identificar modelo mais fair
    """
    print("\n\n")
    print("="*70)
    print("EXEMPLO 2: Comparando Fairness de Diferentes Modelos")
    print("="*70)

    # Create dataset
    df = create_synthetic_lending_dataset(n_samples=1000)

    # Prepare data
    X = df[['income', 'age', 'credit_score', 'employment_years', 'gender', 'race']].copy()
    X['gender'] = (X['gender'] == 'M').astype(int)
    X['race_encoded'] = pd.Categorical(X['race']).codes
    X_encoded = X[['income', 'age', 'credit_score', 'employment_years', 'gender', 'race_encoded']]
    X['race'] = df['race'].values  # Keep original for fairness testing
    y = df['loan_approved'].values

    # Models to compare
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results_comparison = {}

    for model_name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}")

        # Train model
        model.fit(X_encoded, y)
        acc = model.score(X_encoded, y)
        print(f"Accuracy: {acc:.4f}")

        # Create dataset
        dataset = DBDataset(features=X, target=y, model=model)

        # Run fairness tests
        experiment = Experiment(
            dataset=dataset,
            experiment_type="binary_classification",
            tests=["fairness"],
            protected_attributes=['gender', 'race']
        )

        fairness_results = experiment.run_fairness_tests(config='medium')

        results_comparison[model_name] = {
            'accuracy': acc,
            'fairness_score': fairness_results.overall_fairness_score,
            'critical_issues': len(fairness_results.critical_issues),
            'warnings': len(fairness_results.warnings)
        }

        print(f"Fairness Score: {fairness_results.overall_fairness_score:.3f}")
        print(f"Critical Issues: {len(fairness_results.critical_issues)}")

    # Compare results
    print("\n" + "="*70)
    print("COMPARAÇÃO FINAL")
    print("="*70)

    comparison_df = pd.DataFrame(results_comparison).T
    print("\n", comparison_df.to_string())

    # Identify best model
    best_fairness = comparison_df['fairness_score'].idxmax()
    print(f"\n🏆 Modelo mais fair: {best_fairness}")
    print(f"   Fairness Score: {comparison_df.loc[best_fairness, 'fairness_score']:.3f}")


def example_3_fairness_aware_preprocessing():
    """
    Exemplo 3: Demonstrar necessidade de preprocessing para fairness

    Mostra como remover protected attributes pode NÃO ser suficiente
    devido a proxy features (features correlacionadas com atributos protegidos).
    """
    print("\n\n")
    print("="*70)
    print("EXEMPLO 3: Fairness com e sem Protected Attributes")
    print("="*70)

    df = create_synthetic_lending_dataset(n_samples=1000)

    # Scenario 1: Train WITH protected attributes
    print("\n📊 Cenário 1: Treinar COM atributos protegidos")
    X1 = df[['income', 'age', 'credit_score', 'employment_years', 'gender', 'race']].copy()
    X1['gender'] = (X1['gender'] == 'M').astype(int)
    X1['race_encoded'] = pd.Categorical(X1['race']).codes
    X1_encoded = X1[['income', 'age', 'credit_score', 'employment_years', 'gender', 'race_encoded']]
    X1['race'] = df['race'].values
    y = df['loan_approved'].values

    model1 = LogisticRegression(random_state=42, max_iter=1000)
    model1.fit(X1_encoded, y)

    dataset1 = DBDataset(features=X1, target=y, model=model1)
    exp1 = Experiment(
        dataset=dataset1,
        experiment_type="binary_classification",
        tests=["fairness"],
        protected_attributes=['gender', 'race']
    )
    results1 = exp1.run_fairness_tests(config='quick')
    print(f"Fairness Score: {results1.overall_fairness_score:.3f}")
    print(f"Critical Issues: {len(results1.critical_issues)}")

    # Scenario 2: Train WITHOUT protected attributes
    print("\n📊 Cenário 2: Treinar SEM atributos protegidos")
    X2 = df[['income', 'age', 'credit_score', 'employment_years', 'gender', 'race']].copy()
    X2_encoded = X2[['income', 'age', 'credit_score', 'employment_years']]  # Remove protected
    X2['race'] = df['race'].values  # Keep for fairness testing

    model2 = LogisticRegression(random_state=42, max_iter=1000)
    model2.fit(X2_encoded, y)

    dataset2 = DBDataset(features=X2, target=y, model=model2)
    exp2 = Experiment(
        dataset=dataset2,
        experiment_type="binary_classification",
        tests=["fairness"],
        protected_attributes=['gender', 'race']
    )
    results2 = exp2.run_fairness_tests(config='quick')
    print(f"Fairness Score: {results2.overall_fairness_score:.3f}")
    print(f"Critical Issues: {len(results2.critical_issues)}")

    print("\n💡 Insight:")
    print("Mesmo removendo atributos protegidos, o modelo pode ter bias")
    print("devido a proxy features (features correlacionadas com atributos protegidos).")
    print("Ex: 'income' pode estar correlacionado com 'race' devido a desigualdades históricas.")


if __name__ == "__main__":
    print("\n")
    print("#"*70)
    print("# EXEMPLOS DE FAIRNESS TESTING - DeepBridge")
    print("#"*70)

    # Run examples
    example_1_basic_fairness_testing()
    example_2_comparing_models()
    example_3_fairness_aware_preprocessing()

    print("\n\n")
    print("="*70)
    print("EXEMPLOS CONCLUÍDOS!")
    print("="*70)
    print("\nPróximos passos:")
    print("1. Teste com seus próprios dados")
    print("2. Explore configs 'quick', 'medium', 'full'")
    print("3. Implemente técnicas de de-biasing se necessário")
    print("4. Documente resultados para compliance/auditoria")
    print("="*70)
