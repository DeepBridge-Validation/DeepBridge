# Plano de Melhorias: Robustness Testing - DeepBridge

**Data**: 30 de Outubro de 2025
**Baseado em**: Análise Comparativa DeepBridge vs PiML-Toolbox
**Status**: 📋 Plano de Implementação

---

## 📊 Resumo Executivo

Baseado na análise comparativa com PiML-Toolbox, o DeepBridge **já possui 100% dos testes fundamentais de robustness**, mas pode ser significativamente aprimorado com **3 funcionalidades críticas**:

1. 🎯 **Fairness Testing** (Gap Crítico - Não implementado)
2. 🔍 **WeakSpot Detection Granular** (Gap Alto - Parcialmente implementado)
3. 📊 **Sliced Overfitting Analysis** (Gap Alto - Básico)

**Impacto Esperado**: Elevar DeepBridge de **90% para 100%** de paridade com PiML em robustness, com vantagens únicas mantidas (adversarial, custom perturbation).

---

## 🎯 PRIORIDADE 1: Fairness Testing Module

### Justificativa

**Crítico para**:
- Aplicações em Banking (regulações CFPB, ECOA)
- Healthcare (HIPAA compliance)
- Lending (Fair Lending Act)
- Insurance (discriminação por raça, gênero, idade)

**Impacto de não ter**:
- ❌ DeepBridge não pode ser usado em ambientes altamente regulados
- ❌ Perda de competitividade em setor financeiro
- ❌ Risco legal para empresas que usarem sem validação de fairness

### Funcionalidades a Implementar

#### 1.1 Métricas de Fairness

Implementar as **4 métricas principais**:

```python
# deepbridge/validation/fairness/metrics.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class FairnessMetrics:
    """
    Métricas de fairness para modelos de ML.
    Seguindo padrões de fairML, Aequitas, e AI Fairness 360.
    """

    @staticmethod
    def statistical_parity(y_pred, sensitive_feature):
        """
        Statistical Parity (Demographic Parity)

        Mede se a taxa de predições positivas é igual entre grupos.

        Formula:
            P(Y_hat=1 | A=a) = P(Y_hat=1 | A=b)

        Onde A é o atributo sensível (protected attribute).

        Returns:
            dict: {
                'metric_name': 'statistical_parity',
                'group_rates': Dict[str, float],  # Taxa por grupo
                'disparity': float,  # Máxima diferença entre grupos
                'ratio': float,  # Razão min/max (ideal: 1.0)
                'passes_80_rule': bool  # Ratio >= 0.8 (regra dos 80%)
            }
        """
        groups = np.unique(sensitive_feature)
        group_rates = {}

        for group in groups:
            mask = sensitive_feature == group
            positive_rate = np.mean(y_pred[mask] == 1)
            group_rates[str(group)] = positive_rate

        rates = list(group_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        disparity = max_rate - min_rate
        ratio = min_rate / max_rate if max_rate > 0 else 0.0
        passes_80_rule = ratio >= 0.8

        return {
            'metric_name': 'statistical_parity',
            'group_rates': group_rates,
            'disparity': disparity,
            'ratio': ratio,
            'passes_80_rule': passes_80_rule,
            'interpretation': _interpret_statistical_parity(disparity, ratio)
        }

    @staticmethod
    def equal_opportunity(y_true, y_pred, sensitive_feature):
        """
        Equal Opportunity

        Mede se a taxa de verdadeiros positivos (TPR) é igual entre grupos.
        Foca em garantir que o modelo identifica outcomes positivos igualmente.

        Formula:
            P(Y_hat=1 | Y=1, A=a) = P(Y_hat=1 | Y=1, A=b)

        Returns:
            dict: {
                'metric_name': 'equal_opportunity',
                'group_tpr': Dict[str, float],  # TPR por grupo
                'disparity': float,  # Máxima diferença em TPR
                'ratio': float  # Razão min/max TPR
            }
        """
        groups = np.unique(sensitive_feature)
        group_tpr = {}

        for group in groups:
            mask = (sensitive_feature == group) & (y_true == 1)
            if np.sum(mask) > 0:
                tpr = np.mean(y_pred[mask] == 1)
                group_tpr[str(group)] = tpr
            else:
                group_tpr[str(group)] = np.nan

        valid_tprs = [v for v in group_tpr.values() if not np.isnan(v)]
        if len(valid_tprs) > 1:
            max_tpr = max(valid_tprs)
            min_tpr = min(valid_tprs)
            disparity = max_tpr - min_tpr
            ratio = min_tpr / max_tpr if max_tpr > 0 else 0.0
        else:
            disparity = 0.0
            ratio = 1.0

        return {
            'metric_name': 'equal_opportunity',
            'group_tpr': group_tpr,
            'disparity': disparity,
            'ratio': ratio,
            'interpretation': _interpret_equal_opportunity(disparity)
        }

    @staticmethod
    def equalized_odds(y_true, y_pred, sensitive_feature):
        """
        Equalized Odds

        Mede se TPR E FPR são iguais entre grupos.
        Mais rigoroso que Equal Opportunity.

        Formula:
            P(Y_hat=1 | Y=y, A=a) = P(Y_hat=1 | Y=y, A=b) para y ∈ {0,1}

        Returns:
            dict: {
                'metric_name': 'equalized_odds',
                'group_tpr': Dict[str, float],
                'group_fpr': Dict[str, float],
                'tpr_disparity': float,
                'fpr_disparity': float,
                'combined_disparity': float  # max(tpr_disp, fpr_disp)
            }
        """
        groups = np.unique(sensitive_feature)
        group_tpr = {}
        group_fpr = {}

        for group in groups:
            # TPR
            mask_pos = (sensitive_feature == group) & (y_true == 1)
            if np.sum(mask_pos) > 0:
                tpr = np.mean(y_pred[mask_pos] == 1)
                group_tpr[str(group)] = tpr
            else:
                group_tpr[str(group)] = np.nan

            # FPR
            mask_neg = (sensitive_feature == group) & (y_true == 0)
            if np.sum(mask_neg) > 0:
                fpr = np.mean(y_pred[mask_neg] == 1)
                group_fpr[str(group)] = fpr
            else:
                group_fpr[str(group)] = np.nan

        # Calcular disparities
        valid_tprs = [v for v in group_tpr.values() if not np.isnan(v)]
        valid_fprs = [v for v in group_fpr.values() if not np.isnan(v)]

        tpr_disparity = max(valid_tprs) - min(valid_tprs) if len(valid_tprs) > 1 else 0.0
        fpr_disparity = max(valid_fprs) - min(valid_fprs) if len(valid_fprs) > 1 else 0.0
        combined_disparity = max(tpr_disparity, fpr_disparity)

        return {
            'metric_name': 'equalized_odds',
            'group_tpr': group_tpr,
            'group_fpr': group_fpr,
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'combined_disparity': combined_disparity,
            'interpretation': _interpret_equalized_odds(tpr_disparity, fpr_disparity)
        }

    @staticmethod
    def disparate_impact(y_pred, sensitive_feature, threshold=0.8):
        """
        Disparate Impact Ratio

        Razão entre taxa de seleção do grupo menos favorecido e mais favorecido.
        Regulação: Razão < 0.8 é considerada evidência de discriminação (EEOC).

        Formula:
            DI = P(Y_hat=1 | A=unprivileged) / P(Y_hat=1 | A=privileged)

        Returns:
            dict: {
                'metric_name': 'disparate_impact',
                'ratio': float,  # Razão (ideal: 1.0)
                'passes_threshold': bool,  # >= 0.8
                'unprivileged_rate': float,
                'privileged_rate': float,
                'groups': Dict[str, float]  # Taxa por grupo
            }
        """
        groups = np.unique(sensitive_feature)
        group_rates = {}

        for group in groups:
            mask = sensitive_feature == group
            positive_rate = np.mean(y_pred[mask] == 1)
            group_rates[str(group)] = positive_rate

        rates = list(group_rates.values())
        min_rate = min(rates)
        max_rate = max(rates)

        ratio = min_rate / max_rate if max_rate > 0 else 0.0
        passes_threshold = ratio >= threshold

        return {
            'metric_name': 'disparate_impact',
            'ratio': ratio,
            'threshold': threshold,
            'passes_threshold': passes_threshold,
            'unprivileged_rate': min_rate,
            'privileged_rate': max_rate,
            'groups': group_rates,
            'interpretation': _interpret_disparate_impact(ratio, threshold)
        }


def _interpret_statistical_parity(disparity, ratio):
    """Interpretação textual de statistical parity"""
    if disparity < 0.01:
        return "Excelente: Paridade estatística quase perfeita"
    elif ratio >= 0.8:
        return "Bom: Passa na regra dos 80% (EEOC compliant)"
    elif ratio >= 0.6:
        return "Moderado: Alguma disparidade presente"
    else:
        return "CRÍTICO: Disparidade significativa detectada - requer investigação"


def _interpret_equal_opportunity(disparity):
    """Interpretação textual de equal opportunity"""
    if disparity < 0.05:
        return "Excelente: TPR equilibrado entre grupos"
    elif disparity < 0.1:
        return "Bom: Pequena diferença em TPR"
    elif disparity < 0.2:
        return "Moderado: Diferença notável em TPR"
    else:
        return "CRÍTICO: Diferença significativa em TPR - grupo desfavorecido"


def _interpret_equalized_odds(tpr_disp, fpr_disp):
    """Interpretação textual de equalized odds"""
    max_disp = max(tpr_disp, fpr_disp)
    if max_disp < 0.05:
        return "Excelente: TPR e FPR equilibrados entre grupos"
    elif max_disp < 0.1:
        return "Bom: Pequenas diferenças em TPR/FPR"
    elif max_disp < 0.2:
        return "Moderado: Diferenças notáveis em TPR/FPR"
    else:
        return "CRÍTICO: Diferenças significativas em TPR/FPR"


def _interpret_disparate_impact(ratio, threshold):
    """Interpretação textual de disparate impact"""
    if ratio >= 0.95:
        return "Excelente: Impacto quase igual entre grupos"
    elif ratio >= threshold:
        return f"Bom: Passa no threshold {threshold} (EEOC compliant)"
    elif ratio >= 0.6:
        return f"Moderado: Abaixo do threshold {threshold} - requer atenção"
    else:
        return "CRÍTICO: Disparate impact significativo - risco legal alto"
```

#### 1.2 FairnessSuite - Suite Completa

```python
# deepbridge/validation/wrappers/fairness_suite.py

from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from deepbridge.validation.fairness.metrics import FairnessMetrics

class FairnessSuite:
    """
    Suite completa para testes de fairness em modelos de ML.

    Integra-se com o framework DeepBridge existente.
    """

    def __init__(self,
                 dataset,
                 protected_attributes: List[str],
                 privileged_groups: Optional[Dict[str, Any]] = None,
                 verbose: bool = False):
        """
        Initialize fairness testing suite.

        Parameters:
        -----------
        dataset : DBDataset
            Dataset do DeepBridge
        protected_attributes : List[str]
            Lista de atributos protegidos (ex: ['gender', 'race', 'age'])
        privileged_groups : Dict[str, Any], optional
            Definição de grupos privilegiados (ex: {'gender': 'male', 'race': 'white'})
        verbose : bool
            Print progress information
        """
        self.dataset = dataset
        self.protected_attributes = protected_attributes
        self.privileged_groups = privileged_groups or {}
        self.verbose = verbose
        self.metrics_calculator = FairnessMetrics()

        # Validar que protected_attributes existem no dataset
        X = self.dataset.get_feature_data()
        for attr in protected_attributes:
            if attr not in X.columns:
                raise ValueError(f"Protected attribute '{attr}' not found in dataset")

    def run(self, config='full') -> Dict[str, Any]:
        """
        Execute fairness tests.

        Parameters:
        -----------
        config : str
            Configuration level: 'quick', 'medium', 'full'

        Returns:
        --------
        Dict with fairness test results
        """
        if self.verbose:
            print(f"Running fairness tests with config: {config}")

        # Get data
        X = self.dataset.get_feature_data()
        y_true = self.dataset.get_target_data()

        # Get predictions
        if hasattr(self.dataset, 'model') and self.dataset.model is not None:
            y_pred = self.dataset.model.predict(X)
        else:
            raise ValueError("No model found in dataset. Cannot compute predictions.")

        results = {
            'protected_attributes': self.protected_attributes,
            'metrics': {},
            'overall_fairness_score': 0.0,
            'warnings': [],
            'critical_issues': []
        }

        # Run tests for each protected attribute
        for attr in self.protected_attributes:
            if self.verbose:
                print(f"\nTesting fairness for protected attribute: {attr}")

            sensitive_feature = X[attr].values
            attr_results = {}

            # 1. Statistical Parity
            sp = self.metrics_calculator.statistical_parity(y_pred, sensitive_feature)
            attr_results['statistical_parity'] = sp
            if not sp['passes_80_rule']:
                results['warnings'].append(
                    f"{attr}: Falha na regra dos 80% (ratio={sp['ratio']:.3f})"
                )

            # 2. Equal Opportunity
            eo = self.metrics_calculator.equal_opportunity(y_true, y_pred, sensitive_feature)
            attr_results['equal_opportunity'] = eo
            if eo['disparity'] > 0.1:
                results['warnings'].append(
                    f"{attr}: Disparidade em Equal Opportunity (disp={eo['disparity']:.3f})"
                )

            # 3. Equalized Odds (apenas em 'full')
            if config == 'full':
                eq_odds = self.metrics_calculator.equalized_odds(y_true, y_pred, sensitive_feature)
                attr_results['equalized_odds'] = eq_odds
                if eq_odds['combined_disparity'] > 0.1:
                    results['warnings'].append(
                        f"{attr}: Disparidade em Equalized Odds (disp={eq_odds['combined_disparity']:.3f})"
                    )

            # 4. Disparate Impact
            di = self.metrics_calculator.disparate_impact(y_pred, sensitive_feature)
            attr_results['disparate_impact'] = di
            if not di['passes_threshold']:
                results['critical_issues'].append(
                    f"{attr}: Disparate Impact CRÍTICO (ratio={di['ratio']:.3f} < 0.8)"
                )

            results['metrics'][attr] = attr_results

        # Calculate overall fairness score
        results['overall_fairness_score'] = self._calculate_fairness_score(results['metrics'])

        # Summary
        results['summary'] = {
            'total_attributes_tested': len(self.protected_attributes),
            'attributes_with_warnings': len(results['warnings']),
            'critical_issues_found': len(results['critical_issues']),
            'overall_assessment': self._assess_fairness(results['overall_fairness_score'])
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"FAIRNESS ASSESSMENT")
            print(f"{'='*60}")
            print(f"Overall Fairness Score: {results['overall_fairness_score']:.3f}")
            print(f"Assessment: {results['summary']['overall_assessment']}")
            if results['critical_issues']:
                print(f"\n⚠️  CRITICAL ISSUES ({len(results['critical_issues'])}):")
                for issue in results['critical_issues']:
                    print(f"   - {issue}")

        return results

    def _calculate_fairness_score(self, metrics: Dict) -> float:
        """
        Calculate overall fairness score (0-1, higher is better).

        Combina múltiplas métricas em um score único.
        """
        scores = []

        for attr, attr_metrics in metrics.items():
            # Statistical Parity (peso: 0.3)
            sp_score = attr_metrics['statistical_parity']['ratio']
            scores.append(sp_score * 0.3)

            # Equal Opportunity (peso: 0.3)
            eo_disparity = attr_metrics['equal_opportunity']['disparity']
            eo_score = max(0, 1 - eo_disparity)
            scores.append(eo_score * 0.3)

            # Disparate Impact (peso: 0.4 - mais crítico)
            di_score = attr_metrics['disparate_impact']['ratio']
            scores.append(di_score * 0.4)

        return np.mean(scores) if scores else 0.0

    def _assess_fairness(self, score: float) -> str:
        """Assess fairness level based on score"""
        if score >= 0.95:
            return "EXCELENTE - Modelo altamente fair"
        elif score >= 0.85:
            return "BOM - Fairness adequada"
        elif score >= 0.70:
            return "MODERADO - Requer atenção"
        else:
            return "CRÍTICO - Intervenção necessária"

    def save_report(self, results: Dict, output_path: str):
        """
        Save fairness report to HTML.

        Similar ao save_report de outros testes.
        """
        # TODO: Implementar rendering HTML
        pass
```

#### 1.3 Integração com Experiment

```python
# deepbridge/core/experiment/experiment.py

# Adicionar ao __init__:
def __init__(self, ..., protected_attributes: Optional[List[str]] = None):
    # ... código existente ...
    self.protected_attributes = protected_attributes

    # Se protected_attributes fornecidos, inicializar FairnessSuite
    if protected_attributes:
        from deepbridge.validation.wrappers.fairness_suite import FairnessSuite
        self.fairness_suite = FairnessSuite(
            dataset=dataset,
            protected_attributes=protected_attributes,
            verbose=self.verbose
        )

# Adicionar método run_fairness_tests:
def run_fairness_tests(self, config='full'):
    """
    Run fairness tests if protected attributes were provided.

    Returns:
        FairnessResult object with test results
    """
    if not hasattr(self, 'fairness_suite'):
        raise ValueError(
            "No protected attributes provided. Initialize Experiment with "
            "protected_attributes=['attr1', 'attr2', ...] to enable fairness testing."
        )

    results = self.fairness_suite.run(config=config)

    # Wrap results in FairnessResult
    from deepbridge.core.experiment.results import FairnessResult
    return FairnessResult(results)
```

### Exemplo de Uso

```python
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment import Experiment

# Criar dataset
dataset = DBDataset(
    features=X,
    target=y,
    model=trained_model
)

# Criar experimento COM fairness testing
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["robustness", "fairness"],  # ← Novo teste
    protected_attributes=['gender', 'race', 'age']  # ← Novo parâmetro
)

# Executar fairness tests
fairness_results = experiment.run_fairness_tests(config='full')

# Ver resultados
print(f"Fairness Score: {fairness_results.overall_fairness_score:.3f}")
print(f"Critical Issues: {len(fairness_results.critical_issues)}")

# Salvar report
fairness_results.save_html('fairness_report.html')
```

---

## 🔍 PRIORIDADE 2: WeakSpot Detection Granular

### Justificativa

**Problema**: Modelos podem ter boa performance global mas falhar em **regiões específicas** do espaço de features.

**Exemplos Reais**:
- Modelo de credit scoring funciona bem para renda $30k-$100k, mas falha para renda extrema (< $10k ou > $200k)
- Modelo de diagnóstico médico preciso para pacientes 30-60 anos, mas ruim para crianças e idosos

### Funcionalidades a Implementar

#### 2.1 Slice Detection Automática

```python
# deepbridge/validation/robustness/weakspot_detector.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.tree import DecisionTreeRegressor

class WeakspotDetector:
    """
    Detecta regiões (slices) do espaço de features com degradação de performance.

    Usa técnicas de slicing para identificar automaticamente regiões fracas.
    """

    def __init__(self,
                 slice_method: str = 'uniform',
                 n_slices: int = 10,
                 min_samples_per_slice: int = 30,
                 severity_threshold: float = 0.1):
        """
        Parameters:
        -----------
        slice_method : str
            Método de slicing: 'uniform', 'quantile', 'tree-based', 'adaptive'
        n_slices : int
            Número de slices por feature
        min_samples_per_slice : int
            Mínimo de amostras necessário em um slice para ser considerado
        severity_threshold : float
            Threshold de degradação para considerar um slice como "fraco"
        """
        self.slice_method = slice_method
        self.n_slices = n_slices
        self.min_samples_per_slice = min_samples_per_slice
        self.severity_threshold = severity_threshold

    def detect_weak_regions(self,
                           X: pd.DataFrame,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           slice_features: Optional[List[str]] = None,
                           metric: str = 'mae') -> Dict[str, Any]:
        """
        Identifica regiões fracas do modelo.

        Parameters:
        -----------
        X : DataFrame
            Features
        y_true : array
            True labels
        y_pred : array
            Predicted labels
        slice_features : List[str], optional
            Features específicas para analisar (None = todas)
        metric : str
            Métrica para avaliar weakness: 'mae', 'mse', 'residual', 'error_rate'

        Returns:
        --------
        Dict com informações sobre regiões fracas:
        {
            'weakspots': List[Dict],  # Lista de regiões fracas encontradas
            'summary': Dict,  # Resumo estatístico
            'slice_analysis': Dict  # Análise detalhada por feature
        }
        """
        if slice_features is None:
            slice_features = X.columns.tolist()

        # Calcular resíduos/erros
        residuals = self._calculate_residuals(y_true, y_pred, metric)

        weakspots = []
        slice_analysis = {}

        for feature in slice_features:
            # Criar slices para essa feature
            slices = self._create_slices(X[feature].values, method=self.slice_method)

            feature_analysis = {
                'feature': feature,
                'slices': [],
                'worst_slice': None,
                'best_slice': None
            }

            for slice_idx, (slice_range, slice_mask) in enumerate(slices):
                n_samples = np.sum(slice_mask)

                # Pular slices com poucas amostras
                if n_samples < self.min_samples_per_slice:
                    continue

                # Calcular métricas do slice
                slice_residuals = residuals[slice_mask]
                slice_mean_residual = np.mean(np.abs(slice_residuals))
                slice_std_residual = np.std(slice_residuals)

                # Global baseline
                global_mean_residual = np.mean(np.abs(residuals))

                # Severity: quanto pior que global
                severity = (slice_mean_residual - global_mean_residual) / global_mean_residual

                slice_info = {
                    'slice_idx': slice_idx,
                    'feature': feature,
                    'range': slice_range,
                    'n_samples': n_samples,
                    'mean_residual': slice_mean_residual,
                    'std_residual': slice_std_residual,
                    'severity': severity,
                    'is_weak': severity > self.severity_threshold
                }

                feature_analysis['slices'].append(slice_info)

                # Identificar se é weakspot
                if slice_info['is_weak']:
                    weakspots.append(slice_info)

            # Encontrar worst e best slices
            if feature_analysis['slices']:
                feature_analysis['worst_slice'] = max(
                    feature_analysis['slices'],
                    key=lambda s: s['severity']
                )
                feature_analysis['best_slice'] = min(
                    feature_analysis['slices'],
                    key=lambda s: s['severity']
                )

            slice_analysis[feature] = feature_analysis

        # Ordenar weakspots por severidade
        weakspots = sorted(weakspots, key=lambda w: w['severity'], reverse=True)

        # Summary
        summary = {
            'total_weakspots': len(weakspots),
            'features_with_weakspots': len(set(w['feature'] for w in weakspots)),
            'avg_severity': np.mean([w['severity'] for w in weakspots]) if weakspots else 0.0,
            'max_severity': max([w['severity'] for w in weakspots]) if weakspots else 0.0,
            'critical_weakspots': len([w for w in weakspots if w['severity'] > 0.5])
        }

        return {
            'weakspots': weakspots,
            'summary': summary,
            'slice_analysis': slice_analysis,
            'global_mean_residual': global_mean_residual
        }

    def _calculate_residuals(self, y_true, y_pred, metric):
        """Calcula resíduos baseado na métrica"""
        if metric == 'mae' or metric == 'residual':
            return y_true - y_pred
        elif metric == 'mse':
            return (y_true - y_pred) ** 2
        elif metric == 'error_rate':
            return (y_true != y_pred).astype(int)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _create_slices(self, feature_values, method='uniform'):
        """
        Cria slices de uma feature.

        Returns:
            List of (slice_range, slice_mask) tuples
        """
        if method == 'uniform':
            return self._uniform_slices(feature_values)
        elif method == 'quantile':
            return self._quantile_slices(feature_values)
        elif method == 'tree-based':
            return self._tree_based_slices(feature_values)
        else:
            raise ValueError(f"Unknown slice method: {method}")

    def _uniform_slices(self, feature_values):
        """Slices com bins uniformes"""
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        bin_edges = np.linspace(min_val, max_val, self.n_slices + 1)

        slices = []
        for i in range(self.n_slices):
            slice_range = (bin_edges[i], bin_edges[i+1])
            if i == self.n_slices - 1:  # Last slice includes upper edge
                slice_mask = (feature_values >= slice_range[0]) & (feature_values <= slice_range[1])
            else:
                slice_mask = (feature_values >= slice_range[0]) & (feature_values < slice_range[1])

            slices.append((slice_range, slice_mask))

        return slices

    def _quantile_slices(self, feature_values):
        """Slices com quantis (bins com igual número de amostras)"""
        quantiles = np.linspace(0, 1, self.n_slices + 1)
        bin_edges = np.quantile(feature_values, quantiles)

        # Similar ao uniform
        slices = []
        for i in range(self.n_slices):
            slice_range = (bin_edges[i], bin_edges[i+1])
            if i == self.n_slices - 1:
                slice_mask = (feature_values >= slice_range[0]) & (feature_values <= slice_range[1])
            else:
                slice_mask = (feature_values >= slice_range[0]) & (feature_values < slice_range[1])

            slices.append((slice_range, slice_mask))

        return slices

    def _tree_based_slices(self, feature_values):
        """
        Slices adaptativos usando árvore de decisão.

        Encontra automaticamente regiões naturais de separação.
        """
        # TODO: Implementar usando DecisionTreeRegressor para encontrar splits ótimos
        # Esta é uma abordagem mais sofisticada que requer residuals
        pass

    def visualize_weakspots(self, weakspot_results: Dict) -> Dict[str, Any]:
        """
        Gera visualizações dos weakspots.

        Returns:
            Dict com Plotly figures para cada feature
        """
        # TODO: Implementar visualizações com Plotly
        # Heatmap de severidade por slice
        # Bar chart de top weakspots
        # Line plot de residual por slice
        pass
```

#### 2.2 Integração com RobustnessSuite

```python
# deepbridge/validation/wrappers/robustness_suite.py

# Adicionar ao run():
def run(self, X=None, y=None):
    # ... código existente ...

    # NOVO: WeakSpot Detection ao final
    if self.verbose:
        print("\n🔍 Running WeakSpot Detection...")

    from deepbridge.validation.robustness.weakspot_detector import WeakspotDetector

    weakspot_detector = WeakspotDetector(
        slice_method='uniform',
        n_slices=10,
        severity_threshold=0.1
    )

    # Get predictions
    y_pred = self.dataset.model.predict(X)

    # Detect weakspots
    weakspot_results = weakspot_detector.detect_weak_regions(
        X=X,
        y_true=y,
        y_pred=y_pred,
        slice_features=self.feature_subset,
        metric='mae'
    )

    results['weakspot_analysis'] = weakspot_results

    if self.verbose:
        print(f"   Found {weakspot_results['summary']['total_weakspots']} weakspots")
        if weakspot_results['summary']['critical_weakspots'] > 0:
            print(f"   ⚠️  {weakspot_results['summary']['critical_weakspots']} CRITICAL weakspots!")

    return results
```

---

## 📊 PRIORIDADE 3: Sliced Overfitting Analysis

### Justificativa

**Problema**: Overfitting pode ser **local**, não global.

**Exemplo**:
- Modelo tem train AUC = 0.95, test AUC = 0.93 (parece OK)
- Mas para feature "Income" > $150k: train AUC = 0.98, test AUC = 0.75 (OVERFITTING CRÍTICO!)

### Implementação

```python
# deepbridge/validation/robustness/overfit_analyzer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

class OverfitAnalyzer:
    """
    Analisa overfitting em slices específicos do dataset.

    Identifica regiões onde train-test gap é alto.
    """

    def __init__(self,
                 n_slices: int = 10,
                 slice_method: str = 'quantile',
                 gap_threshold: float = 0.1):
        """
        Parameters:
        -----------
        n_slices : int
            Número de slices por feature
        slice_method : str
            Método de slicing: 'uniform', 'quantile'
        gap_threshold : float
            Threshold para considerar gap significativo
        """
        self.n_slices = n_slices
        self.slice_method = slice_method
        self.gap_threshold = gap_threshold

    def compute_gap_by_slice(self,
                            X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: np.ndarray,
                            y_test: np.ndarray,
                            model,
                            slice_feature: str,
                            metric_func) -> Dict[str, Any]:
        """
        Calcula train-test gap por slices de uma feature.

        Parameters:
        -----------
        X_train, X_test : DataFrame
            Train e test features
        y_train, y_test : array
            Train e test labels
        model : fitted model
            Modelo treinado
        slice_feature : str
            Feature para fazer slicing
        metric_func : callable
            Função que calcula métrica: metric_func(y_true, y_pred) -> float

        Returns:
        --------
        Dict com análise de gaps:
        {
            'feature': str,
            'slices': List[Dict],  # Gap por slice
            'max_gap': float,
            'avg_gap': float,
            'overfit_slices': List[Dict]  # Slices com overfit
        }
        """
        # Criar slices
        train_slices = self._create_slices(X_train[slice_feature].values)
        test_slices = self._create_slices(X_test[slice_feature].values)

        slice_results = []
        overfit_slices = []

        for slice_idx, ((train_range, train_mask), (test_range, test_mask)) in enumerate(
            zip(train_slices, test_slices)
        ):
            # Skip se muito poucas amostras
            if np.sum(train_mask) < 10 or np.sum(test_mask) < 10:
                continue

            # Train metrics
            X_train_slice = X_train[train_mask]
            y_train_slice = y_train[train_mask]
            y_train_pred = model.predict(X_train_slice)
            train_metric = metric_func(y_train_slice, y_train_pred)

            # Test metrics
            X_test_slice = X_test[test_mask]
            y_test_slice = y_test[test_mask]
            y_test_pred = model.predict(X_test_slice)
            test_metric = metric_func(y_test_slice, y_test_pred)

            # Gap
            gap = train_metric - test_metric

            slice_info = {
                'slice_idx': slice_idx,
                'range': train_range,
                'train_samples': np.sum(train_mask),
                'test_samples': np.sum(test_mask),
                'train_metric': train_metric,
                'test_metric': test_metric,
                'gap': gap,
                'is_overfitting': gap > self.gap_threshold
            }

            slice_results.append(slice_info)

            if slice_info['is_overfitting']:
                overfit_slices.append(slice_info)

        # Summary
        gaps = [s['gap'] for s in slice_results]

        return {
            'feature': slice_feature,
            'slices': slice_results,
            'max_gap': max(gaps) if gaps else 0.0,
            'avg_gap': np.mean(gaps) if gaps else 0.0,
            'std_gap': np.std(gaps) if gaps else 0.0,
            'overfit_slices': overfit_slices,
            'summary': {
                'total_slices': len(slice_results),
                'overfit_slices_count': len(overfit_slices),
                'overfit_percentage': len(overfit_slices) / len(slice_results) * 100 if slice_results else 0
            }
        }

    def _create_slices(self, feature_values):
        """Cria slices (similar ao WeakspotDetector)"""
        if self.slice_method == 'quantile':
            quantiles = np.linspace(0, 1, self.n_slices + 1)
            bin_edges = np.quantile(feature_values, quantiles)
        else:  # uniform
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            bin_edges = np.linspace(min_val, max_val, self.n_slices + 1)

        slices = []
        for i in range(self.n_slices):
            slice_range = (bin_edges[i], bin_edges[i+1])
            if i == self.n_slices - 1:
                slice_mask = (feature_values >= slice_range[0]) & (feature_values <= slice_range[1])
            else:
                slice_mask = (feature_values >= slice_range[0]) & (feature_values < slice_range[1])

            slices.append((slice_range, slice_mask))

        return slices
```

---

## 📈 Melhorias Adicionais (Médio Prazo)

### 4. Fine-grained Perturbation Steps

**Mudança Simples**: Adicionar configuração com steps de 0.05

```python
# deepbridge/core/experiment/parameter_standards.py

ROBUSTNESS_CONFIGS = {
    # ... configs existentes ...

    'fine_grained': {
        'perturbation_methods': ['raw', 'quantile'],
        'levels': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],  # Steps de 0.05
        'n_trials': 5
    }
}
```

### 5. Local Robustness Scoring

```python
# deepbridge/validation/robustness/local_robustness.py

def compute_local_robustness_scores(X, model, perturber, n_iterations=10):
    """
    Calcula score de vulnerabilidade para cada data point.

    Returns:
        scores: Array com score de robustness por sample (0-1, higher = more robust)
    """
    scores = []

    for i in range(len(X)):
        sample = X.iloc[i:i+1]
        original_pred = model.predict(sample)[0]

        # Perturbar múltiplas vezes
        pred_changes = []
        for _ in range(n_iterations):
            perturbed = perturber.perturb(sample, level=0.1)
            perturbed_pred = model.predict(perturbed)[0]
            change = abs(original_pred - perturbed_pred)
            pred_changes.append(change)

        # Score: inversamente proporcional à mudança média
        avg_change = np.mean(pred_changes)
        robustness_score = 1 / (1 + avg_change)
        scores.append(robustness_score)

    return np.array(scores)
```

---

## 🎯 Roadmap de Implementação

### Fase 1: Fairness (2-3 semanas)
**Semana 1**:
- [ ] Implementar `FairnessMetrics` (4 métricas)
- [ ] Testes unitários para métricas
- [ ] Validação com datasets conhecidos (Adult, COMPAS)

**Semana 2**:
- [ ] Implementar `FairnessSuite`
- [ ] Integração com `Experiment`
- [ ] Testes de integração

**Semana 3**:
- [ ] HTML report rendering
- [ ] Visualizações (bar charts, heatmaps)
- [ ] Documentação e exemplos

### Fase 2: WeakSpot Detection (2 semanas)
**Semana 1**:
- [ ] Implementar `WeakspotDetector`
- [ ] Métodos de slicing (uniform, quantile)
- [ ] Integração com `RobustnessSuite`

**Semana 2**:
- [ ] Visualizações de weakspots
- [ ] Testes e validação
- [ ] Documentação

### Fase 3: Sliced Overfitting (1 semana)
- [ ] Implementar `OverfitAnalyzer`
- [ ] Integração com teste de robustness
- [ ] Visualizações de gaps
- [ ] Testes e documentação

### Fase 4: Melhorias Adicionais (1-2 semanas)
- [ ] Fine-grained steps
- [ ] Local robustness scoring
- [ ] Otimizações de performance

---

## 📊 Métricas de Sucesso

### Após Implementação, DeepBridge deve:

1. ✅ **Fairness Compliance**
   - Suportar as 4 métricas principais de fairness
   - Atender regulações EEOC (regra dos 80%)
   - Gerar reports compliant para auditoria

2. ✅ **Diagnostic Depth**
   - Identificar weakspots automaticamente
   - Detectar overfitting localizado
   - Scores de vulnerabilidade por sample

3. ✅ **Paridade com PiML**
   - 100% de paridade em funcionalidades core
   - Vantagens mantidas (adversarial, custom)
   - Melhor structured output

4. ✅ **Usabilidade**
   - APIs consistentes com testes existentes
   - Documentação clara
   - Exemplos de uso real

---

## 🔄 Comparação Final (Após Implementação)

| Funcionalidade | Antes | Depois | Status |
|---------------|-------|---------|--------|
| **Fairness Testing** | ❌ | ✅ | NOVA |
| **WeakSpot Detection** | ⚠️ | ✅ | MELHORADA |
| **Sliced Overfit** | ⚠️ | ✅ | MELHORADA |
| **Fine-grained Steps** | ⚠️ | ✅ | MELHORADA |
| **Local Robustness** | ❌ | ✅ | NOVA |
| **Paridade PiML** | 90% | **100%** | ✅ |

---

## 📚 Recursos Adicionais

### Referências de Implementação

1. **Fairness Metrics**:
   - AI Fairness 360 (IBM): https://github.com/Trusted-AI/AIF360
   - Fairlearn (Microsoft): https://github.com/fairlearn/fairlearn
   - Aequitas: https://github.com/dssg/aequitas

2. **WeakSpot Detection**:
   - Slice Finder (Google): https://github.com/google/sliceline
   - Spotlight (Microsoft): https://github.com/interpretml/interpret

3. **Overfitting Analysis**:
   - PiML-Toolbox: https://github.com/SelfExplainML/PiML-Toolbox

### Datasets para Validação

1. **Fairness**: Adult Income, COMPAS, German Credit
2. **WeakSpot**: UCI datasets com features contínuas
3. **Overfitting**: Datasets com known distribution shifts

---

## 🎓 Conclusão

Com estas **3 melhorias prioritárias**, DeepBridge alcançará:

1. ✅ **100% de paridade** com PiML em robustness testing
2. ✅ **Compliance** com regulações de fairness (banking, healthcare)
3. ✅ **Diagnóstico avançado** de fraquezas e overfitting localizado
4. ✅ **Diferenciação competitiva** mantida (adversarial, custom methods)

**Esforço Total Estimado**: 6-8 semanas para as 3 prioridades principais.

**ROI Esperado**: Habilitar DeepBridge para uso em **ambientes altamente regulados** (banking, healthcare, insurance) onde fairness é mandatório.

---

**Documento criado em**: 30/10/2025
**Para**: Gustavo Haase - DeepBridge Framework
**Próximo Passo**: Iniciar Fase 1 (Fairness Testing Module)
