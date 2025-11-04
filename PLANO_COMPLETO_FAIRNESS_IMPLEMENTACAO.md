# 📋 PLANO COMPLETO - Implementação de Fairness Testing no DeepBridge

**Status:** Fase 1 ✅ Concluída | Fases 2-6 📋 Pendentes
**Última atualização:** 2025-11-03

---

## 🎯 Objetivo Geral

Implementar um módulo completo de testes de fairness na biblioteca DeepBridge, seguindo os padrões de Robustness, Uncertainty e Resilience, baseado nos arquivos de referência:
- `/simular_lib/analise_v4/run_analise_vies.py`
- `/simular_lib/analise_v4/analise_vies_fairness.py`
- `/simular_lib/analise_v4/analyze_predictions.py`

---

## ✅ FASE 1: Expandir Métricas Core [CONCLUÍDA]

**Tempo estimado:** 2-3h | **Tempo real:** ~2.5h
**Status:** ✅ 100% Concluída

### Implementações
- [x] 4 métricas pré-treino (BCL, BCO, KL, JS)
- [x] 7 métricas pós-treino (TFN, AC, RC, DP, DA, IT, IE)
- [x] 11 funções de interpretação com cores
- [x] Docstrings completos com fórmulas
- [x] Sistema de retorno padronizado
- [x] Script de testes completo
- [x] Documentação detalhada

### Arquivos Modificados
- `deepbridge/validation/fairness/metrics.py` (+977 linhas)
- `deepbridge/validation/fairness/__init__.py` (+29 linhas)

### Arquivos Criados
- `test_fairness_metrics_expanded.py`
- `FASE1_FAIRNESS_METRICAS_IMPLEMENTADAS.md`

### Resultado
15 métricas de fairness funcionais e testadas (4 pré-treino + 11 pós-treino)

---

## 📋 FASE 2: Expandir FairnessSuite [PENDENTE]

**Tempo estimado:** 3-4h
**Prioridade:** 🔴 ALTA

### Objetivos
Integrar as novas métricas ao wrapper `FairnessSuite` e adicionar funcionalidades avançadas.

### Tarefas

#### 2.1. Atualizar Templates de Configuração
```python
_CONFIG_TEMPLATES = {
    'quick': {
        'metrics': ['statistical_parity', 'disparate_impact'],
        'include_pretrain': False,
        'threshold_analysis': False
    },
    'medium': {
        'metrics': [
            'statistical_parity', 'equal_opportunity',
            'disparate_impact', 'precision_difference',
            'false_negative_rate_difference'
        ],
        'include_pretrain': True,
        'threshold_analysis': False
    },
    'full': {
        'metrics': [
            # Todas as 15 métricas
            'class_balance', 'concept_balance',
            'kl_divergence', 'js_divergence',
            'statistical_parity', 'equal_opportunity',
            'equalized_odds', 'disparate_impact',
            'false_negative_rate_difference',
            'conditional_acceptance', 'conditional_rejection',
            'precision_difference', 'accuracy_difference',
            'treatment_equality', 'entropy_index'
        ],
        'include_pretrain': True,
        'threshold_analysis': True
    }
}
```

#### 2.2. Adicionar Método de Matriz de Confusão
```python
def _calculate_confusion_matrix_by_group(
    self,
    y_true,
    y_pred,
    sensitive_feature
) -> Dict[str, Dict[str, int]]:
    """
    Calcula matriz de confusão detalhada para cada grupo.

    Returns:
        {
            'Group_A': {'TP': int, 'FP': int, 'TN': int, 'FN': int},
            'Group_B': {'TP': int, 'FP': int, 'TN': int, 'FN': int}
        }
    """
```

#### 2.3. Implementar Threshold Analysis
```python
def run_threshold_analysis(
    self,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    sensitive_feature: np.ndarray,
    thresholds: np.ndarray = np.arange(0.01, 1.0, 0.01),
    optimize_for: str = 'fairness'  # 'fairness', 'f1', 'balanced'
) -> Dict[str, Any]:
    """
    Analisa como métricas de fairness variam com diferentes thresholds.

    Returns:
        {
            'optimal_threshold': float,
            'optimal_metrics': Dict,
            'threshold_curve': DataFrame,
            'recommendations': List[str]
        }
    """
```

#### 2.4. Expandir Sistema de Warnings
```python
def _generate_warnings_and_critical_issues(self, results: Dict) -> Dict:
    """
    Atualizar para incluir as novas métricas:
    - Warnings para métricas em ⚠ Amarelo
    - Critical para métricas em ✗ Vermelho
    - Adicionar contexto e recomendações
    """
```

#### 2.5. Melhorar Overall Score
```python
def _calculate_overall_fairness_score(self, results: Dict) -> float:
    """
    Calcular score geral considerando:
    - Peso diferente para métricas críticas (EEOC)
    - Penalização maior para alertas vermelhos
    - Bônus para métricas verdes
    """
```

### Arquivos a Modificar
- `deepbridge/validation/wrappers/fairness_suite.py`

### Critérios de Sucesso
- [ ] Todas as 15 métricas disponíveis nos configs
- [ ] Flag `include_pretrain` funcionando
- [ ] Matriz de confusão detalhada por grupo
- [ ] Threshold analysis implementado
- [ ] Warnings/critical expandidos
- [ ] Testes passando

---

## 📊 FASE 3: Sistema de Visualizações [PENDENTE]

**Tempo estimado:** 2-3h
**Prioridade:** 🟡 MÉDIA

### Objetivos
Criar módulo de visualizações para análise visual de fairness.

### Tarefas

#### 3.1. Criar FairnessVisualizer
Novo arquivo: `deepbridge/validation/fairness/visualizations.py`

```python
class FairnessVisualizer:
    """Gerador de visualizações para análise de fairness"""

    @staticmethod
    def plot_distribution_by_group(
        df: pd.DataFrame,
        target_col: str,
        sensitive_feature: str,
        output_path: Optional[str] = None
    ) -> str:
        """Gráfico de distribuição do target por grupo"""

    @staticmethod
    def plot_metrics_comparison(
        metrics_results: Dict,
        protected_attrs: List[str],
        output_path: Optional[str] = None
    ) -> str:
        """Gráfico de barras comparando todas as métricas"""

    @staticmethod
    def plot_threshold_impact(
        threshold_results: pd.DataFrame,
        metrics: List[str] = ['statistical_parity', 'equal_opportunity'],
        output_path: Optional[str] = None
    ) -> str:
        """Curva mostrando impacto do threshold nas métricas"""

    @staticmethod
    def plot_confusion_matrices(
        cm_by_group: Dict[str, Dict],
        output_path: Optional[str] = None
    ) -> str:
        """Matrizes de confusão lado a lado para cada grupo"""

    @staticmethod
    def plot_fairness_radar(
        metrics_results: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """Radar chart com todas as métricas de fairness"""
```

#### 3.2. Gráficos Necessários
1. **Distribution Plot**: Distribuição de target por grupo
2. **Metrics Comparison**: Barras com todas as métricas
3. **Threshold Curves**: Impacto do threshold
4. **Confusion Matrices**: Matrizes lado a lado
5. **Radar Chart**: Visão geral de fairness
6. **Group Comparison**: Comparação detalhada entre grupos

### Arquivos a Criar
- `deepbridge/validation/fairness/visualizations.py`

### Critérios de Sucesso
- [ ] Classe FairnessVisualizer implementada
- [ ] 6 tipos de gráficos funcionando
- [ ] Integração com matplotlib/seaborn/plotly
- [ ] Salvar em PNG/SVG
- [ ] Testes visuais passando

---

## 📄 FASE 4: Geração de Relatórios [PENDENTE]

**Tempo estimado:** 2-3h
**Prioridade:** 🟡 MÉDIA

### Objetivos
Integrar fairness ao sistema de relatórios do DeepBridge.

### Tarefas

#### 4.1. Integrar com ReportManager
Modificar: `deepbridge/core/experiment/report/report_manager.py`

```python
def generate_fairness_report(
    self,
    results: Dict,
    output_path: str,
    format: str = 'html',
    include_excel: bool = True,
    include_visualizations: bool = True
) -> str:
    """
    Gera relatório completo de fairness.

    Args:
        format: 'html' ou 'pdf'
        include_excel: Se True, gera .xlsx adicional
        include_visualizations: Se True, gera gráficos
    """
```

#### 4.2. Template HTML Interativo
Criar: `deepbridge/templates/report_types/fairness/interactive/`

Estrutura similar a Robustness/Uncertainty:
```
fairness/
├── interactive/
│   ├── index.html
│   ├── css/
│   │   └── fairness-custom.css
│   ├── js/
│   │   ├── main.js
│   │   ├── controllers/
│   │   │   ├── overview.js
│   │   │   └── details.js
│   │   └── charts/
│   │       ├── overview.js
│   │       └── details.js
│   └── partials/
│       ├── overview.html
│       └── details.html
```

Seções do relatório:
1. **Executive Summary**
   - Overall Fairness Score
   - Critical Issues
   - Quick Stats

2. **Pre-Training Analysis**
   - Class Balance
   - Concept Balance
   - Distribution Divergences

3. **Post-Training Analysis**
   - Statistical Parity
   - Equal Opportunity
   - Todas as 11 métricas pós-treino

4. **Confusion Matrix Analysis**
   - Matrizes por grupo
   - Comparação de erros

5. **Threshold Analysis** (se disponível)
   - Curvas de impacto
   - Recomendação de threshold

6. **Recommendations**
   - Lista de ações sugeridas
   - Prioridades

#### 4.3. Relatório Excel
Criar: `deepbridge/core/experiment/report/renderers/fairness_excel_renderer.py`

Abas do Excel:
1. **Resumo Executivo**
2. **Métricas Pré-treino**
3. **Métricas Pós-treino**
4. **Matriz de Confusão**
5. **Threshold Analysis**
6. **Alertas e Recomendações**

### Arquivos a Criar/Modificar
- Criar template HTML fairness
- Criar renderer Excel
- Modificar `report_manager.py`

### Critérios de Sucesso
- [ ] Template HTML interativo funcionando
- [ ] Relatório Excel com 6 abas
- [ ] Visualizações incorporadas
- [ ] Sistema de alertas destacado
- [ ] Navegação entre seções
- [ ] Export PDF funcional

---

## 🔗 FASE 5: Integração com Experiment [PENDENTE]

**Tempo estimado:** 1-2h
**Prioridade:** 🟢 BAIXA

### Objetivos
Adicionar método `test_fairness()` ao `DBExperiment`.

### Tarefas

#### 5.1. Adicionar ao DBExperiment
Modificar: `deepbridge/core/experiment/experiment.py`

```python
def test_fairness(
    self,
    protected_attributes: List[str],
    privileged_groups: Optional[Dict[str, Any]] = None,
    config: Union[str, Dict] = 'full',
    generate_report: bool = True,
    output_path: Optional[str] = None,
    include_threshold_analysis: bool = True
) -> Dict[str, Any]:
    """
    Testa fairness do modelo em atributos protegidos.

    Args:
        protected_attributes: Lista de colunas sensíveis
        privileged_groups: Dict mapeando atributo -> valor privilegiado
        config: 'quick', 'medium', 'full' ou dict customizado
        generate_report: Se True, gera relatório HTML
        output_path: Caminho para salvar relatório
        include_threshold_analysis: Se True, analisa thresholds

    Returns:
        Dict com resultados completos de fairness

    Example:
        >>> exp.test_fairness(
        ...     protected_attributes=['gender', 'race', 'age'],
        ...     privileged_groups={
        ...         'gender': 'M',
        ...         'race': 'white',
        ...         'age': 'young'
        ...     },
        ...     config='full',
        ...     generate_report=True,
        ...     output_path='fairness_report.html'
        ... )
    """
```

#### 5.2. Integrar com ResultsManager
Adicionar suporte para salvar resultados de fairness:
```python
# Em deepbridge/core/experiment/results.py
def save_fairness_results(self, results: Dict, path: str):
    """Salva resultados de fairness testing"""
```

### Arquivos a Modificar
- `deepbridge/core/experiment/experiment.py`
- `deepbridge/core/experiment/results.py`

### Critérios de Sucesso
- [ ] Método `test_fairness()` funcionando
- [ ] Integração com FairnessSuite
- [ ] Auto-detecção de atributos sensíveis
- [ ] Geração automática de relatório
- [ ] Salvamento de resultados
- [ ] Testes end-to-end passando

---

## 📖 FASE 6: Documentação e Exemplos [PENDENTE]

**Tempo estimado:** 1-2h
**Prioridade:** 🟢 BAIXA

### Objetivos
Criar documentação completa e exemplos práticos.

### Tarefas

#### 6.1. Exemplo Completo
Criar: `examples/fairness_complete_example.py`

```python
"""
Exemplo completo de uso do módulo Fairness do DeepBridge.

Este exemplo demonstra:
1. Preparação de dados com atributos sensíveis
2. Treinamento de modelo
3. Análise de fairness completa
4. Geração de relatórios
5. Correção de viés (se necessário)
"""

from deepbridge import DBExperiment
from deepbridge.core.db_data import DBDataset
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 1. Carregar dados
df = pd.read_csv('data.csv')

# 2. Identificar atributos sensíveis
protected_attrs = ['gender', 'race', 'age_group']
privileged_groups = {
    'gender': 'M',
    'race': 'white',
    'age_group': 'young'
}

# 3. Separar features e target
X = df.drop(['target'] + protected_attrs, axis=1)
y = df['target']
sensitive_features = df[protected_attrs]

# 4. Treinar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 5. Criar dataset DeepBridge
dataset = DBDataset(
    features=X,
    target=y,
    model=model,
    experiment_type='classification',
    sensitive_features=sensitive_features  # Novo parâmetro
)

# 6. Criar experimento
exp = DBExperiment(dataset)

# 7. Testar fairness
fairness_results = exp.test_fairness(
    protected_attributes=protected_attrs,
    privileged_groups=privileged_groups,
    config='full',
    generate_report=True,
    output_path='fairness_report.html',
    include_threshold_analysis=True
)

# 8. Analisar resultados
print("Overall Fairness Score:", fairness_results['overall_score'])
print("\nCritical Issues:", fairness_results['critical_issues'])
print("\nWarnings:", fairness_results['warnings'])

# 9. Se houver problemas, analisar threshold ótimo
if fairness_results['threshold_analysis']:
    optimal = fairness_results['threshold_analysis']['optimal_threshold']
    print(f"\nThreshold ótimo para fairness: {optimal}")
```

#### 6.2. Exemplos Específicos
- `examples/fairness_pretrain_analysis.py` - Análise antes do treino
- `examples/fairness_threshold_tuning.py` - Otimização de threshold
- `examples/fairness_multi_attribute.py` - Múltiplos atributos sensíveis

#### 6.3. Documentação
- Atualizar README.md com seção de Fairness
- Criar guia de uso detalhado
- Documentar cada métrica com exemplos
- Adicionar FAQ sobre fairness

### Arquivos a Criar
- `examples/fairness_complete_example.py`
- `examples/fairness_pretrain_analysis.py`
- `examples/fairness_threshold_tuning.py`
- `examples/fairness_multi_attribute.py`
- `docs/fairness_guide.md`

### Critérios de Sucesso
- [ ] Exemplo completo funcionando
- [ ] 3+ exemplos específicos
- [ ] Documentação clara e completa
- [ ] FAQ respondendo dúvidas comuns
- [ ] README atualizado

---

## 📊 Resumo do Plano Completo

| Fase | Descrição | Tempo Est. | Prioridade | Status |
|------|-----------|------------|------------|--------|
| 1 | Expandir métricas core | 2-3h | 🔴 ALTA | ✅ CONCLUÍDA |
| 2 | Expandir FairnessSuite | 3-4h | 🔴 ALTA | 📋 Pendente |
| 3 | Sistema de visualizações | 2-3h | 🟡 MÉDIA | 📋 Pendente |
| 4 | Geração de relatórios | 2-3h | 🟡 MÉDIA | 📋 Pendente |
| 5 | Integração com Experiment | 1-2h | 🟢 BAIXA | 📋 Pendente |
| 6 | Documentação e exemplos | 1-2h | 🟢 BAIXA | 📋 Pendente |
| **TOTAL** | | **11-17h** | | **~15% Completo** |

---

## ✅ Checklist Geral de Validação

### Funcionalidades Core
- [x] 15 métricas de fairness implementadas
- [ ] FairnessSuite com todas as métricas
- [ ] Threshold analysis funcional
- [ ] Matriz de confusão detalhada
- [ ] Sistema de visualizações
- [ ] Relatórios HTML interativos
- [ ] Relatórios Excel completos
- [ ] Integração com DBExperiment

### Qualidade de Código
- [x] Docstrings completos
- [x] Type hints corretos
- [ ] Testes unitários (>80% cobertura)
- [ ] Testes de integração
- [ ] Exemplos funcionais
- [ ] Documentação completa

### Compatibilidade
- [x] Segue padrão de Robustness/Uncertainty/Resilience
- [ ] Compatível com DBDataset
- [ ] Compatível com DBExperiment
- [ ] Integrado ao sistema de relatórios
- [ ] Exportável em múltiplos formatos

---

## 🎯 Próximos Passos Recomendados

**Ordem sugerida de execução:**

1. **FASE 2** (Crítica)
   - Expandir FairnessSuite é essencial para usar as métricas
   - Permite testing end-to-end
   - Base para fases 3 e 4

2. **FASE 3** (Importante)
   - Visualizações melhoram muito a interpretabilidade
   - Necessário antes dos relatórios HTML

3. **FASE 4** (Importante)
   - Relatórios são o output final para usuários
   - Depende de visualizações

4. **FASE 5** (Integração)
   - Torna o módulo acessível via DBExperiment
   - API unificada com resto do DeepBridge

5. **FASE 6** (Polimento)
   - Documentação facilita adoção
   - Exemplos ajudam novos usuários

---

## 📚 Referências

### Arquivos de Origem
- `simular_lib/analise_v4/analise_vies_fairness.py`
- `simular_lib/analise_v4/run_analise_vies.py`
- `simular_lib/analise_v4/analyze_predictions.py`

### Frameworks de Referência
- **AI Fairness 360** (IBM)
- **Fairlearn** (Microsoft)
- **Aequitas** (University of Chicago)

### Regulamentações
- EEOC Uniform Guidelines (1978)
- GDPR Article 22 (Automated Decision-Making)
- Fair Lending Act
- Equal Credit Opportunity Act (ECOA)

---

**Última atualização:** 2025-11-03
**Autor:** Claude Code
**Status:** Documento Vivo - Atualizar conforme implementação progride
