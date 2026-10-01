"""
Domain models for Robustness reports (Phase 3 Sprint 10.3).

Type-safe Pydantic models for robustness experiment results.

Benefits:
- Eliminates .get() calls
- Type safety with validation
- IDE autocomplete support
- Clear data contracts
"""

from typing import Dict, List, Optional

from pydantic import Field

from ..data.base import SCORE_MAX
from .base import ReportBaseModel

# ---------------------------------------------------------------------------
# Contrato de score: UMA regra, nao duas
# ---------------------------------------------------------------------------
#
# Este layer pydantic e paralelo ao layer de dataclasses em
# report/data/, e os dois descrevem os MESMOS campos. Enquanto aqui os scores
# tinham le=1.0 e lá o contrato era [0, 1.25], o repositorio afirmava duas
# regras contraditorias para o mesmo numero e recusava aqui o resultado que
# aceitava lá. SCORE_MIN/SCORE_MAX vem de report/data/base.py, que e a unica
# fonte do contrato (a justificativa dos limites esta documentada lá).
#
# Consequencias aplicadas neste modulo:
# - score agregado: limite superior SCORE_MAX, sem limite inferior. Passar de
#   1.0 e legitimo (o modelo foi melhor sob perturbacao) e ficar abaixo de 0.0
#   tambem e medicao possivel para metrica de erro (perda relativa > 100%).
# - impacto / gap: sao valores COM SINAL. Impacto negativo significa "foi
#   melhor sob perturbacao" e gap negativo significa "o subconjunto deslocado
#   foi melhor"; ge=0.0 recusava exatamente esses casos.
# - metrica bruta (base_score, mean_score, worst_metric...): NAO e normalizada.
#   Para regressao com MSE/MAE ela passa de 1.0 com facilidade e com R2 pode
#   ser negativa, logo nao pode ter ge/le de proporcao.


class RobustnessMetrics(ReportBaseModel):
    """Core robustness metrics."""

    base_score: float = Field(
        default=0.0,
        description=(
            'Baseline model performance without perturbations, in the unit '
            'of the chosen metric (not normalized: MSE/MAE pass 1.0 easily '
            'and R2 can be negative)'
        ),
    )
    robustness_score: float = Field(
        default=0.0,
        le=SCORE_MAX,
        description=(
            'Overall robustness quality score: fraction of the baseline '
            'performance retained under perturbation. 1.0 = unaffected, '
            'above 1.0 = better under perturbation, below 0.0 = lost more '
            'than the whole baseline (error metrics)'
        ),
    )
    avg_raw_impact: float = Field(
        default=0.0,
        description=(
            'Average raw performance impact across perturbations (signed: '
            'negative means the model performed better)'
        ),
    )
    avg_quantile_impact: float = Field(
        default=0.0,
        description='Average quantile-based performance impact (signed)',
    )
    avg_overall_impact: float = Field(
        default=0.0,
        description=(
            'Average overall impact, mean of raw and quantile (signed)'
        ),
    )
    metric: str = Field(
        default='AUC',
        description='Performance metric used (AUC, accuracy, etc.)',
    )

    @property
    def is_robust(self) -> bool:
        """Check if model is considered robust (score > 0.7)."""
        return self.robustness_score > 0.7

    @property
    def degradation_rate(self) -> float:
        """Calculate performance degradation rate."""
        if self.base_score == 0.0:
            return 0.0
        return self.avg_overall_impact / self.base_score


class PerturbationLevelData(ReportBaseModel):
    """Data for a single perturbation level."""

    level: float = Field(description='Perturbation level (e.g., 0.1, 0.5)')
    level_display: str = Field(
        default='', description='Display string for level'
    )
    mean_score: float = Field(
        default=0.0,
        description=(
            'Mean score at this perturbation level, in the unit of the '
            'metric (not normalized)'
        ),
    )
    std_score: float = Field(
        default=0.0, ge=0.0, description='Standard deviation of scores'
    )
    impact: float = Field(
        default=0.0,
        description=(
            'Performance impact at this level (signed: negative means the '
            'model performed better under perturbation)'
        ),
    )
    worst_score: float = Field(
        default=0.0,
        description='Worst score observed at this level (metric unit)',
    )

    @property
    def has_significant_impact(self) -> bool:
        """Check if this level has significant impact (> 0.1)."""
        return self.impact > 0.1


class FeatureRobustnessData(ReportBaseModel):
    """Feature data with robustness-specific information."""

    name: str = Field(description='Feature name')
    importance: float = Field(
        default=0.0, ge=0.0, description='Base feature importance'
    )
    robustness_impact: float = Field(
        default=0.0,
        description='Feature-specific robustness impact (signed)',
    )

    @property
    def is_sensitive(self) -> bool:
        """Check if feature is sensitive to perturbations (impact > 0.2)."""
        return self.robustness_impact > 0.2


class RobustnessReportData(ReportBaseModel):
    """Complete robustness experiment report data."""

    model_name: str = Field(description='Name of the model being tested')
    model_type: str = Field(
        default='Unknown', description='Type/architecture of the model'
    )
    metrics: RobustnessMetrics = Field(
        default_factory=RobustnessMetrics,
        description='Core robustness metrics',
    )
    perturbation_levels: List[PerturbationLevelData] = Field(
        default_factory=list, description='Data for each perturbation level'
    )
    features: List[FeatureRobustnessData] = Field(
        default_factory=list,
        description='Feature importance and robustness data',
    )
    n_iterations: int = Field(
        default=10, ge=1, description='Number of iterations per perturbation'
    )
    charts: Dict[str, str] = Field(
        default_factory=dict, description='Generated chart data (Plotly JSON)'
    )
    notes: str = Field(default='', description='Additional notes or comments')

    @property
    def has_perturbation_data(self) -> bool:
        """Check if perturbation data is available."""
        return len(self.perturbation_levels) > 0

    @property
    def has_feature_data(self) -> bool:
        """Check if feature data is available."""
        return len(self.features) > 0

    @property
    def num_perturbation_levels(self) -> int:
        """Get number of perturbation levels tested."""
        return len(self.perturbation_levels)

    @property
    def num_features(self) -> int:
        """Get number of features analyzed."""
        return len(self.features)

    @property
    def top_features(self) -> List[FeatureRobustnessData]:
        """Get top 5 most important features."""
        sorted_features = sorted(
            self.features, key=lambda f: f.importance, reverse=True
        )
        return sorted_features[:5]

    @property
    def most_sensitive_features(self) -> List[FeatureRobustnessData]:
        """Get top 5 most sensitive features (highest robustness impact)."""
        sorted_features = sorted(
            self.features, key=lambda f: f.robustness_impact, reverse=True
        )
        return sorted_features[:5]

    @property
    def worst_perturbation_level(self) -> Optional[PerturbationLevelData]:
        """Get perturbation level with highest impact."""
        if not self.perturbation_levels:
            return None
        return max(self.perturbation_levels, key=lambda l: l.impact)

    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics for quick overview."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'base_score': self.metrics.base_score,
            'robustness_score': self.metrics.robustness_score,
            'avg_impact': self.metrics.avg_overall_impact,
            'is_robust': self.metrics.is_robust,
            'num_levels': self.num_perturbation_levels,
            'num_features': self.num_features,
            'metric': self.metrics.metric,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f'RobustnessReport({self.model_name}, '
            f'score={self.metrics.robustness_score:.3f}, '
            f'base={self.metrics.base_score:.3f}, '
            f'impact={self.metrics.avg_overall_impact:.3f}, '
            f'levels={self.num_perturbation_levels})'
        )
