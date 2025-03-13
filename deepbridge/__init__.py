"""
DeepBridge - A library for model distillation and validation.
"""

from deepbridge.db_data import DBDataset
from deepbridge.experiment import Experiment
from deepbridge.distillation.classification.surrogate import SurrogateModel
from deepbridge.auto_distiller import AutoDistiller

# Importar componentes de validação mas sem configurar extensões automáticas
# isso evita o erro de métodos indefinidos
try:
    from deepbridge.validation.hyperparameter_importance import (
        HyperparameterImportance,
        EfficientHyperparameterTuning
    )
    
    # Configuração manual apenas das extensões disponíveis
    # Isso evita o erro de importação, mas ainda permite usar os métodos disponíveis
    from deepbridge.validation.experiment_extensions import (
        analyze_hyperparameter_importance,
        optimize_hyperparameters,
        analyze_hyperparameters_workaround
    )
    
    # Aplicar apenas os métodos existentes
    Experiment.analyze_hyperparameter_importance = analyze_hyperparameter_importance
    Experiment.optimize_hyperparameters = optimize_hyperparameters
    
    # Adicionar a função workaround como um método
    Experiment.analyze_hyperparameters_workaround = analyze_hyperparameters_workaround

    
except ImportError:
    # Se o módulo de validação não estiver disponível, continuamos sem ele
    pass

__version__ = "0.1.10"
__author__ = "Team DeepBridge"

__all__ = [
    "DBDataset",
    "Experiment",
    "AutoDistiller",
    "SurrogateModel"
]

# Adicionar os componentes de validação a __all__ se estiverem disponíveis
try:
    __all__.extend([
        "HyperparameterImportance",
        "EfficientHyperparameterTuning"
    ])
except NameError:
    # Se os componentes não estiverem definidos, continuamos sem eles
    pass