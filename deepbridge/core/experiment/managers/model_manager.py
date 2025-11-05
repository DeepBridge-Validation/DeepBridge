import typing as t
import pandas as pd
import numpy as np
from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode

class ModelManager:
    """
    Manages creation and handling of different models in the experiment.
    """
    
    def __init__(self, dataset, experiment_type, verbose=False):
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.verbose = verbose
        
    def get_default_model_type(self):
        """Get a default model type for XGBoost or similar"""
        # Try to find XGBoost or fallback to first model type
        for model_type in ModelType:
            if 'XGB' in model_type.name:
                return model_type
        # Fallback to first model type
        return next(iter(ModelType))
    
    def create_alternative_models(self, X_train, y_train, lazy=False):
        """
        Create 3 alternative models different from the original model,
        using ModelRegistry directly without SurrogateModel.

        OTIMIZAÇÃO: Suporta lazy loading para evitar treinar modelos
        desnecessariamente. Use lazy=True para retornar dict vazio.

        Priority order for alternative models:
        GLMClassifier, GAMClassifier, GBM, XGB, LOGISTIC_REGRESSION, DECISION_TREE, RANDOM_FOREST, MLP

        Args:
            X_train: Training features
            y_train: Training labels
            lazy: Se True, retorna dict vazio (lazy loading)

        Returns:
            Dict of alternative models (vazio se lazy=True)
        """
        alternative_models = {}

        # OTIMIZAÇÃO: Se lazy loading ativado, retornar vazio
        if lazy:
            if self.verbose:
                print("⚡ Lazy loading ativado: Pulando criação de alternative_models (economizando ~30-50s)")
            return alternative_models

        # Check if dataset has a model
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            if self.verbose:
                print("No original model found in dataset. Skipping alternative model creation.")
            return alternative_models

        # Get original model type if possible
        original_model = self.dataset.model
        original_model_name = original_model.__class__.__name__.upper()

        # Define prioritized model types order
        prioritized_order = [
            ModelType.GLM_CLASSIFIER,
            ModelType.GAM_CLASSIFIER,
            ModelType.GBM,
            ModelType.XGB,
            ModelType.LOGISTIC_REGRESSION,
            ModelType.DECISION_TREE,
            ModelType.RANDOM_FOREST,
            ModelType.MLP
        ]

        if self.verbose:
            print(f"Available model types in priority order: {[mt.name for mt in prioritized_order]}")
            print(f"Original model identified as: {original_model.__class__.__name__}")

        # Identify original model type by name
        original_model_type = None
        for model_type in prioritized_order:
            if model_type.name in original_model_name or original_model_name in model_type.name:
                original_model_type = model_type
                break

        if self.verbose:
            print(f"Mapped to model type: {original_model_type}")

        # Create a list of models to generate, excluding the original model if identified
        models_to_create = []
        for model_type in prioritized_order:
            if model_type != original_model_type:
                models_to_create.append(model_type)
                if len(models_to_create) >= 3:  # Limit to 3 models
                    break

        if self.verbose:
            print(f"Creating alternative models: {[m.name for m in models_to_create]}")

        # Determine if we're working with a classification problem
        is_classification = self.experiment_type == "binary_classification"
        mode = ModelMode.CLASSIFICATION if is_classification else ModelMode.REGRESSION

        # Create and fit each alternative model
        for model_type in models_to_create:
            try:
                # Get model with default parameters directly from ModelRegistry
                model = ModelRegistry.get_model(
                    model_type=model_type,
                    custom_params=None,  # Use default parameters
                    mode=mode  # Use classification or regression mode based on experiment_type
                )

                # Fit the model on training data
                if self.verbose:
                    print(f"Fitting {model_type.name} model...")

                model.fit(X_train, y_train)

                # Store model with its type name
                alternative_models[model_type.name] = model

                if self.verbose:
                    print(f"Successfully created and fitted {model_type.name} as {model.__class__.__name__}")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to fit {model_type.name}: {str(e)}")

        if self.verbose:
            print(f"Created {len(alternative_models)} alternative models")

        return alternative_models
    
    def create_distillation_model(self, 
                            distillation_method: str,
                            student_model_type: ModelType,
                            student_params: t.Optional[dict],
                            temperature: float,
                            alpha: float,
                            use_probabilities: bool,
                            n_trials: int,
                            validation_split: float) -> object:
        """Create appropriate distillation model based on method and available data"""
        if use_probabilities:
            prob_train = self.dataset.original_prob
            if prob_train is None:
                raise ValueError("No teacher probabilities available. Set use_probabilities=False to use teacher model")
            return self._create_model_from_probabilities(
                distillation_method, student_model_type, student_params,
                temperature, alpha, n_trials, validation_split
            )
        else:
            if self.dataset.model is None:
                raise ValueError("No teacher model available. Set use_probabilities=True to use pre-calculated probabilities")
            return self._create_model_from_teacher(
                distillation_method, student_model_type, student_params,
                temperature, alpha, n_trials, validation_split
            )
            
    def _create_model_from_probabilities(self,
                                  distillation_method: str,
                                  student_model_type: ModelType,
                                  student_params: t.Optional[dict],
                                  temperature: float,
                                  alpha: float,
                                  n_trials: int,
                                  validation_split: float) -> object:
        """Create distillation model from pre-calculated probabilities"""
        prob_train = self.dataset.original_prob
        
        if distillation_method.lower() == "surrogate":
            # Import at runtime to avoid circular import
            from deepbridge.distillation.techniques.surrogate import SurrogateModel
            
            return SurrogateModel.from_probabilities(
                probabilities=prob_train,
                student_model_type=student_model_type,
                student_params=student_params,
                random_state=None,  # Use default or get from dataset
                validation_split=validation_split,
                n_trials=n_trials
            )
        elif distillation_method.lower() == "knowledge_distillation":
            # Import at runtime to avoid circular import
            from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
            
            return KnowledgeDistillation.from_probabilities(
                probabilities=prob_train,
                student_model_type=student_model_type,
                student_params=student_params,
                temperature=temperature,
                alpha=alpha,
                n_trials=n_trials,
                validation_split=validation_split,
                random_state=None  # Use default or get from dataset
            )
        else:
            raise ValueError(f"Unknown distillation method: {distillation_method}. Use 'surrogate' or 'knowledge_distillation'")
            
    def _create_model_from_teacher(self,
                            distillation_method: str,
                            student_model_type: ModelType,
                            student_params: t.Optional[dict],
                            temperature: float,
                            alpha: float,
                            n_trials: int,
                            validation_split: float) -> object:
        """Create distillation model from teacher model"""
        if distillation_method.lower() == "surrogate":
            # Surrogate method doesn't support direct use of teacher model
            raise ValueError("The surrogate method does not support direct use of teacher model. "
                           "Please set use_probabilities=True or use method='knowledge_distillation'")
        elif distillation_method.lower() == "knowledge_distillation":
            # Import at runtime to avoid circular import
            from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
            
            return KnowledgeDistillation(
                teacher_model=self.dataset.model,
                student_model_type=student_model_type,
                student_params=student_params,
                temperature=temperature,
                alpha=alpha,
                n_trials=n_trials,
                validation_split=validation_split,
                random_state=None  # Use default or get from dataset
            )
        else:
            raise ValueError(f"Unknown distillation method: {distillation_method}. Use 'surrogate' or 'knowledge_distillation'")
