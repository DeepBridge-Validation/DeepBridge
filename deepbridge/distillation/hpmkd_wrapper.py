"""
HPMKD Wrapper - API simplificada conforme apresentado no paper

Este módulo fornece uma API simplificada para o HPM-KD que corresponde
ao código apresentado no paper científico (Listing 1).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Any
import logging

from deepbridge.distillation.techniques.hpm import HPMDistiller, HPMConfig
from deepbridge.utils.model_registry import ModelType
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class HPMKD:
    """
    Wrapper simplificado para HPM-KD conforme apresentado no paper.

    Esta classe fornece uma API simplificada que corresponde ao código
    do Listing 1 do paper, facilitando o uso e reprodução dos resultados.

    Example (conforme Listing 1 do paper):
        ```python
        from deepbridge.distillation import HPMKD

        # Configuração automática via meta-learning
        hpmkd = HPMKD(
            teacher_model=teacher,
            student_model=student,
            train_loader=train_loader,
            test_loader=test_loader,
            auto_config=True  # Sem ajuste manual!
        )

        # Destilação progressiva multi-professor
        hpmkd.distill(epochs=150)

        # Avaliar estudante comprimido
        student_acc = hpmkd.evaluate()
        print(f"Compressão: {hpmkd.compression_ratio}x")
        print(f"Retenção: {hpmkd.retention_pct}%")
        ```
    """

    def __init__(
        self,
        teacher_model,
        student_model: Optional[Any] = None,
        train_loader: Optional[Any] = None,
        test_loader: Optional[Any] = None,
        auto_config: bool = True,
        config: Optional[HPMConfig] = None,
        **kwargs
    ):
        """
        Inicializa o HPM-KD com configuração automática.

        Args:
            teacher_model: Modelo professor pré-treinado
            student_model: Modelo estudante (opcional, será criado automaticamente se None)
            train_loader: DataLoader de treinamento ou tuple (X_train, y_train)
            test_loader: DataLoader de teste ou tuple (X_test, y_test)
            auto_config: Se True, usa meta-learning para configuração automática
            config: Configuração customizada (opcional)
            **kwargs: Argumentos adicionais
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.auto_config = auto_config

        # Processar data loaders
        self._process_data_loaders(train_loader, test_loader)

        # Configuração
        if config is None and auto_config:
            config = HPMConfig(
                use_progressive=True,
                use_multi_teacher=True,
                use_adaptive_temperature=True,
                use_cache=True,
                verbose=True,
                **kwargs
            )
        elif config is None:
            config = HPMConfig(**kwargs)

        # Determinar tipo do modelo estudante
        student_model_type = self._infer_student_model_type(student_model)

        # Inicializar HPMDistiller interno
        self._distiller = HPMDistiller(
            teacher_model=teacher_model,
            student_model_type=student_model_type,
            config=config
        )

        # Métricas
        self._teacher_acc = None
        self._student_acc = None
        self._is_distilled = False

        logger.info("HPMKD initialized with auto_config=%s", auto_config)

    def _process_data_loaders(self, train_loader, test_loader):
        """Processa data loaders ou arrays."""
        # Train data
        if train_loader is not None:
            if isinstance(train_loader, tuple) and len(train_loader) == 2:
                self.X_train, self.y_train = train_loader
            elif hasattr(train_loader, 'dataset'):
                # PyTorch DataLoader
                self.X_train, self.y_train = self._extract_from_dataloader(train_loader)
            else:
                raise ValueError("train_loader deve ser tuple (X, y) ou DataLoader")
        else:
            self.X_train, self.y_train = None, None

        # Test data
        if test_loader is not None:
            if isinstance(test_loader, tuple) and len(test_loader) == 2:
                self.X_test, self.y_test = test_loader
            elif hasattr(test_loader, 'dataset'):
                # PyTorch DataLoader
                self.X_test, self.y_test = self._extract_from_dataloader(test_loader)
            else:
                raise ValueError("test_loader deve ser tuple (X, y) ou DataLoader")
        else:
            self.X_test, self.y_test = None, None

    def _extract_from_dataloader(self, dataloader):
        """Extrai dados de um PyTorch DataLoader."""
        X_list, y_list = [], []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                X_batch, y_batch = batch
                X_list.append(X_batch.numpy() if hasattr(X_batch, 'numpy') else X_batch)
                y_list.append(y_batch.numpy() if hasattr(y_batch, 'numpy') else y_batch)

        X = np.vstack(X_list) if len(X_list[0].shape) > 1 else np.concatenate(X_list)
        y = np.concatenate(y_list)
        return X, y

    def _infer_student_model_type(self, student_model):
        """Infere o tipo do modelo estudante."""
        if student_model is None:
            # Usa DecisionTree como padrão
            return ModelType.DECISION_TREE

        # Mapear tipos conhecidos
        model_name = type(student_model).__name__.lower()

        if 'decisiontree' in model_name or 'dtc' in model_name:
            return ModelType.DECISION_TREE
        elif 'randomforest' in model_name or 'rfc' in model_name:
            return ModelType.RANDOM_FOREST
        elif 'logistic' in model_name:
            return ModelType.LOGISTIC_REGRESSION
        elif 'xgb' in model_name:
            return ModelType.XGB
        elif 'gbm' in model_name or 'gradient' in model_name:
            return ModelType.GBM
        elif 'mlp' in model_name or 'neural' in model_name:
            return ModelType.MLP
        else:
            logger.warning(f"Tipo de modelo desconhecido: {model_name}. Usando DecisionTree.")
            return ModelType.DECISION_TREE

    def distill(self, epochs: int = 150, X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None):
        """
        Executa a destilação progressiva multi-professor.

        Args:
            epochs: Número de épocas (usado principalmente para modelos de deep learning)
            X_val: Dados de validação (opcional, será criado split se None)
            y_val: Labels de validação (opcional)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dados de treinamento não fornecidos. Use train_loader no __init__.")

        logger.info(f"Starting distillation with epochs={epochs}")

        # Se não houver validação, criar split
        if X_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                self.X_train, self.y_train,
                test_size=0.2,
                random_state=42,
                stratify=self.y_train if len(np.unique(self.y_train)) > 1 else None
            )
        else:
            X_train, y_train = self.X_train, self.y_train

        # Calcular acurácia do professor
        if self.X_test is not None and self.y_test is not None:
            teacher_preds = self.teacher_model.predict(self.X_test)
            self._teacher_acc = accuracy_score(self.y_test, teacher_preds)
            logger.info(f"Teacher accuracy: {self._teacher_acc:.4f}")

        # Executar destilação
        self._distiller.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        self._is_distilled = True
        logger.info("Distillation completed successfully")

    def evaluate(self) -> float:
        """
        Avalia o estudante destilado no conjunto de teste.

        Returns:
            Acurácia do modelo estudante
        """
        if not self._is_distilled:
            raise RuntimeError("Modelo ainda não foi destilado. Execute distill() primeiro.")

        if self.X_test is None or self.y_test is None:
            raise ValueError("Dados de teste não fornecidos.")

        # Fazer predições
        student_preds = self._distiller.predict(self.X_test)
        self._student_acc = accuracy_score(self.y_test, student_preds)

        logger.info(f"Student accuracy: {self._student_acc:.4f}")
        return self._student_acc

    @property
    def compression_ratio(self) -> float:
        """
        Retorna a taxa de compressão (razão entre tamanho do professor e estudante).

        Returns:
            Razão de compressão
        """
        teacher_size = self._get_model_size(self.teacher_model)
        student_size = self._get_model_size(self._distiller.student_model)

        if student_size == 0:
            return 0.0

        return teacher_size / student_size

    @property
    def retention_pct(self) -> float:
        """
        Retorna a porcentagem de retenção de acurácia (estudante/professor * 100).

        Returns:
            Porcentagem de retenção
        """
        if self._teacher_acc is None or self._student_acc is None:
            # Calcular se necessário
            if self.X_test is not None and self.y_test is not None:
                if self._teacher_acc is None:
                    teacher_preds = self.teacher_model.predict(self.X_test)
                    self._teacher_acc = accuracy_score(self.y_test, teacher_preds)

                if self._student_acc is None and self._is_distilled:
                    self._student_acc = self.evaluate()

        if self._teacher_acc is None or self._student_acc is None:
            return 0.0

        return (self._student_acc / self._teacher_acc) * 100

    def _get_model_size(self, model) -> int:
        """
        Estima o tamanho do modelo (número de parâmetros/nós).

        Args:
            model: Modelo a ser analisado

        Returns:
            Tamanho estimado do modelo
        """
        if model is None:
            return 0

        # Random Forest / Gradient Boosting
        if hasattr(model, 'estimators_'):
            if hasattr(model, 'n_estimators'):
                total_nodes = sum([
                    tree.tree_.node_count if hasattr(tree, 'tree_') else 0
                    for tree in model.estimators_
                ])
                return total_nodes

        # Decision Tree
        if hasattr(model, 'tree_'):
            return model.tree_.node_count

        # Linear models
        if hasattr(model, 'coef_'):
            return np.prod(model.coef_.shape)

        # Neural networks (sklearn MLPClassifier)
        if hasattr(model, 'coefs_'):
            return sum([np.prod(coef.shape) for coef in model.coefs_])

        # PyTorch/TensorFlow models
        if hasattr(model, 'parameters'):
            try:
                return sum(p.numel() for p in model.parameters())
            except:
                pass

        # Fallback
        logger.warning(f"Could not determine size for model type: {type(model).__name__}")
        return 1

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Faz predições com o modelo estudante destilado.

        Args:
            X: Dados de entrada

        Returns:
            Predições
        """
        if not self._is_distilled:
            raise RuntimeError("Modelo ainda não foi destilado. Execute distill() primeiro.")

        return self._distiller.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Retorna probabilidades de classe do modelo estudante.

        Args:
            X: Dados de entrada

        Returns:
            Probabilidades de classe
        """
        if not self._is_distilled:
            raise RuntimeError("Modelo ainda não foi destilado. Execute distill() primeiro.")

        return self._distiller.predict_proba(X)

    @property
    def student(self):
        """Retorna o modelo estudante destilado."""
        return self._distiller.student_model

    def __repr__(self):
        status = "distilled" if self._is_distilled else "not distilled"
        return f"HPMKD(auto_config={self.auto_config}, status={status})"
