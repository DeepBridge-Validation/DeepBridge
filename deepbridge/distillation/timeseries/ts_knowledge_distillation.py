import numpy as np
import pandas as pd
import optuna
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import warnings
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

from deepbridge.distillation.timeseries.ts_model_registry import TSModelType, TSModelRegistry
from deepbridge.distillation.timeseries.ts_preprocessing import TimeSeriesPreprocessor
from deepbridge.distillation.timeseries.ts_metrics import TimeSeriesMetrics

class TSKnowledgeDistillation(BaseEstimator, RegressorMixin):
    """
    Implementação de destilação de conhecimento para modelos de séries temporais.
    Permite transferir conhecimento de um modelo complexo (professor) para um modelo mais simples (aluno).
    """
    
    def __init__(
        self,
        teacher_model: Optional[BaseEstimator] = None,
        teacher_predictions: Optional[np.ndarray] = None,
        student_model_type: TSModelType = TSModelType.RIDGE,
        student_params: Optional[Dict[str, Any]] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
        temperature: float = 1.0,
        alpha: float = 0.5,
        n_trials: int = 50,
        validation_split: float = 0.2,
        loss_fn: Optional[Callable] = None,
        random_state: int = 42,
        window_size: int = 10,
        horizon: int = 1
    ):
        """
        Inicializa o modelo de destilação.
        
        Args:
            teacher_model: Modelo professor pré-treinado (opcional se teacher_predictions fornecido)
            teacher_predictions: Previsões pré-calculadas do professor (opcional se teacher_model fornecido)
            student_model_type: Tipo de modelo aluno a ser usado
            student_params: Parâmetros personalizados para o modelo aluno
            preprocessor: Preprocessador para dados de séries temporais
            temperature: Parâmetro de temperatura para suavizar as distribuições
            alpha: Peso entre perda do professor e perda da verdade
            n_trials: Número de trials para otimização de hiperparâmetros
            validation_split: Fração dos dados para validação durante otimização
            loss_fn: Função de perda personalizada (opcional)
            random_state: Semente aleatória para reprodutibilidade
            window_size: Tamanho da janela para sequências de dados (quando aplicável)
            horizon: Horizonte de previsão
        """
        if teacher_model is None and teacher_predictions is None:
            raise ValueError("Deve fornecer teacher_model ou teacher_predictions")
            
        self.teacher_model = teacher_model
        self.teacher_predictions = teacher_predictions
        self.student_model_type = student_model_type
        self.student_params = student_params
        self.temperature = temperature
        self.alpha = alpha
        self.n_trials = n_trials
        self.validation_split = validation_split
        self.random_state = random_state
        self.window_size = window_size
        self.horizon = horizon
        
        # Inicializa ou usa o preprocessador fornecido
        if preprocessor is None:
            self.preprocessor = TimeSeriesPreprocessor(
                window_size=window_size,
                horizon=horizon
            )
        else:
            self.preprocessor = preprocessor
        
        # Usa a função de perda padrão ou personalizada
        self.loss_fn = loss_fn if loss_fn is not None else self._combined_mse_loss
        
        # Inicializa o modelo aluno e métricas
        self.student_model = None
        self.metrics_calculator = TimeSeriesMetrics()
        self.best_params = None
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'teacher_loss': [],
            'label_loss': []
        }
        
        # Para armazenar nomes de features, se fornecidos
        self.feature_names_ = None
    
    def _get_teacher_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Obtém previsões do modelo professor.
        
        Args:
            X: Features de entrada
            
        Returns:
            Array de previsões do professor
        """
        if self.teacher_predictions is not None:
            if len(self.teacher_predictions) != len(X):
                raise ValueError(
                    f"Número de previsões do professor ({len(self.teacher_predictions)}) "
                    f"não corresponde ao número de amostras ({len(X)})"
                )
            return self.teacher_predictions
        
        if self.teacher_model is None:
            raise ValueError("teacher_model deve ser fornecido quando teacher_predictions não estão disponíveis")
            
        # Verifica se o professor é univariado ou multivariado
        try:
            try:
                # Tenta previsão com toda a entrada
                return self.teacher_model.predict(X)
            except:
                # Se falhar, pode ser um modelo univariado que usa apenas a série
                if hasattr(X, 'iloc'):  # DataFrame ou Series
                    if isinstance(X, pd.DataFrame) and 'y' in X.columns:
                        return self.teacher_model.predict(X['y'].values)
                    else:
                        return self.teacher_model.predict(X.iloc[:, 0].values)
                else:  # numpy array
                    if len(X.shape) > 1 and X.shape[1] > 1:
                        return self.teacher_model.predict(X[:, 0])
                    else:
                        return self.teacher_model.predict(X.flatten())
        except Exception as e:
            raise ValueError(f"Erro ao obter previsões do professor: {str(e)}")
    
    def _combined_mse_loss(
        self, 
        y_true: np.ndarray, 
        student_pred: np.ndarray, 
        teacher_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula a perda combinada usando MSE.
        
        Args:
            y_true: Valores verdadeiros (rótulos)
            student_pred: Previsões do modelo aluno
            teacher_pred: Previsões do modelo professor
            
        Returns:
            Dicionário com diferentes componentes da perda
        """
        # Perda em relação aos valores reais
        label_loss = np.mean((y_true - student_pred) ** 2)
        
        # Perda em relação às previsões do professor
        teacher_loss = np.mean((teacher_pred - student_pred) ** 2)
        
        # Perda combinada usando o parâmetro alpha
        combined_loss = self.alpha * teacher_loss + (1 - self.alpha) * label_loss
        
        return {
            'combined_loss': combined_loss,
            'label_loss': label_loss,
            'teacher_loss': teacher_loss
        }
    
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, teacher_pred: np.ndarray) -> float:
        """
        Função objetivo para Optuna.
        
        Args:
            trial: Trial do Optuna
            X: Features de treinamento
            y: Valores verdadeiros
            teacher_pred: Previsões do professor
            
        Returns:
            Valor da perda para minimizar
        """
        # Obtém hiperparâmetros para este trial
        trial_params = TSModelRegistry.get_param_space(self.student_model_type, trial)
        
        # Divide os dados para validação
        X_train, X_val, y_train, y_val, teacher_train, teacher_val = train_test_split(
            X, y, teacher_pred, test_size=self.validation_split, random_state=self.random_state
        )
        
        # Cria e treina modelo aluno
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            student = TSModelRegistry.get_model(self.student_model_type, trial_params)
            student.fit(X_train, y_train)
        
        # Obtém previsões do aluno no conjunto de validação
        student_pred = student.predict(X_val)
        
        # Calcula a perda
        loss_values = self.loss_fn(y_val, student_pred, teacher_val)
        
        # Registra valores para análise posterior
        trial.set_user_attr('label_loss', float(loss_values['label_loss']))
        trial.set_user_attr('teacher_loss', float(loss_values['teacher_loss']))
        
        return float(loss_values['combined_loss'])

    @classmethod
    def from_predictions(
        cls,
        predictions: Union[np.ndarray, pd.DataFrame],
        student_model_type: TSModelType = TSModelType.RIDGE,
        student_params: Dict[str, Any] = None,
        temperature: float = 1.0,
        alpha: float = 0.5,
        n_trials: int = 50,
        validation_split: float = 0.2,
        random_state: int = 42,
        window_size: int = 10,
        horizon: int = 1
    ) -> 'TSKnowledgeDistillation':
        """
        Cria uma instância de TSKnowledgeDistillation a partir de previsões pré-calculadas.
        
        Args:
            predictions: Array ou DataFrame de previsões do professor
            student_model_type: Tipo de modelo aluno
            student_params: Parâmetros personalizados para o modelo aluno
            temperature: Parâmetro de temperatura
            alpha: Parâmetro de peso
            n_trials: Número de trials para otimização
            validation_split: Fração dos dados para validação
            random_state: Semente aleatória
            window_size: Tamanho da janela para sequências
            horizon: Horizonte de previsão
            
        Returns:
            Instância de TSKnowledgeDistillation
        """
        # Processar previsões do professor
        if isinstance(predictions, pd.DataFrame):
            teacher_predictions = predictions.values
        else:
            teacher_predictions = predictions
            
        return cls(
            teacher_predictions=teacher_predictions,
            student_model_type=student_model_type,
            student_params=student_params,
            temperature=temperature,
            alpha=alpha,
            n_trials=n_trials,
            validation_split=validation_split,
            random_state=random_state,
            window_size=window_size,
            horizon=horizon
        )

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'TSKnowledgeDistillation':
        """
        Treina o modelo aluno usando destilação de conhecimento.
        
        Args:
            X: Features de entrada
            y: Valores alvo
            verbose: Se deve exibir progresso
            
        Returns:
            self: O modelo treinado
        """
        # Armazenar nomes de features se X for DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        
        # Pré-processamento dos dados
        if not self.preprocessor.is_fitted:
            self.preprocessor.fit(y)
        
        # Gera previsões do professor
        teacher_pred = self._get_teacher_predictions(X)
        
        if self.student_params is None:
            # Filtrar avisos durante otimização
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                # Configurar logger do Optuna
                import logging
                optuna_logger = logging.getLogger("optuna")
                optuna_logger_level = optuna_logger.getEffectiveLevel()
                optuna_logger.setLevel(logging.WARNING if verbose else logging.ERROR)
                
                # Otimizar hiperparâmetros usando Optuna
                study = optuna.create_study(direction="minimize")
                objective = lambda trial: self._objective(trial, X, y, teacher_pred)
                
                try:
                    study.optimize(objective, n_trials=self.n_trials)
                    
                    # Obter os melhores hiperparâmetros
                    self.best_params = study.best_params
                    if verbose:
                        print(f"Melhores hiperparâmetros encontrados: {self.best_params}")
                        print(f"Melhor valor de perda: {study.best_value:.6f}")
                        
                    # Extrair informações adicionais do melhor trial
                    best_trial = study.best_trial
                    if 'label_loss' in best_trial.user_attrs and 'teacher_loss' in best_trial.user_attrs:
                        if verbose:
                            print(f"Perda nos rótulos: {best_trial.user_attrs['label_loss']:.6f}")
                            print(f"Perda no professor: {best_trial.user_attrs['teacher_loss']:.6f}")
                finally:
                    # Restaurar nível de log do Optuna
                    optuna_logger.setLevel(optuna_logger_level)
                
                # Criar modelo aluno com melhores parâmetros
                self.student_model = TSModelRegistry.get_model(
                    model_type=self.student_model_type,
                    custom_params=self.best_params
                )
        else:
            # Usar hiperparâmetros fornecidos
            self.best_params = self.student_params
            self.student_model = TSModelRegistry.get_model(
                model_type=self.student_model_type,
                custom_params=self.student_params
            )
        
        # Treinar o modelo aluno final
        self.student_model.fit(X, y)
        
        # Avaliar o desempenho final
        student_pred = self.student_model.predict(X)
        loss_values = self.loss_fn(y, student_pred, teacher_pred)
        
        # Armazenar valores finais no histórico
        self.training_history['loss'].append(loss_values['combined_loss'])
        self.training_history['label_loss'].append(loss_values['label_loss'])
        self.training_history['teacher_loss'].append(loss_values['teacher_loss'])
        
        if verbose:
            print("\nTreinamento concluído!")
            print(f"Perda combinada final: {loss_values['combined_loss']:.6f}")
            print(f"Perda nos rótulos final: {loss_values['label_loss']:.6f}")
            print(f"Perda no professor final: {loss_values['teacher_loss']:.6f}")
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza previsões com o modelo aluno.
        
        Args:
            X: Features de entrada
            
        Returns:
            Previsões do modelo aluno
        """
        if self.student_model is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        
        return self.student_model.predict(X)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        return_predictions: bool = False
    ) -> Dict:
        """
        Avalia o desempenho do modelo aluno.
        
        Args:
            X: Features de entrada
            y_true: Valores verdadeiros
            return_predictions: Se deve incluir previsões na saída
            
        Returns:
            Dicionário com métricas de avaliação e opcionalmente previsões
        """
        if self.student_model is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        
        # Obter previsões
        y_pred = self.student_model.predict(X)
        
        # Obter previsões do professor, se disponível
        teacher_pred = None
        if self.teacher_model is not None or self.teacher_predictions is not None:
            try:
                teacher_pred = self._get_teacher_predictions(X)
            except Exception as e:
                print(f"Aviso: Não foi possível obter previsões do professor para avaliação: {str(e)}")
        
        # Calcular métricas
        metrics = TimeSeriesMetrics.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            teacher_pred=teacher_pred
        )
        
        # Adicionar informações sobre hiperparâmetros
        metrics['best_params'] = self.best_params
        
        # Incluir previsões se solicitado
        if return_predictions:
            predictions = {
                'y_true': y_true,
                'y_pred': y_pred
            }
            
            if teacher_pred is not None:
                predictions['teacher_pred'] = teacher_pred
                
            return {'metrics': metrics, 'predictions': predictions}
        
        return metrics
    
    def evaluate_from_dataframe(
        self,
        data: pd.DataFrame,
        features_columns: List[str],
        target_column: str,
        return_predictions: bool = False
    ) -> Dict:
        """
        Avalia o modelo usando um DataFrame como entrada.
        
        Args:
            data: DataFrame de entrada
            features_columns: Lista de nomes de colunas de features
            target_column: Nome da coluna alvo
            return_predictions: Se deve incluir previsões na saída
            
        Returns:
            Dicionário contendo métricas de avaliação e opcionalmente previsões
        """
        X = data[features_columns].values
        y_true = data[target_column].values
        
        return self.evaluate(X, y_true, return_predictions=return_predictions)
    
    def plot_predictions(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        title: str = "Comparação de Previsões",
        filename: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> plt.Figure:
        """
        Plota uma comparação entre valores reais e previsões dos modelos.
        
        Args:
            X: Features de entrada
            y_true: Valores verdadeiros
            title: Título do gráfico
            filename: Nome do arquivo para salvar o gráfico (opcional)
            dates: Índice de datas para o eixo x (opcional)
            
        Returns:
            Objeto figura do matplotlib
        """
        if self.student_model is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        
        # Obter previsões do aluno
        student_pred = self.student_model.predict(X)
        
        # Obter previsões do professor, se disponível
        teacher_pred = None
        if self.teacher_model is not None or self.teacher_predictions is not None:
            try:
                teacher_pred = self._get_teacher_predictions(X)
            except:
                pass
        
        # Usar função de plotagem da classe TimeSeriesMetrics
        return TimeSeriesMetrics.plot_prediction_comparison(
            y_true=y_true,
            student_pred=student_pred,
            teacher_pred=teacher_pred,
            title=title,
            filename=filename,
            dates=dates
        )
    
    def save(self, filepath: str, save_teacher: bool = False) -> str:
        """
        Salva o modelo de destilação em um arquivo.
        
        Args:
            filepath: Caminho para salvar o modelo
            save_teacher: Se deve salvar também o modelo professor
            
        Returns:
            Caminho onde o modelo foi salvo
        """
        if self.student_model is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        
        # Cria um dicionário com todos os componentes necessários
        model_data = {
            'student_model': self.student_model,
            'student_model_type': self.student_model_type,
            'preprocessor': self.preprocessor,
            'best_params': self.best_params,
            'training_history': self.training_history,
            'alpha': self.alpha,
            'temperature': self.temperature,
            'window_size': self.window_size,
            'horizon': self.horizon,
            'feature_names': self.feature_names_,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Incluir o modelo professor se solicitado
        if save_teacher and self.teacher_model is not None:
            model_data['teacher_model'] = self.teacher_model
        
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Salvar o modelo
        joblib.dump(model_data, filepath)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'TSKnowledgeDistillation':
        """
        Carrega um modelo de destilação de um arquivo.
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            Modelo de destilação carregado
        """
        # Carregar dados do modelo
        model_data = joblib.load(filepath)
        
        # Criar uma nova instância
        distiller = cls(
            teacher_model=model_data.get('teacher_model'),
            student_model_type=model_data['student_model_type'],
            student_params=model_data['best_params'],
            preprocessor=model_data['preprocessor'],
            temperature=model_data['temperature'],
            alpha=model_data['alpha'],
            window_size=model_data['window_size'],
            horizon=model_data['horizon']
        )
        
        # Restaurar componentes
        distiller.student_model = model_data['student_model']
        distiller.best_params = model_data['best_params']
        distiller.training_history = model_data['training_history']
        distiller.feature_names_ = model_data.get('feature_names')
        
        return distiller
    
    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Obtém a importância das features do modelo aluno, se disponível.
        
        Returns:
            Dicionário com importâncias das features ou None se não disponível
        """
        if self.student_model is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        
        # Verifica se o modelo tem importâncias de features
        if hasattr(self.student_model, 'feature_importances_'):
            importances = self.student_model.feature_importances_
            
            # Se tivermos nomes de features, associar as importâncias
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                return {name: float(imp) for name, imp in zip(self.feature_names_, importances)}
            else:
                return {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
        
        return None
    
    def plot_feature_importances(self, filename: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plota a importância das features, se disponível.
        
        Args:
            filename: Nome do arquivo para salvar o gráfico (opcional)
            
        Returns:
            Figura matplotlib ou None se não houver importâncias disponíveis
        """
        importances_dict = self.get_feature_importances()
        
        if importances_dict is None:
            print("Este modelo não fornece importâncias de features")
            return None
        
        # Ordenar importâncias
        sorted_importances = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_importances]
        values = [x[1] for x in sorted_importances]
        
        # Criar gráfico
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), values, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importância')
        plt.ylabel('Feature')
        plt.title('Importância das Features no Modelo Destilado')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Salvar se necessário
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return plt.gcf()