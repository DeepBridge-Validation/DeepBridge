import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats

class TimeSeriesMetrics:
    """
    Classe para cálculo e avaliação de métricas para séries temporais.
    Implementa métricas específicas para destilação de conhecimento em séries temporais.
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        teacher_pred: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weights: Optional[Union[np.ndarray, pd.Series]] = None,
        multioutput: bool = False
    ) -> Dict[str, float]:
        """
        Calcula várias métricas de avaliação para séries temporais.
        
        Args:
            y_true: Valores reais
            y_pred: Valores previstos
            teacher_pred: Previsões do modelo professor (opcional)
            sample_weights: Pesos para cada amostra (opcional)
            multioutput: Indica se a saída é multidimensional
            
        Returns:
            Dicionário com métricas calculadas
        """
        # Converte para arrays numpy
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        
        # Considera o caso de saída multidimensional
        multioutput_param = 'uniform_average' if multioutput else 'uniform_average'
        
        metrics = {}
        
        # Erro Quadrático Médio (MSE)
        metrics['mse'] = float(mean_squared_error(
            y_true_np, y_pred_np, 
            sample_weight=sample_weights,
            multioutput=multioutput_param
        ))
        
        # Raiz do Erro Quadrático Médio (RMSE)
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        
        # Erro Absoluto Médio (MAE)
        metrics['mae'] = float(mean_absolute_error(
            y_true_np, y_pred_np, 
            sample_weight=sample_weights,
            multioutput=multioutput_param
        ))
        
        # Erro Percentual Absoluto Médio (MAPE)
        # Evita divisão por zero
        mask = y_true_np != 0
        if np.any(mask):
            metrics['mape'] = float(np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100)
        else:
            metrics['mape'] = float('nan')
        
        # Erro Médio Escalonado (SMAPE)
        denominator = np.abs(y_true_np) + np.abs(y_pred_np)
        mask = denominator != 0
        if np.any(mask):
            metrics['smape'] = float(2.0 * np.mean(np.abs(y_true_np[mask] - y_pred_np[mask]) / denominator[mask]) * 100)
        else:
            metrics['smape'] = float('nan')
        
        # Coeficiente de Determinação (R²)
        metrics['r2'] = float(r2_score(
            y_true_np, y_pred_np, 
            sample_weight=sample_weights,
            multioutput=multioutput_param
        ))
        
        # Métricas relacionadas à destilação (comparação entre professor e aluno)
        if teacher_pred is not None:
            teacher_pred_np = np.array(teacher_pred)
            
            # KL Divergence adaptada para séries temporais
            metrics['kl_divergence'] = TimeSeriesMetrics.calculate_kl_divergence(
                teacher_pred_np, y_pred_np
            )
            
            # Correlação entre previsões do professor e aluno
            metrics['teacher_student_correlation'] = float(
                np.corrcoef(teacher_pred_np.flatten(), y_pred_np.flatten())[0, 1]
            )
            
            # Discrepância entre professor e aluno (MSE entre previsões)
            metrics['teacher_student_mse'] = float(mean_squared_error(
                teacher_pred_np, y_pred_np
            ))
            
            # Teste de Kolmogorov-Smirnov para comparar distribuições de erros
            # Erros do professor
            teacher_errors = y_true_np - teacher_pred_np
            # Erros do aluno
            student_errors = y_true_np - y_pred_np
            
            ks_stat, ks_pvalue = stats.ks_2samp(teacher_errors.flatten(), student_errors.flatten())
            metrics['ks_statistic'] = float(ks_stat)
            metrics['ks_pvalue'] = float(ks_pvalue)
        
        return metrics
    
    @staticmethod
    def calculate_kl_divergence(
        p: Union[np.ndarray, pd.Series],
        q: Union[np.ndarray, pd.Series],
        bins: int = 50,
        epsilon: float = 1e-10
    ) -> float:
        """
        Calcula a divergência KL adaptada para previsões de séries temporais.
        
        Args:
            p: Distribuição de referência (professor)
            q: Distribuição aproximada (aluno)
            bins: Número de bins para histograma
            epsilon: Valor pequeno para evitar log(0)
            
        Returns:
            Valor da divergência KL
        """
        # Converte para arrays numpy e achata
        p_flat = np.array(p).flatten()
        q_flat = np.array(q).flatten()
        
        # Usa os mesmos bins para ambas distribuições
        min_val = min(p_flat.min(), q_flat.min())
        max_val = max(p_flat.max(), q_flat.max())
        
        # Cria histogramas
        p_hist, bin_edges = np.histogram(p_flat, bins=bins, range=(min_val, max_val), density=True)
        q_hist, _ = np.histogram(q_flat, bins=bins, range=(min_val, max_val), density=True)
        
        # Adiciona um pequeno valor para evitar log(0)
        p_hist = p_hist + epsilon
        q_hist = q_hist + epsilon
        
        # Normaliza
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()
        
        # Calcula KL divergence
        kl_div = np.sum(p_hist * np.log(p_hist / q_hist))
        
        return float(kl_div)
    
    @staticmethod
    def calculate_metrics_from_predictions(
        data: pd.DataFrame,
        true_column: str,
        pred_column: str,
        teacher_column: Optional[str] = None,
        weight_column: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calcula métricas a partir de colunas de um DataFrame.
        
        Args:
            data: DataFrame com previsões
            true_column: Nome da coluna com valores reais
            pred_column: Nome da coluna com valores previstos pelo aluno
            teacher_column: Nome da coluna com valores previstos pelo professor
            weight_column: Nome da coluna com pesos amostrais
            
        Returns:
            Dicionário com métricas calculadas
        """
        y_true = data[true_column]
        y_pred = data[pred_column]
        
        teacher_pred = data[teacher_column] if teacher_column else None
        sample_weights = data[weight_column] if weight_column else None
        
        return TimeSeriesMetrics.calculate_metrics(
            y_true, y_pred, teacher_pred, sample_weights
        )
    
    @staticmethod
    def plot_prediction_comparison(
        y_true: Union[np.ndarray, pd.Series],
        student_pred: Union[np.ndarray, pd.Series],
        teacher_pred: Optional[Union[np.ndarray, pd.Series]] = None,
        title: str = "Comparação de Previsões",
        filename: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Cria uma visualização comparando as previsões do modelo aluno e professor.
        
        Args:
            y_true: Valores reais
            student_pred: Previsões do modelo aluno
            teacher_pred: Previsões do modelo professor (opcional)
            title: Título do gráfico
            filename: Nome do arquivo para salvar o gráfico (opcional)
            dates: Índice de datas para o eixo x (opcional)
            figsize: Tamanho da figura
            
        Returns:
            Objeto figura do matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Configura o eixo x
        x = dates if dates is not None else np.arange(len(y_true))
        
        # Plota os valores reais
        ax.plot(x, y_true, 'k-', label='Real', linewidth=2)
        
        # Plota as previsões do aluno
        ax.plot(x, student_pred, 'r-', label='Aluno', alpha=0.8)
        
        # Plota as previsões do professor, se disponíveis
        if teacher_pred is not None:
            ax.plot(x, teacher_pred, 'b-', label='Professor', alpha=0.6)
            
            # Adiciona faixa de diferença entre professor e aluno
            ax.fill_between(
                x, student_pred, teacher_pred, 
                color='gray', alpha=0.2,
                label='Diferença'
            )
        
        # Adiciona legendas e título
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Tempo', fontsize=12)
        ax.set_ylabel('Valor', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotaciona as datas no eixo x se necessário
        if dates is not None:
            plt.xticks(rotation=45)
            fig.tight_layout()
        
        # Salva a figura se um nome de arquivo for fornecido
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_error_distribution(
        y_true: Union[np.ndarray, pd.Series],
        student_pred: Union[np.ndarray, pd.Series],
        teacher_pred: Optional[Union[np.ndarray, pd.Series]] = None,
        title: str = "Distribuição de Erros",
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Cria uma visualização da distribuição de erros.
        
        Args:
            y_true: Valores reais
            student_pred: Previsões do modelo aluno
            teacher_pred: Previsões do modelo professor (opcional)
            title: Título do gráfico
            filename: Nome do arquivo para salvar o gráfico (opcional)
            figsize: Tamanho da figura
            
        Returns:
            Objeto figura do matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calcula os erros do aluno
        student_errors = y_true - student_pred
        
        # Plota a distribuição de erros do aluno
        ax.hist(student_errors, bins=30, alpha=0.7, label='Aluno', color='red')
        
        # Calcula e plota os erros do professor, se disponíveis
        if teacher_pred is not None:
            teacher_errors = y_true - teacher_pred
            ax.hist(teacher_errors, bins=30, alpha=0.5, label='Professor', color='blue')
            
            # Adiciona estatísticas comparativas
            ax.axvline(x=np.mean(student_errors), color='red', linestyle='--', 
                      label=f'Média Aluno: {np.mean(student_errors):.4f}')
            ax.axvline(x=np.mean(teacher_errors), color='blue', linestyle='--',
                      label=f'Média Professor: {np.mean(teacher_errors):.4f}')
        else:
            # Adiciona apenas estatísticas do aluno
            ax.axvline(x=np.mean(student_errors), color='red', linestyle='--', 
                      label=f'Média: {np.mean(student_errors):.4f}')
        
        # Adiciona legendas e título
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Erro', fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Salva a figura se um nome de arquivo for fornecido
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig