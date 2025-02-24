import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
from datetime import datetime

from deepbridge.distillation.timeseries.ts_model_registry import TSModelType, TSModelRegistry
from deepbridge.distillation.timeseries.ts_knowledge_distillation import TSKnowledgeDistillation
from deepbridge.distillation.timeseries.ts_preprocessing import TimeSeriesPreprocessor
from deepbridge.distillation.timeseries.ts_metrics import TimeSeriesMetrics

class TSConfig:
    """
    Configuração para experimentos de destilação em séries temporais.
    """
    
    def __init__(
        self,
        output_dir: str = "ts_distillation_results",
        test_size: float = 0.2,
        random_state: int = 42,
        n_trials: int = 10,
        validation_split: float = 0.2,
        window_size: int = 10,
        horizon: int = 1,
        verbose: bool = True
    ):
        """
        Inicializa a configuração.
        
        Args:
            output_dir: Diretório para salvar resultados
            test_size: Fração dos dados para teste
            random_state: Semente aleatória
            n_trials: Número de trials para otimização
            validation_split: Fração dos dados para validação
            window_size: Tamanho da janela para sequências
            horizon: Horizonte de previsão
            verbose: Se deve exibir mensagens de progresso
        """
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        self.n_trials = n_trials
        self.validation_split = validation_split
        self.window_size = window_size
        self.horizon = horizon
        self.verbose = verbose
        
        # Configurações padrão
        self._set_default_config()
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
    
    def _set_default_config(self):
        """Define configurações padrão para tipos de modelos, temperaturas e alphas."""
        self.model_types = [
            TSModelType.RIDGE,
            TSModelType.ELASTIC_NET,
            TSModelType.RANDOM_FOREST,
            TSModelType.XGB
        ]

        for metric, minimize, metric_name in metrics:
            try:
                best = self.find_best_model(metric=metric, minimize=minimize)
                report.append(f"### Melhor Modelo por {metric_name}")
                report.append(f"- Tipo de Modelo: {best['model_type']}")
                report.append(f"- Temperatura: {best['temperature']}")
                report.append(f"- Alpha: {best['alpha']}")
                
                # Adiciona todas as métricas disponíveis
                metrics_to_report = ['rmse', 'mae', 'mape', 'r2', 'kl_divergence', 
                                    'teacher_student_correlation', 'teacher_student_mse']
                
                for m in metrics_to_report:
                    if m in best and not pd.isna(best[m]):
                        report.append(f"- {m.upper()}: {best[m]}")
                        
                report.append("")
            except (ValueError, KeyError) as e:
                report.append(f"### Melhor Modelo por {metric_name}")
                report.append(f"Não foi possível encontrar o melhor modelo: {str(e)}")
                report.append("")
        
        # Adiciona comparação de modelos
        report.append("## Comparação de Modelos")
        
        # Obtém médias por tipo de modelo
        model_comparison = self.results_df.groupby('model_type').agg({
            'rmse': ['mean', 'min', 'std'],
            'mae': ['mean', 'min', 'std'],
            'r2': ['mean', 'max', 'std'],
            'teacher_student_correlation': ['mean', 'max', 'std'],
            'kl_divergence': ['mean', 'min', 'std']
        }).reset_index()
        
        # Converte para string formatado
        model_comparison_str = model_comparison.to_string()
        report.append("```")
        report.append(model_comparison_str)
        report.append("```")
        report.append("")
        
        # Adiciona impacto da temperatura
        report.append("## Impacto da Temperatura")
        
        # Agrupa por temperatura
        temp_impact = self.results_df.groupby(['temperature', 'model_type']).agg({
            'rmse': 'mean',
            'mae': 'mean',
            'r2': 'mean',
            'kl_divergence': 'mean'
        }).reset_index()
        
        temp_impact_str = temp_impact.to_string()
        report.append("```")
        report.append(temp_impact_str)
        report.append("```")
        report.append("")
        
        # Adiciona impacto do alpha
        report.append("## Impacto do Alpha")
        
        # Agrupa por alpha
        alpha_impact = self.results_df.groupby(['alpha', 'model_type']).agg({
            'rmse': 'mean',
            'mae': 'mean',
            'r2': 'mean',
            'kl_divergence': 'mean'
        }).reset_index()
        
        alpha_impact_str = alpha_impact.to_string()
        report.append("```")
        report.append(alpha_impact_str)
        report.append("```")
        report.append("")
        
        # Adiciona conclusões e recomendações
        report.append("## Conclusões e Recomendações")
        
        try:
            best_overall = self.find_best_model(metric='rmse', minimize=True)
            best_fidelity = self.find_best_model(metric='teacher_student_correlation', minimize=False)
            
            report.append("### Melhor Modelo Geral")
            report.append(f"O modelo com menor RMSE é **{best_overall['model_type']}** "
                         f"com temperatura={best_overall['temperature']} e alpha={best_overall['alpha']}.")
            report.append("")
            
            report.append("### Melhor Modelo em Fidelidade ao Professor")
            report.append(f"O modelo com maior correlação com o professor é **{best_fidelity['model_type']}** "
                         f"com temperatura={best_fidelity['temperature']} e alpha={best_fidelity['alpha']}.")
            report.append("")
            
            # Verifica se são modelos diferentes
            if (best_overall['model_type'] != best_fidelity['model_type'] or
                best_overall['temperature'] != best_fidelity['temperature'] or
                best_overall['alpha'] != best_fidelity['alpha']):
                
                report.append("### Trade-off Precisão vs. Fidelidade")
                report.append("Observa-se um trade-off entre a precisão (menor RMSE) e "
                             "a fidelidade ao professor (maior correlação). Considere seu objetivo "
                             "ao escolher o modelo final.")
                report.append("")
            
        except Exception as e:
            report.append("Não foi possível gerar conclusões devido a: " + str(e))
            report.append("")
        
        # Salva o relatório
        report_text = '\n'.join(report)
        report_path = os.path.join(self.config.output_dir, "ts_distillation_report.md")
        
        with open(report_path, 'w') as f:
            f.write(report_text)
            
        self.config.log_info(f"Relatório salvo em {report_path}")
        
        return report_text
    
    def visualize_best_model(self, 
                             X: np.ndarray, 
                             y: np.ndarray, 
                             metric: str = 'rmse',
                             minimize: bool = True,
                             dates: Optional[pd.DatetimeIndex] = None) -> plt.Figure:
        """
        Visualiza as previsões do melhor modelo.
        
        Args:
            X: Features de entrada
            y: Valores alvo
            metric: Métrica para selecionar o melhor modelo
            minimize: Se a métrica deve ser minimizada
            dates: Índice de datas para o eixo x
            
        Returns:
            Figura matplotlib
        """
        if self.results_df is None:
            raise ValueError("Sem resultados disponíveis. Execute run() primeiro.")
            
        # Encontra o melhor modelo
        best_config = self.find_best_model(metric=metric, minimize=minimize)
        
        model_type = best_config['model_type']
        temperature = best_config['temperature']
        alpha = best_config['alpha']
        
        # Obtém o modelo treinado
        best_model = self.get_trained_model(model_type, temperature, alpha)
        
        # Plota as previsões
        fig = best_model.plot_predictions(
            X=X, 
            y_true=y, 
            title=f"Previsões do Melhor Modelo ({model_type}, temp={temperature}, alpha={alpha})",
            filename=os.path.join(self.config.output_dir, f"best_model_predictions_{metric}.png"),
            dates=dates
        )
        
        return fig
    
    def create_visualizations(self, X: np.ndarray, y: np.ndarray, dates: Optional[pd.DatetimeIndex] = None):
        """
        Cria e salva todas as visualizações.
        
        Args:
            X: Features de entrada
            y: Valores alvo
            dates: Índice de datas para o eixo x
        """
        if self.results_df is None:
            raise ValueError("Sem resultados disponíveis. Execute run() primeiro.")
            
        # 1. Comparação de modelos por RMSE
        try:
            self.plot_model_comparison(metric='rmse')
            self.config.log_info("Criada visualização de comparação por RMSE")
        except Exception as e:
            self.config.log_info(f"Erro ao criar visualização de comparação por RMSE: {str(e)}")
        
        # 2. Comparação de modelos por correlação com o professor
        try:
            self.plot_model_comparison(metric='teacher_student_correlation')
            self.config.log_info("Criada visualização de comparação por correlação")
        except Exception as e:
            self.config.log_info(f"Erro ao criar visualização de comparação por correlação: {str(e)}")
        
        # 3. Visualização do melhor modelo por RMSE
        try:
            self.visualize_best_model(X, y, metric='rmse', dates=dates)
            self.config.log_info("Criada visualização do melhor modelo por RMSE")
        except Exception as e:
            self.config.log_info(f"Erro ao criar visualização do melhor modelo por RMSE: {str(e)}")
        
        # 4. Visualização do melhor modelo por fidelidade ao professor
        try:
            self.visualize_best_model(X, y, metric='teacher_student_correlation', minimize=False, dates=dates)
            self.config.log_info("Criada visualização do melhor modelo por fidelidade")
        except Exception as e:
            self.config.log_info(f"Erro ao criar visualização do melhor modelo por fidelidade: {str(e)}")
        
        # 5. Matriz de calor do impacto da temperatura e alpha
        try:
            self._plot_heatmap(metric='rmse')
            self.config.log_info("Criada matriz de calor para RMSE")
        except Exception as e:
            self.config.log_info(f"Erro ao criar matriz de calor: {str(e)}")
    
    def _plot_heatmap(self, metric: str = 'rmse', model_type: Optional[str] = None) -> plt.Figure:
        """
        Cria uma matriz de calor mostrando o impacto de temperatura e alpha.
        
        Args:
            metric: Métrica para visualizar
            model_type: Tipo de modelo específico (se None, usa o primeiro)
            
        Returns:
            Figura matplotlib
        """
        if self.results_df is None:
            raise ValueError("Sem resultados disponíveis. Execute run() primeiro.")
            
        if metric not in self.results_df.columns:
            raise ValueError(f"Métrica '{metric}' não disponível nos resultados")
            
        # Filtra para o modelo específico ou usa o primeiro
        if model_type is None:
            model_types = self.results_df['model_type'].unique()
            if len(model_types) == 0:
                raise ValueError("Nenhum modelo disponível nos resultados")
            model_type = model_types[0]
        
        model_data = self.results_df[self.results_df['model_type'] == model_type]
        if model_data.empty:
            raise ValueError(f"Sem dados disponíveis para o modelo '{model_type}'")
            
        # Obtém temperaturas e alphas únicos
        temps = sorted(model_data['temperature'].unique())
        alphas = sorted(model_data['alpha'].unique())
        
        if not temps or not alphas:
            raise ValueError("Sem valores de temperatura ou alpha disponíveis")
            
        # Cria matriz 2D para o heatmap
        values = np.zeros((len(alphas), len(temps)))
        for i, alpha in enumerate(alphas):
            for j, temp in enumerate(temps):
                filtered = model_data[(model_data['alpha'] == alpha) & (model_data['temperature'] == temp)]
                if not filtered.empty and not filtered[metric].isna().all():
                    values[i, j] = filtered[metric].mean()
                else:
                    values[i, j] = np.nan
        
        # Cria o heatmap
        plt.figure(figsize=(12, 8))
        
        # Define o colormap com base na métrica
        is_minimize = metric not in ['r2', 'teacher_student_correlation']
        cmap = 'viridis_r' if is_minimize else 'viridis'
        
        # Cria o heatmap com valores NaN preenchidos
        masked_values = np.ma.masked_invalid(values)
        im = plt.imshow(masked_values, cmap=cmap, aspect='auto')
        plt.colorbar(im, label=metric.upper())
        
        # Adiciona rótulos
        plt.xticks(range(len(temps)), [f"{t:.1f}" for t in temps])
        plt.yticks(range(len(alphas)), [f"{a:.2f}" for a in alphas])
        plt.xlabel('Temperatura', fontsize=12)
        plt.ylabel('Alpha (α)', fontsize=12)
        
        # Adiciona título
        display_metric = metric.upper()
        plt.title(f'{display_metric} para {model_type} por Temperatura e Alpha', 
                fontsize=14, fontweight='bold')
        
        # Adiciona anotações de texto com valores
        for i in range(len(alphas)):
            for j in range(len(temps)):
                if not np.isnan(values[i, j]):
                    text_color = 'white' if im.norm(values[i, j]) > 0.5 else 'black'
                    plt.text(j, i, f'{values[i, j]:.3f}', 
                            ha='center', va='center', color=text_color)
        
        # Adiciona nota sobre a interpretação
        direction = "menores" if is_minimize else "maiores"
        plt.figtext(0.5, 0.01,
                 f"Este heatmap mostra como diferentes combinações de temperatura e alpha afetam {display_metric}.\n"
                 f"Para esta métrica, valores {direction} (mostrados em {'cores mais escuras' if is_minimize else 'cores mais claras'}) são melhores.",
                 ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Salva a figura
        output_path = os.path.join(self.config.output_dir, f'{model_type}_{metric}_heatmap.png')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.config.log_info(f"Heatmap salvo em {output_path}")
        
        return plt.gcf()
        
        self.temperatures = [0.5, 1.0, 2.0, 3.0]
        self.alphas = [0.3, 0.5, 0.7, 0.9]
    
    def customize(
        self,
        model_types: Optional[List[TSModelType]] = None,
        temperatures: Optional[List[float]] = None,
        alphas: Optional[List[float]] = None
    ):
        """
        Personaliza a configuração para experimentos.
        
        Args:
            model_types: Lista de tipos de modelos a testar
            temperatures: Lista de valores de temperatura
            alphas: Lista de valores de alpha
        """
        if model_types is not None:
            self.model_types = model_types
        if temperatures is not None:
            self.temperatures = temperatures
        if alphas is not None:
            self.alphas = alphas
    
    def get_total_configurations(self) -> int:
        """
        Calcula o número total de configurações a testar.
        
        Returns:
            Número total de configurações
        """
        return len(self.model_types) * len(self.temperatures) * len(self.alphas)
    
    def log_info(self, message: str):
        """
        Registra informações se o modo verbose estiver ativado.
        
        Args:
            message: Mensagem a registrar
        """
        if self.verbose:
            print(message)


class TSAutoDistiller:
    """
    Ferramenta automatizada para destilação de conhecimento em séries temporais.
    
    Automatiza o processo testando múltiplos tipos de modelos, temperaturas e
    valores de alpha para encontrar a configuração ótima para um conjunto de dados.
    """
    
    def __init__(
        self,
        teacher_model: Optional[Any] = None,
        teacher_predictions: Optional[np.ndarray] = None,
        preprocessor: Optional[TimeSeriesPreprocessor] = None,
        output_dir: str = "ts_distillation_results",
        test_size: float = 0.2,
        random_state: int = 42,
        n_trials: int = 10,
        validation_split: float = 0.2,
        window_size: int = 10,
        horizon: int = 1,
        verbose: bool = True
    ):
        """
        Inicializa o AutoDistiller.
        
        Args:
            teacher_model: Modelo professor
            teacher_predictions: Previsões do professor
            preprocessor: Preprocessador para dados
            output_dir: Diretório para resultados
            test_size: Tamanho do conjunto de teste
            random_state: Semente aleatória
            n_trials: Número de trials para otimização
            validation_split: Fração para validação
            window_size: Tamanho da janela
            horizon: Horizonte de previsão
            verbose: Se deve exibir progresso
        """
        # Valida que um dos dois (modelo ou previsões) está presente
        if teacher_model is None and teacher_predictions is None:
            raise ValueError("Deve fornecer teacher_model ou teacher_predictions")
            
        self.teacher_model = teacher_model
        self.teacher_predictions = teacher_predictions
        
        # Inicializa configuração
        self.config = TSConfig(
            output_dir=output_dir,
            test_size=test_size,
            random_state=random_state,
            n_trials=n_trials,
            validation_split=validation_split,
            window_size=window_size,
            horizon=horizon,
            verbose=verbose
        )
        
        # Preprocessador
        self.preprocessor = preprocessor or TimeSeriesPreprocessor(
            window_size=window_size, 
            horizon=horizon
        )
        
        # Resultados e métricas
        self.results_df = None
        self.best_models = {}
    
    def run(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            create_sequences: bool = True,
            store_models: bool = True) -> pd.DataFrame:
        """
        Executa experimentos de destilação.
        
        Args:
            X: Features de entrada
            y: Valores alvo
            create_sequences: Se deve criar sequências para modelos baseados em janelas
            store_models: Se deve armazenar modelos treinados
            
        Returns:
            DataFrame com resultados
        """
        # Pré-processamento
        if not self.preprocessor.is_fitted:
            self.preprocessor.fit(y)
        
        if create_sequences:
            X_processed, y_processed = self.preprocessor.transform(y, create_sequences=True)
        else:
            X_processed = X
            y_processed = y
        
        # Se temos previsões do professor para a série inteira, precisamos processar
        # para corresponder às sequências
        teacher_pred = None
        if self.teacher_predictions is not None:
            if create_sequences:
                # Aplicar o mesmo processamento para create_sequences em previsões do professor
                processed_teacher = self.preprocessor.transform(
                    self.teacher_predictions, create_sequences=False
                )
                # Criar sequências manualmente para previsões do professor
                teacher_sequences = []
                for i in range(len(processed_teacher) - self.config.window_size - self.config.horizon + 1):
                    if self.config.horizon == 1:
                        teacher_sequences.append(processed_teacher[i + self.config.window_size])
                    else:
                        teacher_sequences.append(
                            processed_teacher[(i + self.config.window_size):(i + self.config.window_size + self.config.horizon)]
                        )
                teacher_pred = np.array(teacher_sequences)
            else:
                teacher_pred = self.teacher_predictions
        
        # Inicializa lista para armazenar resultados
        results = []
        
        # Número total de experimentos
        total_experiments = self.config.get_total_configurations()
        self.config.log_info(f"Iniciando {total_experiments} experimentos de destilação")
        
        # Contadores para progresso
        experiment_counter = 0
        
        # Testa todas as combinações
        for model_type in self.config.model_types:
            for temperature in self.config.temperatures:
                for alpha in self.config.alphas:
                    experiment_counter += 1
                    
                    self.config.log_info(
                        f"Experimento {experiment_counter}/{total_experiments}: "
                        f"{model_type.name}, temp={temperature}, alpha={alpha}"
                    )
                    
                    result = self._run_single_experiment(
                        model_type=model_type,
                        temperature=temperature,
                        alpha=alpha,
                        X=X_processed,
                        y=y_processed,
                        teacher_pred=teacher_pred,
                        store_model=store_models
                    )
                    
                    results.append(result)
        
        # Converte resultados para DataFrame
        self.results_df = pd.DataFrame(results)
        
        # Salva resultados
        self.save_results()
        
        return self.results_df
    
    def _run_single_experiment(
        self,
        model_type: TSModelType,
        temperature: float,
        alpha: float,
        X: np.ndarray,
        y: np.ndarray,
        teacher_pred: Optional[np.ndarray] = None,
        store_model: bool = True
    ) -> Dict[str, Any]:
        """
        Executa um único experimento de destilação.
        
        Args:
            model_type: Tipo de modelo
            temperature: Valor de temperatura
            alpha: Valor de alpha
            X: Features de entrada
            y: Valores alvo
            teacher_pred: Previsões do professor
            store_model: Se deve armazenar o modelo treinado
            
        Returns:
            Dicionário com resultados do experimento
        """
        try:
            # Cria modelo de destilação
            distiller = TSKnowledgeDistillation(
                teacher_model=self.teacher_model if teacher_pred is None else None,
                teacher_predictions=teacher_pred,
                student_model_type=model_type,
                temperature=temperature,
                alpha=alpha,
                n_trials=self.config.n_trials,
                validation_split=self.config.validation_split,
                random_state=self.config.random_state,
                window_size=self.config.window_size,
                horizon=self.config.horizon,
                preprocessor=self.preprocessor
            )
            
            # Treina o modelo
            distiller.fit(X, y, verbose=False)
            
            # Avalia o modelo
            metrics = distiller.evaluate(X, y)
            
            # Armazena o modelo se solicitado
            if store_model:
                model_key = f"{model_type.name}_t{temperature}_a{alpha}"
                self.best_models[model_key] = distiller
            
            # Cria resultado com todas as métricas disponíveis
            result = {
                'model_type': model_type.name,
                'temperature': temperature,
                'alpha': alpha,
                'mae': metrics.get('mae', None),
                'mse': metrics.get('mse', None),
                'rmse': metrics.get('rmse', None),
                'mape': metrics.get('mape', None),
                'smape': metrics.get('smape', None),
                'r2': metrics.get('r2', None),
                'kl_divergence': metrics.get('kl_divergence', None),
                'teacher_student_correlation': metrics.get('teacher_student_correlation', None),
                'teacher_student_mse': metrics.get('teacher_student_mse', None),
                'ks_statistic': metrics.get('ks_statistic', None),
                'ks_pvalue': metrics.get('ks_pvalue', None),
                'best_params': str(metrics.get('best_params', {}))
            }
            
            return result
            
        except Exception as e:
            self.config.log_info(f"Erro no experimento: {e}")
            return {
                'model_type': model_type.name,
                'temperature': temperature,
                'alpha': alpha,
                'error': str(e)
            }
    
    def find_best_model(self, metric: str = 'rmse', minimize: bool = True) -> Dict[str, Any]:
        """
        Encontra o melhor modelo com base em uma métrica.
        
        Args:
            metric: Métrica para avaliar modelos
            minimize: Se a métrica deve ser minimizada
            
        Returns:
            Dicionário com a melhor configuração
        """
        if self.results_df is None:
            raise ValueError("Sem resultados disponíveis. Execute run() primeiro.")
            
        try:
            valid_results = self.results_df.dropna(subset=[metric])
            
            if valid_results.empty:
                raise ValueError(f"Sem resultados válidos para a métrica '{metric}'")
                
            # Encontra o melhor índice
            if minimize:
                best_idx = valid_results[metric].idxmin()
                best_value = valid_results.loc[best_idx, metric]
                self.config.log_info(f"Encontrado mínimo {metric} = {best_value}")
            else:
                best_idx = valid_results[metric].idxmax()
                best_value = valid_results.loc[best_idx, metric]
                self.config.log_info(f"Encontrado máximo {metric} = {best_value}")
            
            # Obtém a melhor configuração
            best_config = valid_results.loc[best_idx].to_dict()
            
            # Log de informações úteis
            self.config.log_info(f"Melhor configuração com base em {metric}:")
            for key in ['model_type', 'temperature', 'alpha', metric]:
                if key in best_config:
                    self.config.log_info(f"  {key}: {best_config[key]}")
            
            return best_config
            
        except Exception as e:
            self.config.log_info(f"Erro ao encontrar o melhor modelo para {metric}: {str(e)}")
            raise ValueError(f"Erro ao encontrar o melhor modelo: {str(e)}")
    
    def get_trained_model(self, model_type: str, temperature: float, alpha: float) -> TSKnowledgeDistillation:
        """
        Obtém um modelo treinado com configuração específica.
        
        Args:
            model_type: Tipo de modelo (como string)
            temperature: Valor de temperatura
            alpha: Valor de alpha
            
        Returns:
            Modelo de destilação treinado
        """
        model_key = f"{model_type}_t{temperature}_a{alpha}"
        
        if model_key not in self.best_models:
            raise ValueError(f"Modelo {model_key} não encontrado.")
            
        return self.best_models[model_key]
    
    def save_results(self) -> str:
        """
        Salva resultados em CSV.
        
        Returns:
            Caminho onde os resultados foram salvos
        """
        if self.results_df is None:
            raise ValueError("Sem resultados para salvar")
            
        results_path = os.path.join(self.config.output_dir, "ts_distillation_results.csv")
        self.results_df.to_csv(results_path, index=False)
        self.config.log_info(f"Resultados salvos em {results_path}")
        
        return results_path
    
    def save_best_model(self, metric: str = 'rmse', minimize: bool = True) -> str:
        """
        Encontra o melhor modelo e o salva.
        
        Args:
            metric: Métrica para avaliar modelos
            minimize: Se a métrica deve ser minimizada
            
        Returns:
            Caminho onde o modelo foi salvo
        """
        # Encontra a melhor configuração
        best_config = self.find_best_model(metric=metric, minimize=minimize)
        
        # Obtém o modelo treinado
        model_type = best_config['model_type']
        temperature = best_config['temperature']
        alpha = best_config['alpha']
        
        best_model = self.get_trained_model(model_type, temperature, alpha)
        
        # Define o caminho de saída
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.config.output_dir, 
            f"best_model_{model_type}_t{temperature}_a{alpha}_{timestamp}.pkl"
        )
        
        # Salva o modelo
        best_model.save(output_file)
        
        self.config.log_info(f"Melhor modelo salvo em: {output_file}")
        
        return output_file
    
    def plot_model_comparison(self, metric: str = 'rmse') -> plt.Figure:
        """
        Cria um gráfico comparando modelos com base em uma métrica.
        
        Args:
            metric: Métrica para comparar
            
        Returns:
            Figura matplotlib
        """
        if self.results_df is None:
            raise ValueError("Sem resultados disponíveis. Execute run() primeiro.")
            
        # Filtra resultados válidos
        valid_results = self.results_df.dropna(subset=[metric])
        
        if valid_results.empty:
            raise ValueError(f"Sem resultados válidos para a métrica '{metric}'")
            
        # Agrupa por tipo de modelo
        grouped = valid_results.groupby('model_type').agg({
            metric: ['mean', 'min', 'max', 'std']
        }).reset_index()
        
        # Cria figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(grouped))
        width = 0.3
        
        # Plota valores médios e mínimos
        ax.bar(x, grouped[(metric, 'mean')], width, label=f'Média {metric}', color='royalblue', alpha=0.7)
        ax.bar([i + width for i in x], grouped[(metric, 'min')], width, label=f'Mínimo {metric}', color='darkblue')
        
        # Adiciona barras de erro
        ax.errorbar(x, grouped[(metric, 'mean')], yerr=grouped[(metric, 'std')], fmt='none', color='black', capsize=5)
        
        # Configura o gráfico
        ax.set_xlabel('Tipo de Modelo', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'Comparação de Modelos por {metric.upper()}', fontsize=14, fontweight='bold')
        ax.set_xticks([i + width/2 for i in x])
        ax.set_xticklabels(grouped['model_type'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adiciona valores no topo das barras
        for i, val in enumerate(grouped[(metric, 'mean')]):
            ax.text(i, val + 0.02, f'{val:.4f}', ha='center', fontsize=9)
            ax.text(i + width, grouped[(metric, 'min')][i] + 0.02, f'{grouped[(metric, "min")][i]:.4f}', ha='center', fontsize=9)
        
        # Salva a figura
        output_path = os.path.join(self.config.output_dir, f'model_comparison_{metric}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        self.config.log_info(f"Gráfico de comparação salvo em {output_path}")
        
        return fig
    
    def generate_report(self) -> str:
        """
        Gera um relatório com resultados.
        
        Returns:
            Relatório em formato markdown
        """
        if self.results_df is None:
            raise ValueError("Sem resultados disponíveis. Execute run() primeiro.")
            
        report = ["# Relatório de Destilação de Séries Temporais\n"]
        
        # Adiciona informações gerais
        report.append("## Informações Gerais")
        report.append(f"- Número de modelos testados: {len(self.config.model_types)}")
        report.append(f"- Temperaturas testadas: {self.config.temperatures}")
        report.append(f"- Valores de alpha testados: {self.config.alphas}")
        report.append(f"- Total de configurações: {self.config.get_total_configurations()}")
        report.append(f"- Data/hora do relatório: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Adiciona melhores configurações para diferentes métricas
        report.append("## Melhores Configurações")
        
        # Define todas as métricas para encontrar os melhores modelos
        metrics = [
            ('rmse', True, 'RMSE'),
            ('mae', True, 'MAE'),
            ('mape', True, 'MAPE'),
            ('r2', False, 'R²'),
            ('kl_divergence', True, 'KL Divergence'),
            ('teacher_student_correlation', False, 'Teacher-Student Correlation'),
            ('teacher_student_mse', True, 'Teacher-Student MSE')
        ]