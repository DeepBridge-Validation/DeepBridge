import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from scipy import stats

class TimeSeriesPreprocessor:
    """
    Classe para pré-processamento de séries temporais.
    Implementa funcionalidades para:
    - Criação de features baseadas em lags
    - Transformações para estacionariedade
    - Normalização de dados
    - Janelamento de dados para treino
    - Tratamento de outliers e valores ausentes
    - Análise e decomposição de sazonalidade
    """
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        lags: Optional[List[int]] = None,
        window_size: int = 10,
        horizon: int = 1,
        differencing: int = 0,
        log_transform: bool = False,
        remove_outliers: bool = False,
        outlier_method: str = 'zscore',
        outlier_threshold: float = 3.0,
        target_column: Optional[str] = None,
        date_column: Optional[str] = None,
        freq: Optional[str] = None,
        handle_missing: str = 'interpolate',
        seasonal_period: Optional[int] = None,
        seasonal_adjust: bool = False
    ):
        """
        Inicializa o preprocessador de séries temporais.
        
        Args:
            scaler_type: Tipo de escala ('standard', 'minmax', 'robust', None)
            lags: Lista de lags a serem usados como features
            window_size: Tamanho da janela para sequências de dados
            horizon: Horizonte de previsão (quantos períodos à frente prever)
            differencing: Ordem de diferenciação para estacionariedade
            log_transform: Se deve aplicar transformação logarítmica
            remove_outliers: Se deve remover outliers
            outlier_method: Método para detectar outliers ('zscore', 'iqr')
            outlier_threshold: Limite para considerar um valor como outlier
            target_column: Nome da coluna alvo (para DataFrames)
            date_column: Nome da coluna de data (para DataFrames)
            freq: Frequência dos dados ('D' para diário, 'M' para mensal, etc.)
            handle_missing: Como tratar valores ausentes ('interpolate', 'ffill', 'bfill', 'mean', 'drop')
            seasonal_period: Período de sazonalidade (ex: 12 para dados mensais com padrão anual)
            seasonal_adjust: Se deve remover a componente sazonal
        """
        self.scaler_type = scaler_type
        self.lags = lags if lags is not None else [1, 2, 3, 7]
        self.window_size = window_size
        self.horizon = horizon
        self.differencing = differencing
        self.log_transform = log_transform
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.target_column = target_column
        self.date_column = date_column
        self.freq = freq
        self.handle_missing = handle_missing
        self.seasonal_period = seasonal_period
        self.seasonal_adjust = seasonal_adjust
        
        # Inicialização dos transformadores
        self.scaler = None
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
            
        # Atributos para armazenar estatísticas e metadados
        self.data_stats = {}
        self.is_fitted = False
        self.original_min = None
        self.original_max = None
        self.log_offset = 0
        self.diff_values = []
        self.outlier_indices = []
        self.missing_indices = []
        self.seasonal_components = None
    
    def fit(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> 'TimeSeriesPreprocessor':
        """
        Ajusta o preprocessador aos dados.
        
        Args:
            data: Dados de série temporal (DataFrame, Series ou array)
            
        Returns:
            self: O preprocessador ajustado
        """
        # Extrai a série temporal se for DataFrame
        series = self._extract_series(data)
        
        # Trata valores ausentes
        series = self._handle_missing_values(series)
        
        # Armazena estatísticas originais
        self.original_min = series.min()
        self.original_max = series.max()
        self.data_stats['mean'] = series.mean()
        self.data_stats['std'] = series.std()
        
        # Detecta e trata outliers
        if self.remove_outliers:
            series, self.outlier_indices = self._remove_outliers(series)
        
        # Analisa sazonalidade se solicitado
        if self.seasonal_period is not None:
            self._analyze_seasonality(series)
            
            # Remove componente sazonal se solicitado
            if self.seasonal_adjust:
                series = self._remove_seasonality(series)
        
        # Aplica transformação logarítmica se necessário
        if self.log_transform:
            # Adiciona offset se houver valores negativos ou zero
            if series.min() <= 0:
                self.log_offset = abs(series.min()) + 1
                series = series + self.log_offset
                
        # Armazena valores para diferenciação se necessário
        if self.differencing > 0:
            self.diff_values = []
            temp_series = series.copy()
            for i in range(self.differencing):
                self.diff_values.append(temp_series.iloc[0])
                temp_series = temp_series.diff().dropna()
        
        # Ajusta o scaler se necessário
        if self.scaler is not None:
            # Reshape para 2D se for 1D
            reshaped_data = series.values.reshape(-1, 1) if len(series.shape) == 1 else series.values
            self.scaler.fit(reshaped_data)
            
        # Testa estacionariedade
        self._test_stationarity(series)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.DataFrame, pd.Series, np.ndarray], create_sequences: bool = True) -> Union[pd.Series, Tuple]:
        """
        Transforma os dados aplicando pré-processamento.
        
        Args:
            data: Dados de série temporal
            create_sequences: Se deve criar sequências para modelo de janela deslizante
            
        Returns:
            Tuple: Depende de create_sequences:
                - Se True: (X, y) onde X são as sequências de entrada e y os valores alvo
                - Se False: Série transformada
        """
        if not self.is_fitted:
            raise ValueError("O preprocessador deve ser ajustado (fit) antes de transformar dados")
            
        # Extrai a série temporal se for DataFrame
        series = self._extract_series(data)
        
        # Trata valores ausentes
        series = self._handle_missing_values(series)
        transformed_series = series.copy()
        
        # Trata outliers se necessário
        if self.remove_outliers:
            transformed_series, _ = self._remove_outliers(transformed_series)
        
        # Remove sazonalidade se configurado
        if self.seasonal_adjust and self.seasonal_components is not None:
            transformed_series = self._remove_seasonality(transformed_series)
            
        # Aplica transformação logarítmica
        if self.log_transform:
            if self.log_offset > 0:
                transformed_series = transformed_series + self.log_offset
            transformed_series = np.log1p(transformed_series)
            
        # Aplica diferenciação
        if self.differencing > 0:
            for i in range(self.differencing):
                transformed_series = transformed_series.diff().dropna()
                
        # Aplica normalização
        if self.scaler is not None:
            # Reshape para 2D se for 1D
            reshaped_data = transformed_series.values.reshape(-1, 1) if len(transformed_series.shape) == 1 else transformed_series.values
            transformed_series = pd.Series(
                self.scaler.transform(reshaped_data).flatten(), 
                index=transformed_series.index[-len(transformed_series):]
            )
            
        if not create_sequences:
            return transformed_series
            
        # Cria sequências para modelagem baseada em janelas
        return self._create_sequences(transformed_series)
    
    def inverse_transform(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Reverte as transformações para obter valores na escala original.
        
        Args:
            data: Dados transformados
            
        Returns:
            Dados na escala original
        """
        if not self.is_fitted:
            raise ValueError("O preprocessador deve ser ajustado (fit) antes de reverter transformações")
            
        # Converte para array np se necessário
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            values = data.values
        else:
            values = data
            
        # Reshape para 2D se for 1D
        needs_reshape = len(values.shape) == 1
        if needs_reshape:
            values = values.reshape(-1, 1)
            
        # Reverte normalização
        if self.scaler is not None:
            values = self.scaler.inverse_transform(values)
            
        # Reverte diferenciação
        if self.differencing > 0:
            # Precisa de um índice para usar diff
            temp_df = pd.DataFrame(values)
            
            # Integra os valores usando os valores iniciais armazenados
            for i in range(self.differencing-1, -1, -1):
                temp_df = temp_df.cumsum()
                # Adiciona o valor inicial na primeira posição
                first_val = self.diff_values[i]
                temp_df = first_val + temp_df
                
            values = temp_df.values
            
        # Reverte transformação logarítmica
        if self.log_transform:
            values = np.expm1(values)
            if self.log_offset > 0:
                values = values - self.log_offset
                
        # Readiciona sazonalidade se foi removida
        if self.seasonal_adjust and self.seasonal_components is not None:
            # Esta lógica precisa ser implementada com cuidado
            # Depende de como a sazonalidade foi removida e armazenada
            pass
                
        # Volta para o formato original
        if needs_reshape:
            values = values.flatten()
            
        return values
    
    def create_lag_features(self, data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                          include_original: bool = True) -> pd.DataFrame:
        """
        Cria features de lag a partir da série temporal.
        
        Args:
            data: Dados de série temporal
            include_original: Se deve incluir a série original como coluna 'y'
            
        Returns:
            DataFrame com features de lag
        """
        # Extrai a série temporal se for DataFrame
        series = self._extract_series(data)
        
        # Cria DataFrame para features
        lag_df = pd.DataFrame(index=series.index)
        if include_original:
            lag_df['y'] = series.values
        
        # Adiciona lags como colunas
        for lag in self.lags:
            lag_df[f'lag_{lag}'] = series.shift(lag)
            
        # Remove linhas com NaN
        lag_df = lag_df.dropna()
        
        return lag_df
    
    def create_time_features(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Cria features baseadas em tempo (hora, dia da semana, mês, etc).
        
        Args:
            data: Dados de série temporal com índice de data ou coluna de data
            
        Returns:
            DataFrame com features de tempo
        """
        if isinstance(data, pd.Series):
            df = pd.DataFrame(data)
            col_name = data.name if data.name is not None else 'value'
            df.columns = [col_name]
        else:
            df = data.copy()
            
        # Identifica a coluna de data
        date_col = None
        if self.date_column and self.date_column in df.columns:
            date_col = self.date_column
        elif isinstance(df.index, pd.DatetimeIndex):
            # Usa o índice de data
            df['date'] = df.index
            date_col = 'date'
        else:
            raise ValueError("Não foi possível identificar uma coluna de data. Forneça date_column ou use índice de data.")
            
        # Cria features de tempo
        df['hour'] = df[date_col].dt.hour
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['dayofmonth'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['year'] = df[date_col].dt.year
        
        # Features cíclicas para hora, dia da semana e mês
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        
        # Remove a coluna de data temporária se foi criada
        if date_col == 'date' and self.date_column is None:
            df = df.drop('date', axis=1)
            
        return df
    
    def _extract_series(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        """
        Extrai a série temporal dos dados fornecidos.
        
        Args:
            data: Dados em vários formatos possíveis
            
        Returns:
            pd.Series: Série temporal extraída
        """
        if isinstance(data, pd.DataFrame):
            if self.target_column is None:
                raise ValueError("target_column deve ser especificado para dados em DataFrame")
            return data[self.target_column]
        elif isinstance(data, pd.Series):
            return data
        elif isinstance(data, np.ndarray):
            if len(data.shape) > 1 and data.shape[1] > 1:
                raise ValueError("Para arrays 2D, especifique a coluna alvo")
            return pd.Series(data.flatten())
        else:
            raise ValueError(f"Tipo de dados não suportado: {type(data)}")
    
    def _handle_missing_values(self, series: pd.Series) -> pd.Series:
        """
        Trata valores ausentes na série temporal.
        
        Args:
            series: Série temporal
            
        Returns:
            Série com valores ausentes tratados
        """
        # Armazena índices dos valores ausentes
        self.missing_indices = series.index[series.isna()]
        
        if series.isna().any():
            if self.handle_missing == 'interpolate':
                return series.interpolate(method='time' if isinstance(series.index, pd.DatetimeIndex) else 'linear')
            elif self.handle_missing == 'ffill':
                return series.ffill()
            elif self.handle_missing == 'bfill':
                return series.bfill()
            elif self.handle_missing == 'mean':
                return series.fillna(series.mean())
            elif self.handle_missing == 'drop':
                return series.dropna()
            else:
                raise ValueError(f"Método para tratar valores ausentes não reconhecido: {self.handle_missing}")
        
        return series
    
    def _remove_outliers(self, series: pd.Series) -> Tuple[pd.Series, List[int]]:
        """
        Detecta e remove outliers da série temporal.
        
        Args:
            series: Série temporal
            
        Returns:
            Tuple contendo a série sem outliers e os índices dos outliers
        """
        outlier_indices = []
        
        if self.outlier_method == 'zscore':
            # Método Z-score
            z_scores = np.abs(stats.zscore(series))
            outlier_indices = np.where(z_scores > self.outlier_threshold)[0]
        elif self.outlier_method == 'iqr':
            # Método IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
            outlier_indices = series[(series < lower_bound) | (series > upper_bound)].index
        else:
            raise ValueError(f"Método para detectar outliers não reconhecido: {self.outlier_method}")
            
        # Armazena os valores originais antes de substituir
        outlier_values = series.iloc[outlier_indices].copy()
        self.data_stats['outliers'] = {
            'indices': outlier_indices,
            'values': outlier_values
        }
        
        # Substitui outliers pela média ou mediana (poderia ser mais sofisticado)
        if len(outlier_indices) > 0:
            if self.handle_missing == 'mean':
                series.iloc[outlier_indices] = series.mean()
            else:
                # Use interpolação para substituir outliers
                series.iloc[outlier_indices] = np.nan
                series = series.interpolate()
        
        return series, outlier_indices
    
    def _analyze_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analisa a sazonalidade da série temporal.
        
        Args:
            series: Série temporal
            
        Returns:
            Dicionário com componentes sazonais e estatísticas
        """
        try:
            # Verifica se temos pontos suficientes
            if len(series) < 2 * self.seasonal_period:
                warnings.warn(f"Série muito curta para análise de sazonalidade com período {self.seasonal_period}")
                return {}
            
            # Usa STL para decomposição mais robusta
            if isinstance(series.index, pd.DatetimeIndex):
                # Use seasonal_decompose para séries com índice de data
                decomposition = seasonal_decompose(
                    series, 
                    model='additive', 
                    period=self.seasonal_period
                )
            else:
                # Use STL para mais flexibilidade
                stl = STL(series, period=self.seasonal_period)
                decomposition = stl.fit()
            
            # Armazena componentes
            self.seasonal_components = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'resid': decomposition.resid
            }
            
            # Calcula estatísticas de sazonalidade
            seasonal_strength = 1 - decomposition.resid.var() / decomposition.seasonal.var()
            trend_strength = 1 - decomposition.resid.var() / decomposition.trend.var()
            
            self.data_stats['seasonality'] = {
                'seasonal_strength': seasonal_strength,
                'trend_strength': trend_strength,
                'period': self.seasonal_period
            }
            
            return self.seasonal_components
            
        except Exception as e:
            warnings.warn(f"Erro na análise de sazonalidade: {str(e)}")
            return {}
    
    def _remove_seasonality(self, series: pd.Series) -> pd.Series:
        """
        Remove a componente sazonal da série temporal.
        
        Args:
            series: Série temporal
            
        Returns:
            Série sem componente sazonal
        """
        if self.seasonal_components is None or 'seasonal' not in self.seasonal_components:
            # Tenta analisar a sazonalidade primeiro
            self._analyze_seasonality(series)
            
        if self.seasonal_components is not None and 'seasonal' in self.seasonal_components:
            # Alinha os índices
            seasonal = self.seasonal_components['seasonal']
            
            # Se os tamanhos não correspondem, pode ser necessário recalcular a decomposição
            if len(seasonal) != len(series):
                warnings.warn("Componente sazonal e série têm tamanhos diferentes. Recalculando decomposição.")
                self._analyze_seasonality(series)
                seasonal = self.seasonal_components['seasonal']
                
                # Se ainda não correspondem, desistimos
                if len(seasonal) != len(series):
                    warnings.warn("Não foi possível alinhar componente sazonal. Retornando série original.")
                    return series
            
            # Remove componente sazonal
            deseasonalized = series - seasonal
            return deseasonalized
        
        # Retorna a série original se não houver informação sazonal
        warnings.warn("Sem informação sazonal disponível. Retornando série original.")
        return series
    
    def _create_sequences(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências para modelagem baseada em janelas.
        
        Args:
            series: Série temporal
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (sequências de entrada) e y (valores alvo)
        """
        data = series.values
        X, y = [], []
        
        for i in range(len(data) - self.window_size - self.horizon + 1):
            # Sequência de entrada
            X.append(data[i:(i + self.window_size)])
            # Valor(es) alvo
            if self.horizon == 1:
                y.append(data[i + self.window_size])
            else:
                y.append(data[(i + self.window_size):(i + self.window_size + self.horizon)])
                
        # Verifica se conseguimos criar sequências
        if not X or not y:
            raise ValueError(
                f"Não foi possível criar sequências. Série muito curta para window_size={self.window_size} "
                f"e horizon={self.horizon}. Comprimento da série: {len(data)}"
            )
        
        return np.array(X), np.array(y)
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Testa estacionariedade da série usando testes ADF e KPSS.
        
        Args:
            series: Série temporal
            
        Returns:
            Dict: Resultados dos testes
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Teste ADF - Hipótese nula: a série tem raiz unitária (não é estacionária)
            try:
                adf_result = adfuller(series.dropna())
                adf_pvalue = adf_result[1]
                adf_stationary = adf_pvalue < 0.05  # Rejeita H0 se p < 0.05
                adf_critical = adf_result[4]
            except Exception as e:
                adf_pvalue = None
                adf_stationary = None
                adf_critical = None
                
            # Teste KPSS - Hipótese nula: a série é estacionária
            try:
                kpss_result = kpss(series.dropna())
                kpss_pvalue = kpss_result[1]
                kpss_stationary = kpss_pvalue >= 0.05  # Não rejeita H0 se p >= 0.05
                kpss_critical = kpss_result[3]
            except Exception as e:
                kpss_pvalue = None
                kpss_stationary = None
                kpss_critical = None
                
        # Armazena resultados
        self.data_stats['stationarity'] = {
            'adf_pvalue': adf_pvalue,
            'adf_stationary': adf_stationary,
            'adf_critical': adf_critical,
            'kpss_pvalue': kpss_pvalue,
            'kpss_stationary': kpss_stationary,
            'kpss_critical': kpss_critical,
            'differencing_applied': self.differencing,
            'log_transform_applied': self.log_transform
        }
        
        return self.data_stats['stationarity']
    
    def plot_preprocessed(self, data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                         title: str = "Original vs Preprocessado", 
                         filename: Optional[str] = None) -> plt.Figure:
        """
        Visualiza os dados antes e depois do pré-processamento.
        
        Args:
            data: Dados originais
            title: Título do gráfico
            filename: Nome do arquivo para salvar o gráfico (opcional)
            
        Returns:
            Figura matplotlib
        """
        if not self.is_fitted:
            raise ValueError("O preprocessador deve ser ajustado (fit) antes de visualizar")
            
        # Extrai a série temporal se for DataFrame
        original_series = self._extract_series(data)
        
        # Aplica pré-processamento sem criar sequências
        processed_series = self.transform(data, create_sequences=False)
        
        # Cria visualização
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Série original
        axes[0].plot(original_series, label='Original')
        axes[0].set_title('Série Original')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Série pré-processada
        axes[1].plot(processed_series, color='red', label='Pré-processada')
        axes[1].set_title('Série Pré-processada')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Boxplot para comparação de distribuições
        data_to_plot = [original_series, processed_series]
        axes[2].boxplot(data_to_plot, labels=['Original', 'Pré-processada'])
        axes[2].set_title('Comparação de Distribuições')
        axes[2].grid(True, alpha=0.3)
        
        # Ajusta layout
        plt.tight_layout()
        fig.suptitle(title, fontsize=16, y=1.02)
        
        # Salva se necessário
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_seasonal_decomposition(self, data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                                  filename: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plota a decomposição sazonal da série.
        
        Args:
            data: Dados de série temporal
            filename: Nome do arquivo para salvar o gráfico (opcional)
            
        Returns:
            Figura matplotlib ou None se não houver decomposição sazonal
        """
        if self.seasonal_components is None:
            # Extrai a série temporal se for DataFrame
            series = self._extract_series(data)
            
            # Tenta fazer decomposição
            self._analyze_seasonality(series)
            
            if self.seasonal_components is None:
                warnings.warn("Não foi possível fazer decomposição sazonal")
                return None
                
        # Cria visualização
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        # Série original
        original_series = self._extract_series(data)
        axes[0].plot(original_series, label='Original')
        axes[0].set_title('Série Original')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Componente de tendência
        axes[1].plot(self.seasonal_components['trend'], color='blue', label='Tendência')
        axes[1].set_title('Componente de Tendência')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Componente sazonal
        axes[2].plot(self.seasonal_components['seasonal'], color='green', label='Sazonalidade')
        axes[2].set_title('Componente Sazonal')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Componente residual
        axes[3].plot(self.seasonal_components['resid'], color='red', label='Resíduo')
        axes[3].set_title('Componente Residual')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        # Ajusta layout
        plt.tight_layout()
        fig.suptitle(f'Decomposição Sazonal (Período = {self.seasonal_period})', fontsize=16, y=1.02)
        
        # Salva se necessário
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        return fig
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre os dados processados.
        
        Returns:
            Dicionário com estatísticas e informações sobre o pré-processamento
        """
        if not self.is_fitted:
            raise ValueError("O preprocessador deve ser ajustado (fit) antes de obter estatísticas")
            
        # Compila todas as estatísticas em um único dicionário
        stats = {
            'basic': {
                'min': self.original_min,
                'max': self.original_max,
                'mean': self.data_stats.get('mean'),
                'std': self.data_stats.get('std')
            },
            'preprocessing': {
                'differencing': self.differencing,
                'log_transform': self.log_transform,
                'scaler_type': self.scaler_type,
                'window_size': self.window_size,
                'horizon': self.horizon,
                'lags': self.lags,
                'remove_outliers': self.remove_outliers,
                'outlier_method': self.outlier_method if self.remove_outliers else None,
                'missing_count': len(self.missing_indices),
                'outlier_count': len(self.outlier_indices) if hasattr(self, 'outlier_indices') else 0
            }
        }
        
        # Adiciona informações de estacionariedade, se disponíveis
        if 'stationarity' in self.data_stats:
            stats['stationarity'] = self.data_stats['stationarity']
            
        # Adiciona informações de sazonalidade, se disponíveis
        if 'seasonality' in self.data_stats:
            stats['seasonality'] = self.data_stats['seasonality']
            
        return stats
        
    def detect_optimal_differencing(self, data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                                  max_diff: int = 2) -> int:
        """
        Detecta a ordem ótima de diferenciação para tornar a série estacionária.
        
        Args:
            data: Dados da série temporal
            max_diff: Ordem máxima de diferenciação a testar
            
        Returns:
            Ordem ótima de diferenciação
        """
        # Extrai a série temporal se for DataFrame
        series = self._extract_series(data)
        
        best_pvalue = 1.0
        best_diff = 0
        
        for d in range(max_diff + 1):
            # Aplicar diferenciação
            if d == 0:
                test_series = series
            else:
                test_series = series.diff(d).dropna()
                
            # Teste ADF
            try:
                result = adfuller(test_series.dropna())
                pvalue = result[1]
                
                if pvalue < 0.05:  # Série é estacionária
                    return d
                
                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_diff = d
            except:
                continue
                
        return best_diff
        
    def detect_optimal_parameters(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Detecta parâmetros ótimos para pré-processamento.
        
        Args:
            data: Dados da série temporal
            
        Returns:
            Dicionário com parâmetros ótimos recomendados
        """
        # Extrai a série temporal se for DataFrame
        series = self._extract_series(data)
        
        # Verificar se log transform seria útil
        skewness = series.skew()
        log_recommended = abs(skewness) > 1.0
        
        # Detectar ordem ótima de diferenciação
        diff_order = self.detect_optimal_differencing(series)
        
        # Verificar se há sazonalidade
        seasonal_period = None
        if isinstance(series.index, pd.DatetimeIndex):
            # Tenta detectar frequência com base no índice
            if series.index.freq is not None:
                freq = series.index.freq.name
                if freq in ['D', 'B']:  # Diário ou dia útil
                    seasonal_period = 7  # Semanal
                elif freq in ['W', 'W-SUN', 'W-MON']:  # Semanal
                    seasonal_period = 52  # Anual
                elif freq in ['M', 'MS']:  # Mensal
                    seasonal_period = 12  # Anual
                elif freq in ['Q', 'QS']:  # Trimestral
                    seasonal_period = 4  # Anual
                elif freq in ['H', 'min']:  # Horário ou minuto
                    seasonal_period = 24  # Diário
            else:
                # Tenta inferir com base no intervalo entre pontos
                timedeltas = series.index[1:] - series.index[:-1]
                avg_delta = pd.to_timedelta(pd.Series(timedeltas).mean())
                
                if avg_delta <= pd.Timedelta(minutes=30):
                    seasonal_period = 24 * 2  # 30 min -> 24h * 2
                elif avg_delta <= pd.Timedelta(hours=1):
                    seasonal_period = 24  # Horário -> diário
                elif avg_delta <= pd.Timedelta(days=1):
                    seasonal_period = 7  # Diário -> semanal
                elif avg_delta <= pd.Timedelta(days=7):
                    seasonal_period = 52  # Semanal -> anual
                elif avg_delta <= pd.Timedelta(days=31):
                    seasonal_period = 12  # Mensal -> anual
        
        # Verificar número de outliers
        if self.remove_outliers:
            _, outlier_indices = self._remove_outliers(series)
            outlier_percentage = len(outlier_indices) / len(series) * 100
            outliers_significant = outlier_percentage > 5
        else:
            outliers_significant = False
            
        # Recomendações
        recommendations = {
            'differencing': diff_order,
            'log_transform': log_recommended,
            'seasonal_period': seasonal_period,
            'seasonal_adjust': seasonal_period is not None,
            'remove_outliers': outliers_significant,
            'scaler_type': 'robust' if outliers_significant else 'standard'
        }
        
        return recommendations
    
    def summary(self) -> str:
        """
        Retorna um resumo textual do pré-processamento aplicado.
        
        Returns:
            String com resumo
        """
        if not self.is_fitted:
            return "Preprocessador não ajustado. Chame fit() primeiro."
            
        stats = self.get_statistics()
        
        # Monta o resumo
        summary_text = [
            "Resumo do Pré-processamento de Séries Temporais",
            "================================================",
            f"Diferenciação: {self.differencing}",
            f"Transformação log: {'Sim' if self.log_transform else 'Não'}",
            f"Tipo de escala: {self.scaler_type}",
            f"Tamanho da janela: {self.window_size}",
            f"Horizonte de previsão: {self.horizon}",
            f"Remoção de outliers: {'Sim' if self.remove_outliers else 'Não'}",
            f"Ajuste sazonal: {'Sim' if self.seasonal_adjust else 'Não'}"
        ]
        
        # Informações de outliers
        if 'preprocessing' in stats and 'outlier_count' in stats['preprocessing']:
            summary_text.append(f"Outliers detectados: {stats['preprocessing']['outlier_count']}")
            
        # Informações de valores ausentes
        if 'preprocessing' in stats and 'missing_count' in stats['preprocessing']:
            summary_text.append(f"Valores ausentes: {stats['preprocessing']['missing_count']}")
            
        # Informações de estacionariedade
        if 'stationarity' in stats:
            adf = stats['stationarity'].get('adf_stationary')
            kpss = stats['stationarity'].get('kpss_stationary')
            
            if adf is not None and kpss is not None:
                if adf and kpss:
                    stationarity = "Série é estacionária (confirmado por ADF e KPSS)"
                elif adf:
                    stationarity = "Série pode ser estacionária (ADF indica estacionariedade, KPSS não)"
                elif kpss:
                    stationarity = "Série pode ser estacionária (KPSS indica estacionariedade, ADF não)"
                else:
                    stationarity = "Série não é estacionária (confirmado por ADF e KPSS)"
                
                summary_text.append(f"Estacionariedade: {stationarity}")
                
        # Informações de sazonalidade
        if 'seasonality' in stats:
            seasonal_strength = stats['seasonality'].get('seasonal_strength')
            if seasonal_strength is not None:
                if seasonal_strength > 0.6:
                    season_desc = f"Forte componente sazonal (força: {seasonal_strength:.2f})"
                elif seasonal_strength > 0.3:
                    season_desc = f"Componente sazonal moderado (força: {seasonal_strength:.2f})"
                else:
                    season_desc = f"Componente sazonal fraco (força: {seasonal_strength:.2f})"
                
                summary_text.append(f"Sazonalidade: {season_desc}")
                
        return "\n".join(summary_text)