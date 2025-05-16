import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# Implementação da classe CRQR
class CRQR:
    """
    Conformalized Residual Quantile Regression (CRQR)
    
    Uma abordagem model-agnostic para avaliar a confiabilidade de modelos de regressão 
    dentro do framework de predição conformal.
    """
    
    def __init__(self, base_model=None, alpha=0.1, test_size=0.6, calib_ratio=1/3, random_state=None):
        """
        Inicializador da classe CRQR.
        
        Parâmetros:
        - base_model: modelo de regressão base (padrão: None, usa o modelo padrão)
        - alpha: nível de significância (padrão: 0.1 para intervalos de confiança de 90%)
        - test_size: proporção dos dados a serem usados para teste+calibração (padrão: 0.6 = 60%)
        - calib_ratio: proporção do conjunto test_size a ser usada para calibração (padrão: 1/3, 
                      resultando em 20% do total para calibração e 40% para teste)
        - random_state: semente aleatória para reprodutibilidade
        """
        self.alpha = alpha
        self.test_size = test_size
        self.calib_ratio = calib_ratio
        self.random_state = random_state
        
        # Calcula as proporções efetivas
        self.train_size = 1 - test_size
        self.calib_size = test_size * calib_ratio
        self.test_size_final = test_size * (1 - calib_ratio)
        
        # Modelo base para regressão
        if base_model is None:
            self.base_model = None  # Será configurado durante o fit
        else:
            self.base_model = base_model
            
        # Modelos de regressão quantil
        self.quantile_model_lower = None
        self.quantile_model_upper = None
        
        # Valor de calibração
        self.q_hat = None
        
    def fit(self, X, y):
        """
        Treina o modelo base e os modelos de regressão quantil.
        
        Parâmetros:
        - X: features de treinamento
        - y: target de treinamento
        """
        # Divisão dos dados em conjuntos de treinamento e conjunto temporário (calibração + teste)
        # com base nos parâmetros definidos pelo usuário
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Divisão do conjunto temporário em calibração e teste
        # calib_ratio do conjunto temporário para calibração, o resto para teste
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=(1 - self.calib_ratio), 
            random_state=self.random_state
        )
        
        # Se não foi fornecido um modelo base, cria um modelo padrão
        if self.base_model is None:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.base_model = HistGradientBoostingRegressor(random_state=self.random_state)
        
        # Treina o modelo base com os dados de treinamento
        self.base_model.fit(X_train, y_train)
        
        # Prediz os valores para os dados de treinamento
        y_pred_train = self.base_model.predict(X_train)
        
        # Calcula os resíduos
        residuals = y_train - y_pred_train
        
        # Configura os modelos de regressão quantil
        try:
            # Tenta usar HistGradientBoostingRegressor com loss='quantile'
            import sklearn
            if sklearn.__version__ >= '1.1':
                from sklearn.ensemble import HistGradientBoostingRegressor
                self.quantile_model_lower = HistGradientBoostingRegressor(
                    loss='quantile', quantile=self.alpha/2, max_depth=5, random_state=self.random_state)
                self.quantile_model_upper = HistGradientBoostingRegressor(
                    loss='quantile', quantile=1-self.alpha/2, max_depth=5, random_state=self.random_state)
            else:
                # Para versões mais antigas do sklearn, usa GBDT diretamente
                from sklearn.ensemble import GradientBoostingRegressor
                self.quantile_model_lower = GradientBoostingRegressor(
                    loss='quantile', alpha=self.alpha/2, max_depth=5, random_state=self.random_state)
                self.quantile_model_upper = GradientBoostingRegressor(
                    loss='quantile', alpha=1-self.alpha/2, max_depth=5, random_state=self.random_state)
        except Exception as e:
            # Fallback para GradientBoostingRegressor
            from sklearn.ensemble import GradientBoostingRegressor
            self.quantile_model_lower = GradientBoostingRegressor(
                loss='quantile', alpha=self.alpha/2, max_depth=5, random_state=self.random_state)
            self.quantile_model_upper = GradientBoostingRegressor(
                loss='quantile', alpha=1-self.alpha/2, max_depth=5, random_state=self.random_state)
        
        # Treina os modelos de regressão quantil nos resíduos
        self.quantile_model_lower.fit(X_train, residuals)
        self.quantile_model_upper.fit(X_train, residuals)
        
        # Calibração usando o conjunto de calibração
        y_pred_calib = self.base_model.predict(X_calib)
        residuals_calib = y_calib - y_pred_calib
        
        # Prediz os limites dos resíduos para o conjunto de calibração
        lower_pred = self.quantile_model_lower.predict(X_calib)
        upper_pred = self.quantile_model_upper.predict(X_calib)
        
        # Calcula os scores de conformidade
        scores = np.maximum(lower_pred - residuals_calib, residuals_calib - upper_pred)
        
        # Calcula o quantil para o intervalo de confiança
        n = len(scores)
        level = np.ceil((n+1) * (1-self.alpha)) / n
        self.q_hat = np.quantile(scores, level)
        
        # Armazena os conjuntos de dados para possível uso posterior
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.X_calib_ = X_calib
        self.y_calib_ = y_calib
        self.X_test_ = X_test
        self.y_test_ = y_test
        
        return self
    
    def predict_interval(self, X):
        """
        Constrói intervalos de confiança para novas previsões.
        
        Parâmetros:
        - X: features para previsão
        
        Retorna:
        - Limites inferior e superior do intervalo de confiança
        """
        if self.base_model is None or self.quantile_model_lower is None or self.quantile_model_upper is None:
            raise ValueError("O modelo deve ser treinado antes de fazer previsões.")
        
        # Predição do modelo base
        y_pred = self.base_model.predict(X)
        
        # Predição dos limites dos resíduos
        lower_pred = self.quantile_model_lower.predict(X)
        upper_pred = self.quantile_model_upper.predict(X)
        
        # Constrói os intervalos de confiança
        lower_bound = y_pred + lower_pred - self.q_hat
        upper_bound = y_pred + upper_pred + self.q_hat
        
        return lower_bound, upper_bound
    
    def predict(self, X):
        """
        Realiza a previsão pontual usando o modelo base.
        
        Parâmetros:
        - X: features para previsão
        
        Retorna:
        - Previsões pontuais
        """
        if self.base_model is None:
            raise ValueError("O modelo deve ser treinado antes de fazer previsões.")
        
        return self.base_model.predict(X)
    
    def score(self, X, y):
        """
        Avalia a qualidade dos intervalos de predição.
        
        Parâmetros:
        - X: features de teste
        - y: targets de teste
        
        Retorna:
        - coverage: proporção de valores verdadeiros contidos nos intervalos
        - width: largura média dos intervalos
        """
        lower_bound, upper_bound = self.predict_interval(X)
        
        # Calcula a cobertura (porcentagem de valores verdadeiros dentro dos intervalos)
        coverage = np.mean((y >= lower_bound) & (y <= upper_bound))
        
        # Calcula a largura média dos intervalos
        width = np.mean(upper_bound - lower_bound)
        
        return {"coverage": coverage, "width": width}
    
    def get_split_sizes(self):
        """
        Retorna as proporções dos conjuntos de dados utilizados.
        
        Retorna:
        - Dicionário com os tamanhos proporcionais de cada conjunto
        """
        return {
            "training_set": self.train_size,
            "calibration_set": self.calib_size,
            "test_set": self.test_size_final
        }
    
    def plot_marginal_bandwidth_with_counts(self, X, feature_name, threshold=1.1, bins=10, figsize=(12, 8)):
        """
        Plota a largura marginal dos intervalos de previsão e a contagem de registros por segmento.
        
        Parâmetros:
        - X: DataFrame contendo as features.
        - feature_name: Nome da feature a ser analisada.
        - threshold: Limiar para identificar intervalos largos.
        - bins: Número de bins para discretizar a feature.
        - figsize: Tamanho da figura do gráfico.
        """
        # Calcular os intervalos de confiança
        lower_bound, upper_bound = self.predict_interval(X)
        interval_widths = upper_bound - lower_bound
        
        # Discretizar a feature em bins
        feature_values = X[feature_name]
        bins_values = pd.cut(feature_values, bins=bins, include_lowest=True)

        # Calcular a largura média dos intervalos e a contagem de registros para cada bin
        avg_widths = []
        counts_below_threshold = []
        counts_above_threshold = []
        bin_centers = []
        for bin in bins_values.cat.categories:
            mask = bins_values == bin
            avg_width = interval_widths[mask].mean()
            avg_widths.append(avg_width)
            counts_below_threshold.append(np.sum(interval_widths[mask] < threshold*interval_widths.mean()))
            counts_above_threshold.append(np.sum(interval_widths[mask] >= threshold*interval_widths.mean()))
            bin_centers.append(bin.mid)
            
        # Criar a figura com dois subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Gráfico 1: Largura média dos intervalos
        ax1.plot(bin_centers, avg_widths, marker='o', linestyle='-', color='blue', label='Largura Média dos Intervalos')
        ax1.axhline(y=threshold*interval_widths.mean(), color='red', linestyle='--', label='Limiar (Threshold)')
        ax1.set_ylabel('Largura Média dos Intervalos de Previsão')
        ax1.set_title(f'Marginal Bandwidth e Contagem de Registros para a Feature: {feature_name}')
        ax1.legend()
        ax1.grid(True)

        # Gráfico 2: Contagem de registros por segmento
        ax2.bar(bin_centers, counts_below_threshold, width=(bin_centers[1] - bin_centers[0]) * 0.4, label='Abaixo do Limiar', color='green', alpha=0.6)
        ax2.bar(bin_centers, counts_above_threshold, width=(bin_centers[1] - bin_centers[0]) * 0.4, label='Acima do Limiar', color='orange', alpha=0.6, bottom=counts_below_threshold)
        ax2.set_xlabel(feature_name)
        ax2.set_ylabel('Contagem de Registros')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()




#------------------------------------------------------------
# Carrega e prepara os dados
#------------------------------------------------------------
print("Carregando e preparando os dados...")

# Carrega os dados do California Housing
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Normaliza os features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divide em treino e teste
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=random_state
)

#------------------------------------------------------------
# Define os modelos base a serem comparados
#------------------------------------------------------------
print("Configurando modelos base...")

# Configuração dos modelos base para comparação
base_models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(kernel='rbf', gamma='scale', C=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state)
}

# Configurações do CRQR
alpha = 0.1  # 90% de intervalo de confiança
test_size = 0.6  # 40% treino, 60% para calibração+teste
calib_ratio = 1/3  # 1/3 do conjunto de teste para calibração (20% do total)

#------------------------------------------------------------
# Treina os modelos CRQR com diferentes modelos base
#------------------------------------------------------------
print("Treinando modelos CRQR com diferentes modelos base...")

# Armazena os resultados
results = []
interval_widths = {}

# Treina o CRQR para cada modelo base
for model_name, base_model in base_models.items():
    print(f"Treinando CRQR com {model_name}...")
    
    # Cria e treina o modelo CRQR
    crqr = CRQR(
        base_model=base_model,
        alpha=alpha,
        test_size=test_size,
        calib_ratio=calib_ratio,
        random_state=random_state
    )
    crqr.fit(X_train, y_train)
    
    # Calcula as predições e intervalos
    y_pred = crqr.predict(X_test)
    lower_bound, upper_bound = crqr.predict_interval(X_test)
    
    # Calcula as métricas
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    widths = upper_bound - lower_bound
    coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
    
    # Armazena os resultados
    results.append({
        "model": model_name,
        "mse": mse,
        "mae": mae,
        "mean_width": np.mean(widths),
        "median_width": np.median(widths),
        "min_width": np.min(widths),
        "max_width": np.max(widths),
        "coverage": coverage
    })
    
    # Armazena os comprimentos dos intervalos para boxplot
    interval_widths[model_name] = widths

#------------------------------------------------------------
# Cria um DataFrame para os resultados
#------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\nResultados:")
print(results_df[["model", "mse", "mae", "mean_width", "coverage"]].round(4))

#------------------------------------------------------------
# Cria um boxplot dos comprimentos dos intervalos
#------------------------------------------------------------
print("Criando visualizações...")

# Configura o estilo do gráfico
sns.set(style="whitegrid")
plt.figure(figsize=(14, 10))

# Cria um DataFrame para o boxplot
boxplot_data = pd.DataFrame({model: widths for model, widths in interval_widths.items()})

# Adiciona uma tabela com estatísticas resumidas
stat_info = []
for model in results_df['model']:
    row = results_df[results_df['model'] == model].iloc[0]
    stat_info.append([
        model, 
        f"{row['mean_width']:.4f}", 
        f"{row['coverage']:.4f}", 
        f"{row['mse']:.4f}"
    ])

# Plota uma visualização detalhada (boxplot + violino + pontos)
plt.subplot(2, 1, 2)
sns.violinplot(data=boxplot_data, orient='v', palette='Set2', inner=None, alpha=0.4)
sns.boxplot(data=boxplot_data, orient='v', palette='Set2', width=0.3, saturation=1)
sns.stripplot(data=boxplot_data, orient='v', palette='dark:blue', size=2, alpha=0.2, jitter=True)
plt.title('Distribuição Detalhada do Comprimento dos Intervalos', fontsize=14)
plt.xlabel('Comprimento do Intervalo de Confiança', fontsize=12)
plt.ylabel('Modelo Base', fontsize=12)

plt.tight_layout()
plt.savefig('boxplot_intervalos_confianca.png', dpi=300, bbox_inches='tight')


# Cria uma tabela de resultados
plt.figure(figsize=(12, 6))
plt.axis('off')
table_data = [['Modelo', 'Largura Média', 'Cobertura', 'MSE']] + stat_info
table = plt.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', 
                 bbox=[0.0, 0.0, 1.0, 1.0])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)
plt.savefig('tabela_resultados.png', dpi=300, bbox_inches='tight')

print("\nTabela de resultados salva como 'tabela_resultados.png'")

# Ajusta os dados para melhor visualização
boxplot_data_melted = pd.melt(boxplot_data, var_name='Modelo', value_name='Largura')

# Cria uma combinação de visualizações para facilitar a comparação dos modelos
plt.figure(figsize=(14, 6))
sns.boxplot(x='Modelo', y='Largura', data=boxplot_data_melted, palette='Set2')
plt.title('Distribuição do Comprimento dos Intervalos', fontsize=14)
plt.xticks(rotation=45)
plt.xlabel('')
plt.ylabel('Comprimento do Intervalo', fontsize=12)


#------------------------------------------------------------
# Análise de distância entre dados confiáveis e não confiáveis
#------------------------------------------------------------
print("\nRealizando análise de distância entre dados confiáveis e não confiáveis...")

# Define o modelo para analisar (usando o Gradient Boosting como exemplo)
model_to_analyze = "Gradient Boosting"
threshold_ratio = 1.1  # Conforme exemplo na imagem

# Obtenha os dados de intervalo para o modelo selecionado
lower_bound, upper_bound = None, None
for model_name, base_model in base_models.items():
    if model_name == model_to_analyze:
        # Cria e treina o modelo CRQR
        crqr = CRQR(
            base_model=base_model,
            alpha=alpha,
            test_size=test_size,
            calib_ratio=calib_ratio,
            random_state=random_state
        )
        crqr.fit(X_train, y_train)
        
        # Calcula os intervalos de predição
        lower_bound, upper_bound = crqr.predict_interval(X_test)
        break

# Calcula a largura dos intervalos
interval_widths_array = upper_bound - lower_bound

# Calcula o limiar para distinguir dados confiáveis de não confiáveis
# Usando o valor fixo 0.233 como exemplo da imagem
fixed_value = 0.233
threshold_value = threshold_ratio * fixed_value

print(f"Threshold para largura do intervalo: {threshold_value:.4f}")

# Classifica os pontos como confiáveis ou não confiáveis
reliable_mask = interval_widths_array < threshold_value
unreliable_mask = ~reliable_mask

reliable_points = X_test[reliable_mask]
unreliable_points = X_test[unreliable_mask]

print(f"Número de pontos confiáveis: {reliable_points.shape[0]}")
print(f"Número de pontos não confiáveis: {unreliable_points.shape[0]}")

# Se não houver pontos em alguma categoria, cria alguns exemplos artificiais para demonstração
if reliable_points.shape[0] == 0 or unreliable_points.shape[0] == 0:
    print("Aviso: Uma das categorias não tem pontos. Criando dados sintéticos para demonstração.")
    # Divide os dados em 70% confiáveis e 30% não confiáveis artificialmente
    n_samples = X_test.shape[0]
    reliable_mask = np.zeros(n_samples, dtype=bool)
    reliable_mask[:int(n_samples * 0.7)] = True
    np.random.shuffle(reliable_mask)
    unreliable_mask = ~reliable_mask
    reliable_points = X_test[reliable_mask]
    unreliable_points = X_test[unreliable_mask]

# Calcula a distância distribucional entre os grupos confiáveis e não confiáveis
# Implementação do PSI (Population Stability Index)
def calculate_psi(feature_reliable, feature_unreliable, buckets=10, method='uniform'):
    """
    Calcula o Population Stability Index (PSI) para uma feature.
    
    PSI = Σ (% da distribuição confiável - % da distribuição não confiável) * ln(% da distribuição confiável / % da distribuição não confiável)
    """
    # Determina os bins
    if method == 'uniform':
        all_values = np.concatenate([feature_reliable, feature_unreliable])
        min_val, max_val = np.min(all_values), np.max(all_values)
        bins = np.linspace(min_val, max_val, buckets + 1)
    else:  # 'quantile'
        all_values = np.concatenate([feature_reliable, feature_unreliable])
        bins = np.percentile(all_values, np.linspace(0, 100, buckets + 1))
    
    # Calcular histogramas
    hist_reliable, _ = np.histogram(feature_reliable, bins=bins)
    hist_unreliable, _ = np.histogram(feature_unreliable, bins=bins)
    
    # Converter para proporções
    hist_reliable = hist_reliable / np.sum(hist_reliable)
    hist_unreliable = hist_unreliable / np.sum(hist_unreliable)
    
    # Substituir zeros para evitar divisão por zero e log(0)
    small_value = 0.0001
    hist_reliable = np.where(hist_reliable == 0, small_value, hist_reliable)
    hist_unreliable = np.where(hist_unreliable == 0, small_value, hist_unreliable)
    
    # Calcular PSI
    psi_values = (hist_reliable - hist_unreliable) * np.log(hist_reliable / hist_unreliable)
    psi = np.sum(psi_values)
    
    return psi

# Calcula PSI para cada feature
feature_names = housing.feature_names
psi_values = []

for i in range(X_test.shape[1]):
    feature_reliable = reliable_points[:, i]
    feature_unreliable = unreliable_points[:, i]
    psi = calculate_psi(feature_reliable, feature_unreliable, buckets=10, method='uniform')
    psi_values.append(psi)

# Cria um DataFrame para visualização
psi_df = pd.DataFrame({
    'Feature': feature_names,
    'PSI': psi_values
})

# Ordena por PSI (maior para menor)
psi_df = psi_df.sort_values('PSI', ascending=False)

# Visualiza o gráfico de barras do PSI
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Feature', y='PSI', data=psi_df, palette='Blues_d')
plt.title('Distribution Shift: Unreliable vs. Reliable Data (PSI)', fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('PSI', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reliability_distance_analysis.png', dpi=300, bbox_inches='tight')

print("\nAnálise de distância de confiabilidade salva como 'reliability_distance_analysis.png'")

# Visualize a distribuição das features com maior PSI
top_features = psi_df.head(3)['Feature'].values
print(f"\nTop features com maior shift distribucional: {', '.join(top_features)}")

plt.figure(figsize=(15, 5))
for i, feature in enumerate(top_features):
    feature_idx = feature_names.index(feature)
    
    plt.subplot(1, 3, i+1)
    plt.hist(reliable_points[:, feature_idx], bins=20, alpha=0.5, label='Reliable', density=True)
    plt.hist(unreliable_points[:, feature_idx], bins=20, alpha=0.5, label='Unreliable', density=True)
    plt.title(f'Distribuição: {feature}')
    plt.xlabel('Valor')
    plt.ylabel('Densidade')
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig('top_features_distribution.png', dpi=300, bbox_inches='tight')

print("\nDistribuição das top features salva como 'top_features_distribution.png'")

# Define o modelo para analisar (usando o Gradient Boosting como exemplo)
model_to_analyze = "Gradient Boosting"
threshold_ratio = 1.1  # Conforme exemplo na imagem

# Obtenha os dados de intervalo para o modelo selecionado
lower_bound, upper_bound = None, None
for model_name, base_model in base_models.items():
    if model_name == model_to_analyze:
        # Cria e treina o modelo CRQR
        crqr = CRQR(
            base_model=base_model,
            alpha=alpha,
            test_size=test_size,
            calib_ratio=calib_ratio,
            random_state=random_state
        )
        crqr.fit(X_train, y_train)
        
        # Calcula os intervalos de predição
        lower_bound, upper_bound = crqr.predict_interval(X_test)
        break

X_test1=pd.DataFrame(X_test)
X_test1.columns=feature_names

crqr.plot_marginal_bandwidth_with_counts(X_test1, feature_name="MedInc")

# Define o modelo para analisar (usando o Gradient Boosting como exemplo)
model_to_analyze = "Gradient Boosting"
threshold_ratio = 1.1  # Conforme exemplo na imagem

# Obtenha os dados de intervalo para o modelo selecionado
lower_bound, upper_bound = None, None
for model_name, base_model in base_models.items():
    if model_name == model_to_analyze:
        # Cria e treina o modelo CRQR
        crqr = CRQR(
            base_model=base_model,
            alpha=alpha,
            test_size=test_size,
            calib_ratio=calib_ratio,
            random_state=random_state
        )
        crqr.fit(X_train, y_train)
        
        # Calcula os intervalos de predição
        lower_bound, upper_bound = crqr.predict_interval(X_test)
        break

X_test1=pd.DataFrame(X_test)
X_test1.columns=feature_names

crqr.plot_marginal_bandwidth_with_counts(X_test1, feature_name="Population", bins=5)

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Removido
# import seaborn as sns # Removido
from sklearn.metrics import mean_squared_error, mean_absolute_error # Apenas métricas
from typing import List, Dict, Union, Callable, Optional, Tuple
from copy import deepcopy
import pprint # Para imprimir dicionários e DataFrames de forma mais legível

# Imports necessários para os modelos e dados
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR # Adicionado SVR aqui
from sklearn.preprocessing import StandardScaler
import sklearn # Para checar versão

# ==============================================================================
# Definição da Classe CRQR (SEM O MÉTODO DE PLOT)
# ==============================================================================
class CRQR:
    """
    Conformalized Residual Quantile Regression (CRQR)

    Uma abordagem model-agnostic para avaliar a confiabilidade de modelos de regressão
    dentro do framework de predição conformal.
    """

    def __init__(self, base_model=None, alpha=0.1, test_size=0.6, calib_ratio=1/3, random_state=None):
        """
        Inicializador da classe CRQR.

        Parâmetros:
        - base_model: modelo de regressão base (padrão: None, usa o modelo padrão)
        - alpha: nível de significância (padrão: 0.1 para intervalos de confiança de 90%)
        - test_size: proporção dos dados a serem usados para teste+calibração (padrão: 0.6 = 60%)
        - calib_ratio: proporção do conjunto test_size a ser usada para calibração (padrão: 1/3,
                         resultando em 20% do total para calibração e 40% para teste)
        - random_state: semente aleatória para reprodutibilidade
        """
        self.alpha = alpha
        self.test_size = test_size
        self.calib_ratio = calib_ratio
        self.random_state = random_state

        # Calcula as proporções efetivas
        self.train_size = 1 - test_size
        self.calib_size = test_size * calib_ratio
        self.test_size_final = test_size * (1 - calib_ratio)

        # Modelo base para regressão
        if base_model is None:
            self.base_model = None  # Será configurado durante o fit
        else:
            self.base_model = base_model

        # Modelos de regressão quantil
        self.quantile_model_lower = None
        self.quantile_model_upper = None

        # Valor de calibração
        self.q_hat = None

        # Armazena os dados divididos internamente
        self.X_train_ = None
        self.y_train_ = None
        self.X_calib_ = None
        self.y_calib_ = None
        self.X_test_internal_ = None # Renomeado para evitar conflito com X_test externo
        self.y_test_internal_ = None # Renomeado para evitar conflito com y_test externo

    def fit(self, X, y):
        """
        Treina o modelo base e os modelos de regressão quantil, e calibra.

        Parâmetros:
        - X: features (pode ser todo o conjunto antes da divisão externa)
        - y: target (pode ser todo o conjunto antes da divisão externa)
        """
        # Divisão interna dos dados em treino, calibração e teste interno (para avaliação se necessário)
        # Esta divisão é feita a partir dos dados fornecidos a fit (que já podem ser o X_train externo)
        X_train_internal, X_temp, y_train_internal, y_temp = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Divisão do conjunto temporário em calibração e teste (teste interno)
        relative_test_size = 1 - self.calib_ratio
        if relative_test_size > 0 and len(X_temp) > 1 : # Garante que há dados para teste
             X_calib, X_test_int, y_calib, y_test_int = train_test_split(
                 X_temp, y_temp,
                 test_size=relative_test_size,
                 random_state=self.random_state
             )
        else: # Se não há dados suficientes para teste interno, usa tudo para calibração
             X_calib, y_calib = X_temp, y_temp
             X_test_int, y_test_int = X_calib[:0], y_calib[:0] # Cria arrays vazios


        # Armazena os dados divididos
        self.X_train_ = X_train_internal
        self.y_train_ = y_train_internal
        self.X_calib_ = X_calib
        self.y_calib_ = y_calib
        self.X_test_internal_ = X_test_int
        self.y_test_internal_ = y_test_int


        # Se não foi fornecido um modelo base, cria um modelo padrão
        if self.base_model is None:
            self.base_model = HistGradientBoostingRegressor(random_state=self.random_state)

        # Treina o modelo base com os dados de treinamento internos
        self.base_model.fit(self.X_train_, self.y_train_)

        # Prediz os valores para os dados de treinamento internos
        y_pred_train = self.base_model.predict(self.X_train_)

        # Calcula os resíduos no treino interno
        residuals = self.y_train_ - y_pred_train

        # Configura os modelos de regressão quantil (tentativa/erro para compatibilidade)
        try:
            # Tenta usar HistGradientBoostingRegressor com loss='quantile' (Sklearn >= 1.1)
            if sklearn.__version__ >= '1.1':
                self.quantile_model_lower = HistGradientBoostingRegressor(
                    loss='quantile', quantile=self.alpha/2, max_depth=5, random_state=self.random_state)
                self.quantile_model_upper = HistGradientBoostingRegressor(
                    loss='quantile', quantile=1-self.alpha/2, max_depth=5, random_state=self.random_state)
                if not hasattr(self.quantile_model_lower, 'loss') or self.quantile_model_lower.loss != 'quantile':
                     raise AttributeError("HistGradientBoostingRegressor não suporta loss='quantile' nesta versão.")
            else:
                # Para versões mais antigas, força o uso de GradientBoostingRegressor
                raise ImportError("HistGradientBoostingRegressor com quantil requer sklearn >= 1.1")
        except (ImportError, AttributeError, Exception) as e:
            # Fallback para GradientBoostingRegressor
            print(f"Aviso: Usando GradientBoostingRegressor para quantis (causa: {e}).")
            self.quantile_model_lower = GradientBoostingRegressor(
                loss='quantile', alpha=self.alpha/2, max_depth=5, random_state=self.random_state)
            self.quantile_model_upper = GradientBoostingRegressor(
                loss='quantile', alpha=1-self.alpha/2, max_depth=5, random_state=self.random_state)

        # Treina os modelos de regressão quantil nos resíduos do treino interno
        self.quantile_model_lower.fit(self.X_train_, residuals)
        self.quantile_model_upper.fit(self.X_train_, residuals)

        # --- Calibração ---
        # Prediz no conjunto de calibração
        y_pred_calib = self.base_model.predict(self.X_calib_)
        residuals_calib = self.y_calib_ - y_pred_calib

        # Prediz os limites dos resíduos para o conjunto de calibração
        lower_pred_resid = self.quantile_model_lower.predict(self.X_calib_)
        upper_pred_resid = self.quantile_model_upper.predict(self.X_calib_)

        # Calcula os scores de não-conformidade (o quão fora dos limites quantílicos o resíduo real está)
        # scores = max(lower_pred_resid - residual_real, residual_real - upper_pred_resid)
        scores = np.maximum(lower_pred_resid - residuals_calib, residuals_calib - upper_pred_resid)

        # Calcula o quantil ajustado dos scores de não-conformidade
        n_calib = len(scores)
        if n_calib == 0:
             raise ValueError("Conjunto de calibração está vazio. Ajuste test_size e calib_ratio.")
        level = np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib
        # Garante que o nível esteja dentro dos limites válidos para np.quantile
        level = min(1.0, max(0.0, level))
        self.q_hat = np.quantile(scores, level, interpolation='higher' if hasattr(np.quantile, 'interpolation') else 'higher') # Usa 'higher' para garantir cobertura


        return self

    def predict_interval(self, X):
        """
        Constrói intervalos de confiança para novas previsões.

        Parâmetros:
        - X: features para previsão

        Retorna:
        - lower_bound: Limites inferiores do intervalo de confiança
        - upper_bound: Limites superiores do intervalo de confiança
        """
        if self.base_model is None or self.quantile_model_lower is None or self.quantile_model_upper is None or self.q_hat is None:
            raise ValueError("O modelo deve ser treinado e calibrado antes de fazer previsões de intervalo ('fit').")

        # Predição do modelo base (estimativa pontual)
        y_pred = self.base_model.predict(X)

        # Predição dos limites dos resíduos (usando os modelos quantílicos treinados nos resíduos)
        lower_pred_resid = self.quantile_model_lower.predict(X)
        upper_pred_resid = self.quantile_model_upper.predict(X)

        # Constrói os intervalos de confiança ajustados pela calibração (q_hat)
        # Limite = previsão_pontual + previsão_do_limite_do_resíduo +/- ajuste_de_calibração
        lower_bound = y_pred + lower_pred_resid - self.q_hat
        upper_bound = y_pred + upper_pred_resid + self.q_hat

        return lower_bound, upper_bound

    def predict(self, X):
        """
        Realiza a previsão pontual usando o modelo base.

        Parâmetros:
        - X: features para previsão

        Retorna:
        - Previsões pontuais
        """
        if self.base_model is None:
            raise ValueError("O modelo base deve ser treinado antes de fazer previsões ('fit').")

        return self.base_model.predict(X)

    def score(self, X, y):
        """
        Avalia a qualidade dos intervalos de predição em dados de teste.

        Parâmetros:
        - X: features de teste
        - y: targets de teste

        Retorna:
        - Dicionário com 'coverage' (cobertura empírica) e 'width' (largura média dos intervalos)
        """
        lower_bound, upper_bound = self.predict_interval(X)

        # Calcula a cobertura (proporção de valores verdadeiros que caem dentro dos intervalos)
        coverage = np.mean((y >= lower_bound) & (y <= upper_bound))

        # Calcula a largura média dos intervalos
        width = np.mean(upper_bound - lower_bound)

        return {"coverage": coverage, "width": width}

    def get_split_sizes(self):
        """
        Retorna as proporções dos conjuntos de dados utilizados na divisão interna.

        Retorna:
        - Dicionário com os tamanhos proporcionais de cada conjunto interno (treino, calibração, teste interno)
        """
        total_fit_samples = len(self.X_train_) + len(self.X_calib_) + len(self.X_test_internal_)
        if total_fit_samples == 0:
             return {
                 "training_set_internal": 0,
                 "calibration_set_internal": 0,
                 "test_set_internal": 0
             }
        return {
            "training_set_internal": len(self.X_train_) / total_fit_samples,
            "calibration_set_internal": len(self.X_calib_) / total_fit_samples,
            "test_set_internal": len(self.X_test_internal_) / total_fit_samples
        }

    # --- Método plot_marginal_bandwidth_with_counts REMOVIDO ---


# ==============================================================================
# Script Principal (SEM PLOTS)
# ==============================================================================

#------------------------------------------------------------
# Carrega e prepara os dados
#------------------------------------------------------------
print("Carregando e preparando os dados...")

# Carrega os dados do California Housing
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Normaliza os features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divide em treino (para CRQR.fit) e teste (para avaliação final)
random_state = 42
X_train_crqr, X_test_final, y_train_crqr, y_test_final = train_test_split(
    X_scaled, y, test_size=0.2, random_state=random_state # 20% para teste final externo
)
print(f"Dados divididos: {X_train_crqr.shape[0]} para treino/calibração do CRQR, {X_test_final.shape[0]} para teste final.")

#------------------------------------------------------------
# Define os modelos base a serem comparados
#------------------------------------------------------------
print("\nConfigurando modelos base...")

# Configuração dos modelos base para comparação
base_models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(kernel='rbf', gamma='scale', C=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state)
}

# Configurações do CRQR (aplicadas dentro do X_train_crqr)
alpha = 0.1  # 90% de intervalo de confiança (nível de significância)
test_size_internal = 0.6  # Dentro do X_train_crqr: 40% treino interno, 60% para calibração+teste_interno
calib_ratio_internal = 1/3  # Dentro do test_size_internal: 1/3 para calibração (20% do X_train_crqr), 2/3 para teste interno (40% do X_train_crqr)

#------------------------------------------------------------
# Treina os modelos CRQR com diferentes modelos base
#------------------------------------------------------------
print("\nTreinando modelos CRQR com diferentes modelos base...")

# Armazena os resultados
results = []
# interval_widths = {} # Removido, pois era usado para plot

# Treina o CRQR para cada modelo base
for model_name, base_model_instance in base_models.items():
    print(f"  Treinando CRQR com {model_name}...")

    # Cria e treina o modelo CRQR usando os dados de TREINO EXTERNO
    # A divisão interna em treino/calibração/teste_interno é feita dentro do .fit
    crqr = CRQR(
        base_model=base_model_instance,
        alpha=alpha,
        test_size=test_size_internal,
        calib_ratio=calib_ratio_internal,
        random_state=random_state
    )
    crqr.fit(X_train_crqr, y_train_crqr) # Usa o conjunto de treino externo aqui

    # Avalia no conjunto de TESTE FINAL EXTERNO
    y_pred_final = crqr.predict(X_test_final)
    # Calcula as métricas de score (cobertura e largura) no TESTE FINAL EXTERNO
    score_metrics = crqr.score(X_test_final, y_test_final)

    # Calcula métricas de erro pontual no TESTE FINAL EXTERNO
    mse_final = mean_squared_error(y_test_final, y_pred_final)
    mae_final = mean_absolute_error(y_test_final, y_pred_final)

    # Armazena os resultados da avaliação final
    results.append({
        "model": model_name,
        "mse_final": mse_final,
        "mae_final": mae_final,
        "mean_width_final": score_metrics["width"], # Largura média no teste final
        "coverage_final": score_metrics["coverage"] # Cobertura no teste final
    })

    # Exibe os tamanhos das divisões internas usadas pelo CRQR
    # print(f"    Divisões internas para {model_name}: {crqr.get_split_sizes()}") # Opcional

#------------------------------------------------------------
# Exibe os resultados numéricos
#------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n--- Resultados da Avaliação Final (no conjunto de teste externo) ---")
# Ajusta a ordem das colunas para melhor visualização
print(results_df[["model", "mse_final", "mae_final", "mean_width_final", "coverage_final"]].round(4))


# --- SEÇÃO DE PLOTAGEM 1 REMOVIDA (Boxplot/Violin/Strip) ---
# --- SEÇÃO DE PLOTAGEM 2 REMOVIDA (Tabela como imagem) ---
# --- SEÇÃO DE PLOTAGEM 3 REMOVIDA (Boxplot simples) ---


#------------------------------------------------------------
# Análise de distância entre dados confiáveis e não confiáveis (SEM PLOTS)
#------------------------------------------------------------
print("\n--- Realizando análise de distância entre dados confiáveis e não confiáveis (usando Gradient Boosting) ---")

# Define o modelo para analisar (usando o Gradient Boosting como exemplo)
model_to_analyze = "Gradient Boosting"
threshold_ratio = 1.1  # Exemplo: intervalos 10% maiores que um valor base são "não confiáveis"

# Recria e treina o modelo CRQR especificamente para esta análise
# (poderia reutilizar se armazenado, mas refazer garante consistência)
base_model_gb = base_models.get(model_to_analyze)
if base_model_gb is None:
    print(f"Erro: Modelo '{model_to_analyze}' não encontrado.")
else:
    print(f"  Re-treinando CRQR com {model_to_analyze} para análise de confiabilidade...")
    crqr_analyzer = CRQR(
        base_model=base_model_gb,
        alpha=alpha,
        test_size=test_size_internal,
        calib_ratio=calib_ratio_internal,
        random_state=random_state
    )
    crqr_analyzer.fit(X_train_crqr, y_train_crqr)

    # Calcula os intervalos de predição no CONJUNTO DE TESTE FINAL
    lower_bound_final, upper_bound_final = crqr_analyzer.predict_interval(X_test_final)

    # Calcula a largura dos intervalos no teste final
    interval_widths_final = upper_bound_final - lower_bound_final

    # Calcula o limiar para distinguir dados confiáveis de não confiáveis
    # Usando a LARGURA MÉDIA no teste final como valor base (alternativa ao valor fixo 0.233 do exemplo)
    base_width_value = np.mean(interval_widths_final)
    threshold_value = threshold_ratio * base_width_value

    print(f"  Largura média no teste final: {base_width_value:.4f}")
    print(f"  Threshold ({threshold_ratio}x Média) para largura do intervalo: {threshold_value:.4f}")

    # Classifica os pontos do TESTE FINAL como confiáveis ou não confiáveis
    reliable_mask_final = interval_widths_final < threshold_value
    unreliable_mask_final = ~reliable_mask_final

    reliable_points_final = X_test_final[reliable_mask_final]
    unreliable_points_final = X_test_final[unreliable_mask_final]

    print(f"  Número de pontos confiáveis no teste final: {reliable_points_final.shape[0]}")
    print(f"  Número de pontos não confiáveis no teste final: {unreliable_points_final.shape[0]}")

    # Verifica se há pontos em ambas as categorias para calcular PSI
    if reliable_points_final.shape[0] > 0 and unreliable_points_final.shape[0] > 0:

        # Implementação do PSI (Population Stability Index)
        def calculate_psi(feature_reliable, feature_unreliable, buckets=10, method='uniform'):
            """ Calcula o Population Stability Index (PSI) para uma feature. """
            # Determina os bins
            all_values = np.concatenate([feature_reliable, feature_unreliable])
            if len(np.unique(all_values)) < 2: return 0.0 # Evita erro se todos os valores forem iguais
            min_val, max_val = np.min(all_values), np.max(all_values)
            if min_val == max_val: return 0.0 # Evita erro se todos os valores forem iguais

            if method == 'uniform':
                bins = np.linspace(min_val, max_val, buckets + 1)
            else:  # 'quantile'
                bins = np.unique(np.percentile(all_values, np.linspace(0, 100, buckets + 1))) # Usa unique para evitar bins duplicados
                if len(bins) < 2: # Se houver poucos valores únicos, usa uniform
                     bins = np.linspace(min_val, max_val, min(buckets + 1, len(np.unique(all_values))+1 ))

            # Calcular histogramas
            hist_reliable, _ = np.histogram(feature_reliable, bins=bins)
            hist_unreliable, _ = np.histogram(feature_unreliable, bins=bins)

            # Lida com o caso de histogramas vazios
            if np.sum(hist_reliable) == 0 or np.sum(hist_unreliable) == 0:
                return np.inf # Retorna infinito se um dos grupos não tiver dados nos bins

            # Converter para proporções
            hist_reliable = hist_reliable / np.sum(hist_reliable)
            hist_unreliable = hist_unreliable / np.sum(hist_unreliable)

            # Substituir zeros para evitar divisão por zero e log(0)
            small_value = 1e-10 # Valor pequeno
            hist_reliable = np.where(hist_reliable == 0, small_value, hist_reliable)
            hist_unreliable = np.where(hist_unreliable == 0, small_value, hist_unreliable)

            # Calcular PSI
            psi_values = (hist_reliable - hist_unreliable) * np.log(hist_reliable / hist_unreliable)
            psi = np.sum(psi_values)

            return psi

        # Calcula PSI para cada feature
        feature_names_housing = housing.feature_names
        psi_values = []

        for i in range(X_test_final.shape[1]):
            feature_reliable = reliable_points_final[:, i]
            feature_unreliable = unreliable_points_final[:, i]
            psi = calculate_psi(feature_reliable, feature_unreliable, buckets=10, method='uniform')
            psi_values.append(psi)

        # Cria um DataFrame para visualização
        psi_df = pd.DataFrame({
            'Feature': feature_names_housing,
            'PSI': psi_values
        })

        # Ordena por PSI (maior para menor)
        psi_df = psi_df.sort_values('PSI', ascending=False)

        print("\n--- Análise de Distância Distribucional (PSI) entre Grupos Confiável/Não Confiável ---")
        pprint.pprint(psi_df.round(4)) # Imprime o DataFrame PSI

        top_features = psi_df.head(3)['Feature'].values
        print(f"\n  Top features com maior shift distribucional (PSI mais alto): {', '.join(top_features)}")

    else:
        print("\n  Não foi possível calcular PSI: uma das categorias (confiável ou não confiável) está vazia.")


# --- SEÇÃO DE PLOTAGEM 4 REMOVIDA (Barplot PSI) ---
# --- SEÇÃO DE PLOTAGEM 5 REMOVIDA (Histogramas Top Features) ---


print("\nExecução concluída.")

reliable_points_final

top_features

psi_values

