# Geração de Relatórios

O DeepBridge oferece recursos poderosos para gerar relatórios interativos e estáticos que visualizam os resultados dos experimentos. Esta documentação mostra como usar o sistema de relatórios para diferentes casos de uso.

## Relatório Básico de Experimento

O método mais simples de gerar relatórios é através da classe `Experiment`, que integra todos os componentes necessários:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# Preparar dados
X = np.random.rand(200, 4)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
df['target'] = y

# Criar modelo e dataset
model = RandomForestClassifier(random_state=42)
model.fit(df.drop('target', axis=1), df['target'])

dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

# Criar experimento
experiment = Experiment(
    name="simple_report_demo",
    dataset=dataset,
    test_size=0.3,
    random_state=42
)

# Executar testes e gerar relatório
experiment.run_robustness_tests(config_name='quick')
experiment.generate_report(
    output_dir="./reports/robustness",
    report_type="robustness"
)

print("Relatório gerado com sucesso em ./reports/robustness")
```

## Gerando Múltiplos Tipos de Relatórios

Podemos gerar relatórios para diferentes testes em um único experimento:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# Preparar dados
X = np.random.rand(300, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
df['target'] = y

# Criar modelo e dataset
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(df.drop('target', axis=1), df['target'])

dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

# Criar experimento
experiment = Experiment(
    name="multi_report_demo",
    dataset=dataset,
    test_size=0.3,
    random_state=42
)

# Executar diferentes tipos de testes
experiment.run_robustness_tests(config_name='quick')
experiment.run_uncertainty_tests(config_name='quick')
experiment.run_resilience_tests(config_name='quick')

# Gerar relatórios para cada tipo de teste
base_dir = "./reports/multi_test"

# Relatório de robustez
experiment.generate_report(
    output_dir=f"{base_dir}/robustness",
    report_type="robustness"
)

# Relatório de incerteza
experiment.generate_report(
    output_dir=f"{base_dir}/uncertainty",
    report_type="uncertainty"
)

# Relatório de resiliência
experiment.generate_report(
    output_dir=f"{base_dir}/resilience",
    report_type="resilience"
)

print("Múltiplos relatórios gerados com sucesso em ./reports/multi_test")
```

## Relatório de Comparação de Modelos

Podemos gerar relatórios que comparam diferentes modelos:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# Preparar dados
X = np.random.rand(300, 4)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
df['target'] = y

# Criar e treinar modelos
X_train = df.drop('target', axis=1)
y_train = df['target']

rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
lr_model = LogisticRegression(random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Criar dataset com modelo principal
dataset = DBDataset(
    data=df,
    target_column='target',
    model=rf_model
)

# Configurar modelos alternativos
alternative_models = {
    'gradient_boost': gb_model,
    'logistic_regression': lr_model
}

# Criar experimento com modelos alternativos
experiment = Experiment(
    name="model_comparison",
    dataset=dataset,
    alternative_models=alternative_models,
    test_size=0.3,
    random_state=42
)

# Executar testes de comparação
experiment.compare_models_robustness(config_name='quick')

# Gerar relatório de comparação
experiment.generate_report(
    output_dir="./reports/model_comparison",
    report_type="robustness",
    include_comparisons=True
)

print("Relatório de comparação gerado com sucesso em ./reports/model_comparison")
```

## Usando o ReportManager Diretamente

Para casos mais avançados, podemos usar o `ReportManager` diretamente:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.report.report_manager import ReportManager

# Configurar experimento e executar testes como antes
X = np.random.rand(200, 3)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
df['target'] = y

model = RandomForestClassifier(random_state=42)
model.fit(df.drop('target', axis=1), df['target'])

dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

experiment = Experiment(
    name="direct_report_manager",
    dataset=dataset,
    test_size=0.3,
    random_state=42
)

# Executar testes e obter resultados
robustness_results = experiment.run_robustness_tests(config_name='quick')

# Usar o ReportManager diretamente
report_manager = ReportManager()

# Gerar relatório interativo
interactive_report_path = report_manager.generate_report(
    report_type="robustness",
    results=robustness_results,
    output_file="./reports/custom/interactive_report.html",
    model_name="RandomForest",
    format_type="interactive"  # Formato interativo
)

# Gerar relatório estático (para alguns tipos de relatórios)
static_report_path = report_manager.generate_report(
    report_type="resilience",
    results=experiment.run_resilience_tests(config_name='quick'),
    output_file="./reports/custom/static_report.html",
    model_name="RandomForest",
    format_type="static"  # Formato estático
)

print(f"Relatórios gerados em:\n- {interactive_report_path}\n- {static_report_path}")
```

## Customizando a Aparência dos Relatórios

Podemos customizar a aparência dos relatórios injetando CSS personalizado:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.report.report_manager import ReportManager
from deepbridge.core.experiment.report.asset_manager import AssetManager

# Configurar experimento e executar testes
X = np.random.rand(200, 3)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
df['target'] = y

model = RandomForestClassifier(random_state=42)
model.fit(df.drop('target', axis=1), df['target'])

dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

experiment = Experiment(
    name="custom_style_report",
    dataset=dataset,
    test_size=0.3,
    random_state=42
)

# Executar testes
robustness_results = experiment.run_robustness_tests(config_name='quick')

# Criar um AssetManager customizado
asset_manager = AssetManager()

# Adicionar CSS personalizado
custom_css = """
/* CSS personalizado para relatórios */
:root {
    --primary-color: #2c3e50;
    --secondary-color: #e74c3c;
    --bg-color: #f9f9f9;
    --text-color: #333;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: var(--bg-color);
    color: var(--text-color);
    margin: 0;
    padding: 20px;
}

.header {
    background-color: var(--primary-color);
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 25px;
}

.chart-container {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 20px;
}

.metric-card {
    border-left: 5px solid var(--secondary-color);
    padding-left: 15px;
}

table th {
    background-color: var(--primary-color);
    color: white;
}
"""

# Criar um ReportManager com o AssetManager personalizado
report_manager = ReportManager(asset_manager=asset_manager)

# Adicionar o CSS à coleção de ativos para relatórios de robustez
asset_manager.add_custom_css("robustness", custom_css)

# Gerar relatório com estilos personalizados
custom_report_path = report_manager.generate_report(
    report_type="robustness",
    results=robustness_results,
    output_file="./reports/custom/styled_report.html",
    model_name="RandomForest Customizado"
)

print(f"Relatório com estilo personalizado gerado em: {custom_report_path}")
```

## Criando um Tipo de Relatório Personalizado

Para necessidades avançadas, podemos implementar um tipo de relatório totalmente personalizado:

```python
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.report.base import DataTransformer
from deepbridge.core.experiment.report.renderers.base_renderer import BaseRenderer
from deepbridge.core.experiment.report.report_manager import ReportManager
from deepbridge.core.experiment.report.template_manager import TemplateManager
from deepbridge.core.experiment.report.asset_manager import AssetManager

# 1. Criar um transformador de dados personalizado
class CustomDataTransformer(DataTransformer):
    def transform(self, results, model_name=None, timestamp=None):
        # Criar uma cópia profunda dos resultados
        report_data = self._deep_copy(results)
        
        # Adicionar metadados
        report_data['model_name'] = model_name or report_data.get('model_name', 'Modelo')
        report_data['timestamp'] = timestamp or report_data.get('timestamp', 
                                                              pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Adicionar métricas personalizadas
        report_data['custom_score'] = 0.85  # Exemplo de métrica
        report_data['feature_scores'] = {
            'feature1': 0.7,
            'feature2': 0.3,
            'feature3': 0.5
        }
        
        # Converter tipos numpy para serialização
        return self.convert_numpy_types(report_data)

# 2. Criar um renderizador personalizado
class CustomRenderer(BaseRenderer):
    def __init__(self, template_manager, asset_manager):
        super().__init__(template_manager, asset_manager)
        self.data_transformer = CustomDataTransformer()
    
    def render(self, results, file_path, model_name="Modelo", format_type="interactive"):
        # Preparar o caminho do template
        template_dir = os.path.join(self.template_manager.template_dir, "custom")
        template_path = os.path.join(template_dir, "index.html")
        
        # Obter conteúdo CSS e JS
        css_content = "body { font-family: Arial, sans-serif; }"  # CSS simples para exemplo
        js_content = "console.log('Relatório customizado carregado!');"  # JS simples para exemplo
        
        # Transformar dados
        report_data = self.data_transformer.transform(results, model_name)
        
        # Criar contexto para o template
        context = self._create_context(report_data, "custom", css_content, js_content)
        
        # Criar diretório para o relatório se não existir
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Criar um template simples inline para o exemplo
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{model_name} - Relatório Personalizado</title>
            <style>{css_content}</style>
        </head>
        <body>
            <div class="header">
                <h1>{model_name} - Análise Personalizada</h1>
                <p>Gerado em: {report_data['timestamp']}</p>
            </div>
            
            <div class="content">
                <h2>Pontuação Geral: {report_data['custom_score']:.2f}</h2>
                
                <h3>Importância das Características</h3>
                <ul>
                    {"".join([f"<li>{feature}: {score:.2f}</li>" for feature, score in report_data['feature_scores'].items()])}
                </ul>
            </div>
            
            <script>{js_content}</script>
        </body>
        </html>
        """
        
        # Escrever o HTML no arquivo
        with open(file_path, 'w') as f:
            f.write(html_content)
        
        return file_path

# Configurar experimento e executar testes
X = np.random.rand(200, 3)
y = (X[:, 0] > 0.5).astype(int)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
df['target'] = y

model = RandomForestClassifier(random_state=42)
model.fit(df.drop('target', axis=1), df['target'])

dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

experiment = Experiment(
    name="custom_report_type",
    dataset=dataset,
    test_size=0.3,
    random_state=42
)

# Executar algum teste para obter resultados
robustness_results = experiment.run_robustness_tests(config_name='quick')

# Criar um ReportManager
template_manager = TemplateManager()
asset_manager = AssetManager()
report_manager = ReportManager(template_manager=template_manager, asset_manager=asset_manager)

# Registrar o novo renderizador
custom_renderer = CustomRenderer(template_manager, asset_manager)
report_manager.renderers['custom'] = custom_renderer

# Gerar o relatório personalizado
custom_report_path = report_manager.generate_report(
    report_type="custom",  # Usar o tipo personalizado
    results=robustness_results,  # Podemos usar qualquer conjunto de resultados
    output_file="./reports/custom/fully_custom_report.html",
    model_name="RandomForest Customizado"
)

print(f"Relatório totalmente personalizado gerado em: {custom_report_path}")
```

## Relatório Combinado de Múltiplos Testes

Podemos criar um relatório abrangente que combina os resultados de vários testes:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# Preparar dados
X = np.random.rand(300, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
df['target'] = y

# Criar modelo e dataset
model = RandomForestClassifier(random_state=42)
model.fit(df.drop('target', axis=1), df['target'])

dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

# Criar experimento
experiment = Experiment(
    name="comprehensive_report",
    dataset=dataset,
    test_size=0.3,
    random_state=42
)

# Executar todos os tipos de testes
experiment.run_robustness_tests(config_name='quick')
experiment.run_resilience_tests(config_name='quick')
experiment.run_uncertainty_tests(config_name='quick')

# Opcional: executar comparações de modelos
alternative_models = {
    'rf_deep': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(df.drop('target', axis=1), df['target']),
    'rf_shallow': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42).fit(df.drop('target', axis=1), df['target'])
}
experiment.alternative_models = alternative_models
experiment.compare_models_robustness(config_name='quick')

# Gerar relatório abrangente
experiment.generate_comprehensive_report(
    output_dir="./reports/comprehensive",
    include_sections=["robustness", "resilience", "uncertainty"],
    include_comparisons=True
)

print("Relatório abrangente gerado com sucesso em ./reports/comprehensive")
```

## Exportando Relatórios para Outros Formatos

Além de HTML, podemos exportar relatórios para outros formatos, como PDF:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
import subprocess
import os

# Configurar experimento e executar testes (código similar ao anterior)
X = np.random.rand(200, 3)
y = (X[:, 0] > 0.5).astype(int)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
df['target'] = y

model = RandomForestClassifier(random_state=42)
model.fit(df.drop('target', axis=1), df['target'])

dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)

experiment = Experiment(
    name="pdf_export",
    dataset=dataset,
    test_size=0.3,
    random_state=42
)

# Executar testes
experiment.run_robustness_tests(config_name='quick')

# Gerar relatório HTML primeiro
html_path = experiment.generate_report(
    output_dir="./reports/pdf_export",
    report_type="robustness"
)

# Verificar se o relatório foi gerado
if os.path.exists(html_path):
    # Converter HTML para PDF usando wkhtmltopdf (deve estar instalado no sistema)
    pdf_path = html_path.replace('.html', '.pdf')
    
    try:
        # Usando wkhtmltopdf para conversão
        subprocess.run([
            'wkhtmltopdf',
            '--enable-javascript',
            '--javascript-delay', '2000',  # Atraso para JS carregar
            '--no-stop-slow-scripts',
            '--page-size', 'A4',
            html_path,
            pdf_path
        ], check=True)
        print(f"Relatório PDF gerado com sucesso em: {pdf_path}")
    except subprocess.CalledProcessError:
        print("Erro ao converter para PDF. Verifique se wkhtmltopdf está instalado.")
    except FileNotFoundError:
        print("wkhtmltopdf não está instalado. Instale-o para converter relatórios para PDF.")
else:
    print("Falha ao gerar o relatório HTML.")
```

O sistema de relatórios do DeepBridge é flexível e extensível, permitindo uma ampla gama de personalizações para diferentes necessidades de análise e apresentação dos resultados de experimentos de machine learning.