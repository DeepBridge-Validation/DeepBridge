import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Any
import pkg_resources
import jinja2
import json
import plotly

class RobustnessReportGenerator:
    """
    Generates HTML reports for model robustness analysis.
    
    This class takes the results from robustness tests and creates
    a comprehensive HTML report with visualizations, metrics,
    and recommendations.
    """
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize the report generator.
        
        Args:
            template_path: Optional path to a custom template file
        """
        self.template_path = template_path
        
        # Load the template
        if template_path and os.path.exists(template_path):
            with open(template_path, 'r') as f:
                self.template_string = f.read()
        else:
            # Use the template from the repository
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            default_template_path = os.path.join(base_path, 'deepbridge', 'utils', 'templates', 'robustness_report_template.html')
            
            try:
                with open(default_template_path, 'r') as f:
                    self.template_string = f.read()
            except (FileNotFoundError, IOError):
                # Fallback - embed a minimal template here
                self.template_string = """
                [template minimo aqui]
                """
        
        # Configurando o ambiente Jinja2 com mais segurança para JSON
        self.env = jinja2.Environment(autoescape=True)
        # Adicionando filtro personalizado para JSON
        self.env.filters['tojson'] = lambda x: jinja2.Markup(json.dumps(x))
        # Inicializando o template
        self.template = self.env.from_string(self.template_string)
    
    def _plotly_fig_to_json(self, fig) -> str:
        """
        Convert a Plotly figure to JSON for embedding into HTML.
        
        Args:
            fig: Plotly figure object.
        
        Returns:
            JSON string representing the figure.
        """
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def _parse_json_string(self, json_str: str) -> Dict:
        """
        Parse a JSON string safely and return a Python dictionary.
        
        Args:
            json_str: JSON string to parse
            
        Returns:
            Dictionary parsed from JSON
        """
        try:
            # Verificando se a string é válida
            if not json_str or json_str == '{}':
                print("JSON string vazia ou inválida")
                return {}
                
            # Tentando converter de string para dicionário
            parsed = json.loads(json_str)
            print(f"JSON analisado com sucesso. Chaves: {list(parsed.keys())}")
            return parsed
        except Exception as e:
            print(f"Erro ao analisar string JSON: {e}")
            return {}
    
    def _get_robustness_color(self, index):
        """Obtém cor baseada no valor do índice de robustez com verificação de tipo robusta."""
        try:
            # Converter para float para garantir comparações numéricas
            # Se index já for uma string formatada (com %), remover o % e converter
            if isinstance(index, str) and '%' in index:
                index_value = float(index.replace('%', '')) / 100
            else:
                index_value = float(index)
                
            # Agora compara valores numéricos
            if index_value >= 0.8:
                return "#28a745"  # Verde de sucesso
            elif index_value >= 0.6:
                return "#17a2b8"  # Azul de informação
            elif index_value >= 0.4:
                return "#ffc107"  # Amarelo de aviso
            else:
                return "#dc3545"  # Vermelho de perigo
        except (ValueError, TypeError):
            # Cor padrão se a conversão falhar
            return "#6c757d"  # Cinza
    
    def _format_value(self, value, precision=4):
        """Version that maintains numeric type for internal use."""
        # Return the original value for internal calculations
        return value

    def _format_display_value(self, value, precision=4):
        """Format value only for display purposes."""
        # Treat None values
        if value is None:
            return "N/A"
                
        # Handle string values
        if isinstance(value, str):
            return value
                
        # Format numeric values
        try:
            return f"{float(value):.{precision}f}"
        except (ValueError, TypeError):
            # Return string representation if conversion fails
            return str(value)
    
    def generate_report(
        self,
        robustness_results: Dict,
        robustness_indices: Dict[str, float],
        feature_importance_chart: str,  # JSON string of plotly figure for feature importance
        boxplot_chart: str,  # JSON string of plotly figure for boxplot
        model_comparison_chart: str,  # JSON string of plotly figure for model comparison
        perturbation_methods_chart: str,  # JSON string of plotly figure for perturbation methods
        feature_importance_results: Optional[Dict] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate an HTML report from robustness test results.
        
        Args:
            robustness_results: Results from RobustnessTest.evaluate_robustness
            robustness_indices: Robustness indices from RobustnessScore.calculate_robustness_index
            feature_importance_chart: JSON string of plotly figure for feature importance
            boxplot_chart: JSON string of plotly figure for boxplot
            model_comparison_chart: JSON string of plotly figure for model comparison
            perturbation_methods_chart: JSON string of plotly figure for perturbation methods
            feature_importance_results: Optional feature importance results
            output_path: Optional path to save the report
            
        Returns:
            str: Generated HTML report
        """
        # Process data for the template
        model_names = list(robustness_results.keys())
        best_model = max(robustness_indices.items(), key=lambda x: x[1])[0]
        
        # Get perturbation levels
        if model_names:
            first_model = robustness_results[model_names[0]]
            perturbation_levels = first_model['perturb_sizes']
        else:
            perturbation_levels = []
        
        # Prepare model metrics
        model_metrics = []
        model_detailed_results = []
        
        for model_name in model_names:
            model_data = robustness_results[model_name]
            robustness_index = robustness_indices.get(model_name, 0)
            
            # Calculate baseline and perturbed performance
            baseline = model_data['mean_scores'][0] if model_data['mean_scores'] else 0
            
            # Get performance at approximately 50% perturbation
            perturb_idx = 0
            for i, size in enumerate(model_data['perturb_sizes']):
                if size >= 0.5:
                    perturb_idx = i
                    break
            
            perturbed = model_data['mean_scores'][perturb_idx] if perturb_idx < len(model_data['mean_scores']) else 0
            
            # Calculate performance drop (depends on whether higher is better)
            # Assume higher is better for most metrics except those containing 'mse', 'mae', etc.
            is_lower_better = any(term in model_name.lower() for term in ['mse', 'mae', 'rmse', 'error'])
            
            if is_lower_better:
                perf_drop = ((perturbed - baseline) / baseline) * 100 if baseline != 0 else 0
            else:
                perf_drop = ((baseline - perturbed) / baseline) * 100 if baseline != 0 else 0
            
            color = self._get_robustness_color(robustness_index)
            
            model_metrics.append({
                'name': model_name,
                'robustness_index': self._format_display_value(robustness_index * 100, 1),
                'robustness_index_raw': robustness_index * 100,  # Keep raw value for comparisons
                'baseline': self._format_display_value(baseline),
                'perturbed': self._format_display_value(perturbed),
                'drop': self._format_display_value(perf_drop, 1),
                'color': self._get_robustness_color(robustness_index)
            })
            
            # Detailed results for table
            scores = [self._format_value(score) for score in model_data['mean_scores']]
            model_detailed_results.append({
                'name': model_name,
                'robustness_index': self._format_value(robustness_index, 4),
                'scores': scores
            })
        
        # Extract top features
        top_features = []
        if feature_importance_results:
            features = feature_importance_results.get('sorted_features', [])
            impacts = feature_importance_results.get('sorted_impacts', [])
            
            # Get top 5 or less
            num_features = min(5, len(features))
            max_abs_impact = max([abs(impact) for impact in impacts[:num_features]], default=1.0)
            
            for i in range(num_features):
                percentage = abs(impacts[i]) / max_abs_impact * 100
                top_features.append({
                    'name': features[i],
                    'value': self._format_value(impacts[i], 3),
                    'percentage': percentage
                })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            robustness_results,
            robustness_indices,
            feature_importance_results
        )
        
        # Generate summary text
        summary_text = self._generate_summary(
            robustness_results,
            robustness_indices,
            feature_importance_results
        )
        
        # Analisar as strings JSON para objetos Python
        try:
            print("Analisando charts JSON...")
            model_comparison_obj = self._parse_json_string(model_comparison_chart)
            boxplot_obj = self._parse_json_string(boxplot_chart)
            feature_importance_obj = self._parse_json_string(feature_importance_chart)
            
            # Tratamento especial para perturbation_methods_chart que pode ser None
            perturbation_methods_obj = None
            if perturbation_methods_chart:
                perturbation_methods_obj = self._parse_json_string(perturbation_methods_chart)
                
            print("Análise de JSON concluída com sucesso")
        except Exception as e:
            print(f"Erro na análise de JSON: {e}")
            model_comparison_obj = {}
            boxplot_obj = {}
            feature_importance_obj = {}
            perturbation_methods_obj = None
        
        # Criar scripts inline para cada gráfico
        chart_scripts = self._create_chart_scripts(
            model_comparison_obj,
            boxplot_obj,
            feature_importance_obj,
            perturbation_methods_obj
        )
        
        # Prepare template data
        template_data = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'models_evaluated': ', '.join(model_names),
            'key_finding': f"The most robust model is {best_model} with robustness index {str(self._format_value(robustness_indices[best_model] * 100, 1))}%",
            'summary_text': summary_text,
            'model_metrics': model_metrics,
            'best_model': best_model,
            'model_comparison_chart': model_comparison_chart,
            'boxplot_chart': boxplot_chart,
            'feature_importance_chart': feature_importance_chart,
            'perturbation_methods_chart': perturbation_methods_chart,
            'top_features': top_features,
            'perturbation_levels': [str(self._format_value(level, 1)) for level in perturbation_levels],
            'model_detailed_results': model_detailed_results,
            'recommendations': recommendations,
            'chart_scripts': chart_scripts,  # Adicionado script inline para gráficos
            'a': 3
        }
        
        # Render HTML
        html_report = self.template.render(**template_data)
        
        # Injetar scripts diretamente no HTML para garantir que os gráficos funcionem
        html_report = self._inject_chart_scripts(html_report)
        
        # Save report if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
        
        return html_report
    
    def _create_chart_scripts(self, model_comparison_obj, boxplot_obj, feature_importance_obj, perturbation_methods_obj):
        """
        Cria scripts JavaScript para renderizar os gráficos diretamente.
        
        Args:
            model_comparison_obj: Objeto Python do gráfico de comparação de modelos
            boxplot_obj: Objeto Python do boxplot
            feature_importance_obj: Objeto Python do gráfico de importância de recursos
            perturbation_methods_obj: Objeto Python do gráfico de métodos de perturbação
            
        Returns:
            String contendo os scripts JavaScript
        """
        # Converter objetos de volta para strings JSON para inserção no JavaScript
        model_comparison_json = json.dumps(model_comparison_obj) if model_comparison_obj else '{}'
        boxplot_json = json.dumps(boxplot_obj) if boxplot_obj else '{}'
        feature_importance_json = json.dumps(feature_importance_obj) if feature_importance_obj else '{}'
        perturbation_methods_json = json.dumps(perturbation_methods_obj) if perturbation_methods_obj else '{}'
        
        # Criar script JavaScript
        script = f"""
        <script>
        document.addEventListener("DOMContentLoaded", function() {{
            console.log("DOM carregado - iniciando renderização de gráficos");
            
            // Função para renderizar gráficos com segurança
            function safelyRenderPlot(elementId, chartData, chartName) {{
                console.log(`Tentando renderizar ${{chartName || elementId}}`);
                
                if (!chartData || !chartData.data) {{
                    console.warn(`Dados inválidos para o gráfico ${{chartName || elementId}}`);
                    return;
                }}
                
                const element = document.getElementById(elementId);
                if (!element) {{
                    console.error(`Elemento DOM não encontrado: #${{elementId}}`);
                    return;
                }}
                
                try {{
                    console.log(`Renderizando ${{chartName || elementId}} com Plotly.newPlot`);
                    Plotly.newPlot(elementId, chartData.data, chartData.layout, {{responsive: true}});
                    console.log(`${{chartName || elementId}} renderizado com sucesso`);
                }} catch (error) {{
                    console.error(`Erro ao renderizar ${{chartName || elementId}}:`, error);
                }}
            }}

            // Definindo os objetos de dados dos gráficos diretamente
            try {{
                var model_comparison_chart = {model_comparison_json};
                safelyRenderPlot('model_comparison_chart', model_comparison_chart, "Model Comparison");
            }} catch (e) {{
                console.error("Erro ao processar model_comparison_chart:", e);
            }}
            
            try {{
                var boxplot_chart = {boxplot_json};
                safelyRenderPlot('boxplot_chart', boxplot_chart, "Boxplot");
            }} catch (e) {{
                console.error("Erro ao processar boxplot_chart:", e);
            }}
            
            try {{
                var feature_importance_chart = {feature_importance_json};
                safelyRenderPlot('feature_importance_chart', feature_importance_chart, "Feature Importance");
            }} catch (e) {{
                console.error("Erro ao processar feature_importance_chart:", e);
            }}
            
            try {{
                var perturbation_methods_chart = {perturbation_methods_json};
                if (perturbation_methods_chart && perturbation_methods_chart.data) {{
                    safelyRenderPlot('perturbation_methods_chart', perturbation_methods_chart, "Perturbation Methods");
                }}
            }} catch (e) {{
                console.error("Erro ao processar perturbation_methods_chart:", e);
            }}
            
            // Teste simples para verificar se o Plotly está funcionando
            try {{
                var testDiv = document.getElementById('test-chart');
                if (!testDiv) {{
                    testDiv = document.createElement('div');
                    testDiv.id = 'test-chart';
                    testDiv.style.width = '100%';
                    testDiv.style.height = '200px';
                    testDiv.style.border = '1px dashed #ccc';
                    testDiv.style.margin = '20px 0';
                    
                    var container = document.querySelector('.container');
                    if (container) {{
                        var footer = container.querySelector('footer');
                        if (footer) {{
                            container.insertBefore(testDiv, footer);
                        }} else {{
                            container.appendChild(testDiv);
                        }}
                    }}
                }}
                
                if (testDiv) {{
                    console.log("Criando gráfico de teste para verificar o Plotly");
                    Plotly.newPlot('test-chart', 
                        [{{x: [1, 2, 3], y: [1, 2, 3], type: 'scatter', name: 'Teste'}}], 
                        {{title: 'Gráfico de Teste - Se você vê isso, o Plotly está funcionando!'}}, 
                        {{responsive: true}}
                    );
                }}
            }} catch (e) {{
                console.error("Erro ao criar gráfico de teste:", e);
            }}
        }});
        </script>
        """
        
        return script
    
    def _inject_chart_scripts(self, html_report):
        """
        Injeta scripts diretamente no HTML para garantir que os gráficos funcionem.
        
        Args:
            html_report: HTML report string
            
        Returns:
            HTML report com scripts injetados
        """
        # Verificar se {{ chart_scripts }} foi substituído corretamente
        if "{{ chart_scripts }}" in html_report:
            # Se não foi substituído, forçar a substituição
            chart_scripts = self.template.render(chart_scripts=self._create_chart_scripts({}, {}, {}, {}))
            html_report = html_report.replace("{{ chart_scripts }}", chart_scripts)
        
        # Procurar por </body> para injetar o script antes dele
        if "</body>" in html_report and "function safelyRenderPlot" not in html_report:
            # Criar um script de contingência mínimo
            script = """
            <script>
            console.log("Script contingência injetado diretamente no HTML");
            
            document.addEventListener("DOMContentLoaded", function() {
                // Teste simples para verificar se o Plotly está funcionando
                try {
                    var testDiv = document.getElementById('test-chart');
                    if (!testDiv) {
                        testDiv = document.createElement('div');
                        testDiv.id = 'test-chart';
                        testDiv.style.width = '100%';
                        testDiv.style.height = '200px';
                        testDiv.style.border = '1px dashed #ccc';
                        testDiv.style.margin = '20px 0';
                        
                        var container = document.querySelector('.container');
                        if (container) {
                            container.appendChild(testDiv);
                        } else {
                            document.body.appendChild(testDiv);
                        }
                    }
                    
                    console.log("Criando gráfico de teste para verificar o Plotly");
                    Plotly.newPlot('test-chart', 
                        [{x: [1, 2, 3], y: [1, 2, 3], type: 'scatter', name: 'Teste'}], 
                        {title: 'Gráfico de Teste - Injetado como contingência'}, 
                        {responsive: true}
                    );
                } catch (e) {
                    console.error("Erro ao criar gráfico de teste:", e);
                }
            });
            </script>
            """
            
            html_report = html_report.replace("</body>", script + "\n</body>")
        
        return html_report
        
    def _generate_recommendations(
        self,
        robustness_results: Dict,
        robustness_indices: Dict[str, float],
        feature_importance_results: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            robustness_results: Results from robustness testing
            robustness_indices: Robustness indices
            feature_importance_results: Feature importance analysis
            
        Returns:
            List[Dict]: List of recommendation objects
        """
        recommendations = []
        
        # Find best and worst models
        if robustness_indices:
            best_model = max(robustness_indices.items(), key=lambda x: x[1])
            worst_model = min(robustness_indices.items(), key=lambda x: x[1])
            
            if best_model[1] > 0.7:
                recommendations.append({
                    'title': f"Deploy {best_model[0]} for Production",
                    'content': f"The {best_model[0]} model demonstrates strong robustness with an index of {best_model[1]:.2f}. "
                              f"This model maintains consistent performance under perturbation and is recommended for production environments."
                })
            
            if worst_model[1] < 0.5:
                recommendations.append({
                    'title': f"Improve {worst_model[0]} Robustness",
                    'content': f"The {worst_model[0]} model shows lower robustness with an index of {worst_model[1]:.2f}. "
                              f"Consider applying regularization techniques or using more diverse training data to improve robustness."
                })
        
        # Feature importance recommendations
        if feature_importance_results:
            # Get most impactful features
            features = feature_importance_results.get('sorted_features', [])
            impacts = feature_importance_results.get('sorted_impacts', [])
            
            if features and impacts and len(features) > 0:
                most_impactful = features[0]
                impact_value = impacts[0]
                
                recommendations.append({
                    'title': f"Focus on Feature Quality: {most_impactful}",
                    'content': f"The feature '{most_impactful}' has the highest impact on model robustness "
                              f"with a normalized impact of {impact_value:.3f}. Ensure data quality for this feature "
                              f"and consider collecting more varied examples to improve model stability."
                })
        
        # General recommendations
        recommendations.append({
            'title': "Implement Regular Robustness Testing",
            'content': "Set up a pipeline for continuous robustness assessment as part of your model monitoring workflow. "
                      "This will help catch degradation in model stability over time as data distributions may shift."
        })
        
        # Add recommendation about perturbation levels if applicable
        for model_name, model_results in robustness_results.items():
            # Check if there's a significant drop in performance at some perturbation level
            if len(model_results['mean_scores']) > 2:
                scores = model_results['mean_scores']
                sizes = model_results['perturb_sizes']
                
                # Calculate differences between consecutive scores
                diffs = [abs(scores[i] - scores[i-1]) for i in range(1, len(scores))]
                max_diff_idx = diffs.index(max(diffs))
                
                # If there's a significant drop (> 10% of the range)
                score_range = max(scores) - min(scores)
                if score_range > 0 and diffs[max_diff_idx] > 0.1 * score_range:
                    critical_size = sizes[max_diff_idx + 1]
                    recommendations.append({
                        'title': f"Critical Perturbation Threshold for {model_name}",
                        'content': f"The {model_name} model shows a significant performance drop at perturbation level {critical_size:.2f}. "
                                  f"This suggests the model may be sensitive to noise at this level. "
                                  f"Consider testing with real-world data containing similar noise levels to validate robustness."
                    })
                    break
        
        # Add recommendations about comparing models if multiple models were tested
        if len(robustness_indices) > 1:
            recommendations.append({
                'title': "Consider Model Ensemble",
                'content': "Based on the different robustness profiles of the tested models, consider implementing an ensemble "
                          "approach that combines models with complementary robustness characteristics. This can help "
                          "mitigate the weaknesses of individual models and create a more stable overall system."
            })
        
        return recommendations
    
    def _generate_summary(
        self,
        robustness_results: Dict,
        robustness_indices: Dict[str, float],
        feature_importance_results: Optional[Dict] = None
    ) -> str:
        """
        Generate a summary text based on analysis results.
        
        Args:
            robustness_results: Results from robustness testing
            robustness_indices: Robustness indices
            feature_importance_results: Feature importance analysis
            
        Returns:
            str: Summary text
        """
        summary_parts = []
        
        # Overall robustness assessment
        if robustness_indices:
            avg_robustness = sum(robustness_indices.values()) / len(robustness_indices)
            if avg_robustness > 0.8:
                summary_parts.append("The evaluated models demonstrate excellent robustness overall, maintaining consistent performance under perturbation.")
            elif avg_robustness > 0.6:
                summary_parts.append("The evaluated models show good robustness overall, with some performance degradation under higher perturbation levels.")
            elif avg_robustness > 0.4:
                summary_parts.append("The evaluated models exhibit moderate robustness, with noticeable performance degradation under perturbation.")
            else:
                summary_parts.append("The evaluated models show limited robustness, with significant performance drops even under mild perturbation.")
        
        # Model comparison if multiple models
        if len(robustness_indices) > 1:
            best_model = max(robustness_indices.items(), key=lambda x: x[1])
            worst_model = min(robustness_indices.items(), key=lambda x: x[1])
            
            diff = best_model[1] - worst_model[1]
            if diff > 0.3:
                summary_parts.append(f"There is a significant difference in robustness between the models, with {best_model[0]} outperforming {worst_model[0]} by a wide margin.")
            elif diff > 0.1:
                summary_parts.append(f"There are noticeable differences in robustness between models, with {best_model[0]} being the most stable under perturbation.")
            else:
                summary_parts.append(f"All models show similar robustness profiles, with {best_model[0]} having a slight edge in stability.")
        
        # Feature importance insights
        if feature_importance_results and feature_importance_results.get('sorted_features'):
            top_features = feature_importance_results['sorted_features'][:3]
            summary_parts.append(f"The analysis identified {', '.join(top_features)} as the features with the most significant impact on model robustness.")
        
        # Combine all parts into a coherent summary
        summary = " ".join(summary_parts)
        
        # Add recommendations preview
        summary += " Based on these findings, we recommend focusing on the most robust model for deployment while continuing to monitor and improve the stability of all models."
        
        return summary