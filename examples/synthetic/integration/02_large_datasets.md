# Processamento de Grandes Conjuntos de Dados com Dask

Este exemplo demonstra como usar o módulo `synthetic` com suporte a Dask para gerar dados sintéticos a partir de conjuntos de dados grandes, aproveitando processamento distribuído.

```python
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Importações do DeepBridge
from deepbridge.synthetic.synthesizer import Synthesize
from deepbridge.synthetic.methods.gaussian_copula import GaussianCopulaGenerator

# Criar um conjunto de dados grande para demonstração
def create_large_dataset(n_samples=100000, n_features=20):
    """Cria um conjunto de dados grande para demonstração."""
    print(f"Gerando conjunto de dados com {n_samples} amostras e {n_features} atributos...")
    
    # Gerar dados para classificação binária
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=2,
        random_state=42
    )
    
    # Adicionar algumas correlações fortes
    X[:, 0] = X[:, 1] + X[:, 2] + np.random.normal(0, 0.5, n_samples)
    X[:, 3] = X[:, 4] * 0.8 + np.random.normal(0, 0.5, n_samples)
    
    # Adicionar algumas variáveis categóricas
    cat1 = np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.5, 0.3, 0.2])
    cat2 = np.random.choice(['X', 'Y', 'Z', 'W'], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # Criar DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['cat1'] = cat1
    df['cat2'] = cat2
    
    return df

# Criar diretório temporário para Dask
temp_dir = os.path.join(os.getcwd(), 'dask_temp')
os.makedirs(temp_dir, exist_ok=True)

# Criar conjuntos de dados de diferentes tamanhos
small_df = create_large_dataset(n_samples=5000, n_features=10)
medium_df = create_large_dataset(n_samples=50000, n_features=15)
large_df = create_large_dataset(n_samples=200000, n_features=20)  # Ajustar com base na memória disponível

print(f"Conjunto pequeno: {small_df.shape}")
print(f"Conjunto médio: {medium_df.shape}")
print(f"Conjunto grande: {large_df.shape}")

# Configurações das experiências
configurations = [
    {
        'name': 'Pequeno sem Dask',
        'data': small_df,
        'use_dask': False,
        'n_samples': 5000
    },
    {
        'name': 'Pequeno com Dask',
        'data': small_df,
        'use_dask': True,
        'n_samples': 5000
    },
    {
        'name': 'Médio sem Dask',
        'data': medium_df,
        'use_dask': False,
        'n_samples': 20000
    },
    {
        'name': 'Médio com Dask',
        'data': medium_df,
        'use_dask': True,
        'n_samples': 20000
    },
    {
        'name': 'Grande com Dask',
        'data': large_df,
        'use_dask': True,
        'n_samples': 50000
    }
]

# Função para executar experimento
def run_experiment(config):
    """Executa um experimento de geração de dados sintéticos com configuração específica."""
    print(f"\n\n===== Experimento: {config['name']} =====")
    print(f"Conjunto de dados: {config['data'].shape}")
    print(f"Usando Dask: {config['use_dask']}")
    print(f"Gerando {config['n_samples']} amostras sintéticas")
    
    start_time = time.time()
    
    # Criar e executar o gerador
    try:
        synthetic = Synthesize(
            dataset=config['data'],
            method='gaussian_copula',  # Método que suporta Dask
            num_samples=config['n_samples'],
            random_state=42,
            verbose=True,
            return_quality_metrics=False,  # Desativar métricas para focar na geração
            print_metrics=False,
            
            # Parâmetros relacionados ao Dask
            use_dask=config['use_dask'],
            dask_temp_directory=temp_dir,
            dask_threads_per_worker=2,
            # Se None, será determinado automaticamente com base no sistema
            dask_n_workers=None,
            
            # Parâmetros de gerenciamento de memória
            memory_limit_percentage=70.0
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Geração concluída em {elapsed_time:.2f} segundos")
        print(f"Dados sintéticos gerados: {synthetic.data.shape}")
        
        # Amostrar alguns resultados
        print("Primeiras 3 linhas dos dados sintéticos:")
        print(synthetic.data.head(3))
        
        return {
            'name': config['name'],
            'success': True,
            'time': elapsed_time,
            'data_shape': synthetic.data.shape,
            'data_sample': synthetic.data.head(5)
        }
        
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Erro durante a geração: {str(e)}")
        
        return {
            'name': config['name'],
            'success': False,
            'time': elapsed_time,
            'error': str(e)
        }

# Executar os experimentos
results = []
for config in configurations:
    result = run_experiment(config)
    results.append(result)
    
    # Pequena pausa para permitir que os recursos sejam liberados
    time.sleep(3)

# Mostrar resultados comparativos
print("\n\n===== Resultados Comparativos =====")
print(f"{'Nome':<20} {'Sucesso':<10} {'Tempo (s)':<15} {'Tamanho':<15}")
print("-" * 60)

for result in results:
    success = "Sim" if result['success'] else "Não"
    time_str = f"{result['time']:.2f}"
    shape_str = str(result.get('data_shape', 'N/A'))
    
    print(f"{result['name']:<20} {success:<10} {time_str:<15} {shape_str:<15}")

# Visualizar tempos de execução
plt.figure(figsize=(10, 6))
names = [r['name'] for r in results if r['success']]
times = [r['time'] for r in results if r['success']]

bars = plt.bar(names, times)
plt.ylabel('Tempo de Execução (segundos)')
plt.title('Tempo de Geração de Dados Sintéticos')
plt.xticks(rotation=45, ha='right')

# Adicionar valores nos topos das barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.1f}s', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Exemplo de uso da classe GaussianCopulaGenerator diretamente com configurações avançadas para Dask
print("\n\n===== Demonstração Avançada com GaussianCopulaGenerator e Dask =====")

# Configuração avançada do Dask para maior controle
advanced_generator = GaussianCopulaGenerator(
    random_state=42,
    verbose=True,
    preserve_dtypes=True,
    preserve_constraints=True,
    
    # Configurações de Dask
    use_dask=True,
    dask_temp_directory=temp_dir,
    dask_n_workers=4,  # Fixar número de workers
    dask_threads_per_worker=2,
    
    # Configurações de amostragem
    fit_sample_size=10000,  # Limitar amostras para ajuste
    
    # Configurações de memória
    memory_limit_percentage=70.0
)

print("Ajustando modelo no conjunto de dados grande...")
start_time = time.time()

# Ajustar o modelo
advanced_generator.fit(
    data=large_df,
    target_column='target',
    categorical_columns=['cat1', 'cat2']
)

# Gerar com controle granular das configurações
synthetic_data = advanced_generator.generate(
    num_samples=30000,
    chunk_size=5000,  # Tamanho específico do chunk
    dynamic_chunk_sizing=True,  # Mas permitir ajuste dinâmico
    noise_level=0.05  # Adicionar ruído para maior diversidade
)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Geração avançada concluída em {elapsed_time:.2f} segundos")
print(f"Dados sintéticos gerados: {synthetic_data.shape}")

# Mostrar estatísticas dos dados originais vs. sintéticos para features numéricas
print("\n===== Comparação de Estatísticas =====")
numerical_features = [col for col in large_df.columns 
                     if col not in ['cat1', 'cat2', 'target'] 
                     and pd.api.types.is_numeric_dtype(large_df[col])]

# Selecionar algumas features para comparar
features_to_compare = numerical_features[:5]

for feature in features_to_compare:
    print(f"\nFeature: {feature}")
    print(f"{'Estatística':<15} {'Original':<15} {'Sintético':<15}")
    print("-" * 45)
    
    # Calcular estatísticas
    orig_mean = large_df[feature].mean()
    orig_std = large_df[feature].std()
    orig_min = large_df[feature].min()
    orig_max = large_df[feature].max()
    
    synth_mean = synthetic_data[feature].mean()
    synth_std = synthetic_data[feature].std()
    synth_min = synthetic_data[feature].min()
    synth_max = synthetic_data[feature].max()
    
    # Mostrar comparação
    print(f"{'Média':<15} {orig_mean:<15.4f} {synth_mean:<15.4f}")
    print(f"{'Desvio Padrão':<15} {orig_std:<15.4f} {synth_std:<15.4f}")
    print(f"{'Mínimo':<15} {orig_min:<15.4f} {synth_min:<15.4f}")
    print(f"{'Máximo':<15} {orig_max:<15.4f} {synth_max:<15.4f}")

# Demonstrar salvamento e carregamento de modelos para reutilização
model_path = os.path.join(temp_dir, 'large_copula_model.pkl')
advanced_generator.save_model(model_path)
print(f"\nModelo salvo em: {model_path}")

# Limpar o gerador original e carregar novamente para demonstrar reutilização
del advanced_generator
import gc
gc.collect()

print("\nCarregando modelo salvo e gerando novas amostras...")
loaded_generator = GaussianCopulaGenerator(
    random_state=43,  # Diferente semente para variedade
    use_dask=True,
    dask_temp_directory=temp_dir,
    verbose=True
)

loaded_generator.load_model(model_path)
new_synthetic_data = loaded_generator.generate(
    num_samples=10000,
    chunk_size=2000
)

print(f"Novas amostras geradas com modelo carregado: {new_synthetic_data.shape}")

# Limpar recursos
print("\nLimpando recursos...")
del loaded_generator
gc.collect()

# Limpar diretório temporário do Dask (comentado por segurança)
# import shutil
# shutil.rmtree(temp_dir)
# print(f"Diretório temporário do Dask removido: {temp_dir}")

print("\nDemonstração concluída com sucesso!")
```

## Como o Processamento Distribuído Funciona

O DeepBridge integra-se com o Dask para permitir o processamento distribuído de grandes conjuntos de dados. Isso funciona da seguinte forma:

### 1. Inicialização do Dask
- Cria um cluster local com múltiplos workers
- Cada worker processa uma parte dos dados
- O número de workers pode ser configurado ou determinado automaticamente

### 2. Processamento em Chunks
- Os dados são divididos em chunks para processamento em paralelo
- Cada chunk é processado por um worker do Dask
- Os resultados são reunidos ao final do processamento

### 3. Gerenciamento de Memória
- O parâmetro `memory_limit_percentage` controla o uso máximo de memória
- O tamanho dos chunks é ajustado dinamicamente para evitar estouro de memória
- A limpeza de memória é feita periodicamente

## Configurações Importantes para Grandes Conjuntos de Dados

### Parâmetros de Dask
- **use_dask**: Ativar/desativar processamento distribuído
- **dask_temp_directory**: Diretório para armazenamento temporário
- **dask_n_workers**: Número de workers (None = automático)
- **dask_threads_per_worker**: Threads por worker

### Parâmetros de Memória
- **memory_limit_percentage**: Limite de uso de memória (% do total)
- **chunk_size**: Tamanho dos chunks para processamento
- **dynamic_chunk_sizing**: Ajuste automático do tamanho dos chunks

### Parâmetros de Amostragem
- **fit_sample_size**: Limitar número de amostras para ajuste do modelo
- **num_samples**: Número de amostras sintéticas a gerar

## Quando Usar Dask

O processamento distribuído com Dask é mais benéfico nos seguintes cenários:

1. **Conjuntos de dados grandes** (> 100.000 amostras ou > 50 atributos)
2. **Sistemas com múltiplos núcleos** ou múltiplas máquinas
3. **Geração de grandes volumes** de dados sintéticos
4. **Métodos complexos** como Gaussian Copula que são computacionalmente intensivos

Para conjuntos de dados pequenos a médios, o processamento sem Dask pode ser mais eficiente devido à sobrecarga de inicialização do Dask.

## Pontos-Chave

- O Dask permite escalabilidade para conjuntos de dados muito grandes
- A classe `GaussianCopulaGenerator` oferece controle granular sobre o processo
- Modelos podem ser salvos e carregados para reutilização
- O processamento em chunks permite gerenciar eficientemente a memória
- Para datasets pequenos, a sobrecarga do Dask pode não valer a pena
- O ajuste automático de parâmetros facilita o uso mesmo com grandes datasets
- O desempenho melhora significativamente em sistemas com múltiplos núcleos