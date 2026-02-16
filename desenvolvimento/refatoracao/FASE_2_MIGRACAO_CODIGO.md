# FASE 2: Migração de Código para Novos Repositórios

**Status:** ⏳ PENDENTE
**Duração estimada:** 2-3 semanas
**Objetivo:** Migrar código de distillation e synthetic para os novos repositórios

---

## 📋 Visão Geral

Nesta fase vamos:
1. Clonar os novos repositórios localmente
2. Configurar estrutura básica (Poetry, CI/CD)
3. Migrar código do backup para os novos repos
4. Ajustar imports (deepbridge.distillation → deepbridge_distillation)
5. Configurar dependências corretamente

**Repositórios alvo:**
- ✅ https://github.com/DeepBridge-Validation/deepbridge-distillation.git
- ✅ https://github.com/DeepBridge-Validation/deepbridge-synthetic.git

---

## 📦 PASSO 1: Clonar Repos e Setup Inicial

### 1.1 Clonar repositórios

```bash
# Criar diretório toolkit (se não existir)
mkdir -p /home/guhaase/projetos/deepbridge_toolkit
cd /home/guhaase/projetos/deepbridge_toolkit

# Clonar repos
git clone https://github.com/DeepBridge-Validation/deepbridge-distillation.git
git clone https://github.com/DeepBridge-Validation/deepbridge-synthetic.git

# Verificar estrutura
ls -la
# Deve mostrar:
# DeepBridge/                 (repo atual - já existe em /home/guhaase/projetos/DeepBridge)
# deepbridge-distillation/    (novo)
# deepbridge-synthetic/       (novo)
```

**Nota:** O repo `DeepBridge` atual já está em `/home/guhaase/projetos/DeepBridge`.
Você pode movê-lo para dentro de `deepbridge_toolkit/` se preferir ter tudo no mesmo lugar:

```bash
# Opcional: mover DeepBridge para dentro do toolkit
mv /home/guhaase/projetos/DeepBridge /home/guhaase/projetos/deepbridge_toolkit/
```

### 1.2 Verificar backup existe

```bash
# Verificar que o backup da Fase 1 existe
if [ -d "/tmp/deepbridge-migration" ]; then
    echo "✅ Backup encontrado"
    ls -la /tmp/deepbridge-migration/
else
    echo "❌ ERRO: Backup não encontrado!"
    echo "Execute a Fase 1 primeiro"
    exit 1
fi
```

---

## 🔧 PASSO 2: Configurar deepbridge-distillation

### 2.1 Estrutura de diretórios

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation

# Criar estrutura
mkdir -p deepbridge_distillation
mkdir -p tests
mkdir -p examples
mkdir -p docs

# Verificar
tree -L 1
```

### 2.2 Migrar código

```bash
# Copiar código do backup
cp -r /tmp/deepbridge-migration/distillation/* deepbridge_distillation/

# Copiar testes
if [ -d "/tmp/deepbridge-migration/tests/test_distillation" ]; then
    cp -r /tmp/deepbridge-migration/tests/test_distillation/* tests/
fi

echo "✅ Código copiado para deepbridge_distillation/"
```

### 2.3 Ajustar imports

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation

# Substituir imports em todos os arquivos .py
find deepbridge_distillation -name "*.py" -type f -exec sed -i 's/from deepbridge\.distillation/from deepbridge_distillation/g' {} +
find deepbridge_distillation -name "*.py" -type f -exec sed -i 's/import deepbridge\.distillation/import deepbridge_distillation/g' {} +

# Substituir nos testes também
find tests -name "*.py" -type f -exec sed -i 's/from deepbridge\.distillation/from deepbridge_distillation/g' {} +
find tests -name "*.py" -type f -exec sed -i 's/import deepbridge\.distillation/import deepbridge_distillation/g' {} +

echo "✅ Imports ajustados"
```

### 2.4 Criar __init__.py

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation

cat > deepbridge_distillation/__init__.py << 'EOF'
"""
DeepBridge Distillation - Model Compression and Knowledge Distillation

This package provides automated knowledge distillation and model compression
tools for machine learning models.

Requires: deepbridge>=2.0.0
"""

__version__ = '2.0.0-alpha.1'
__author__ = 'Team DeepBridge'

from deepbridge_distillation.auto_distiller import AutoDistiller
from deepbridge_distillation.experiment_runner import ExperimentRunner
from deepbridge_distillation.hpmkd_wrapper import HPMKD

try:
    from deepbridge_distillation.techniques.knowledge_distillation import (
        KnowledgeDistillation,
    )
    from deepbridge_distillation.techniques.surrogate import SurrogateModel
    from deepbridge_distillation.techniques.ensemble import EnsembleDistillation
except ImportError as e:
    # Pode falhar se dependências não estiverem instaladas
    pass

__all__ = [
    'AutoDistiller',
    'ExperimentRunner',
    'HPMKD',
    'KnowledgeDistillation',
    'SurrogateModel',
    'EnsembleDistillation',
]
EOF

echo "✅ __init__.py criado"
```

### 2.5 Criar pyproject.toml

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation

cat > pyproject.toml << 'EOF'
[tool.poetry]
name = "deepbridge-distillation"
version = "2.0.0-alpha.1"
description = "Model Compression and Knowledge Distillation Toolkit - Extension for DeepBridge"
authors = ["Team DeepBridge <gustavo.haase@gmail.com>"]
readme = "README.md"
license = "MIT"
homepage = "https://github.com/DeepBridge-Validation/deepbridge-distillation"
repository = "https://github.com/DeepBridge-Validation/deepbridge-distillation"
documentation = "https://deepbridge.readthedocs.io"
keywords = ["machine-learning", "model-compression", "knowledge-distillation", "deep-learning"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
packages = [{include = "deepbridge_distillation"}]

[tool.poetry.dependencies]
python = ">=3.10,<3.13"
# IMPORTANTE: Depende do deepbridge core!
deepbridge = ">=2.0.0-alpha.1"
# Dependências específicas de distillation
torch = ">=2.0.0"
xgboost = ">=2.0.0"
optuna = ">=3.0.0"
numpy = ">=1.24.0"
pandas = ">=2.0.0"
scikit-learn = ">=1.3.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.4"
pytest-cov = "^6.0.0"
mypy = "^1.19.1"
ruff = "^0.15.0"
black = "^24.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
pythonpath = "."
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
line-length = 79
target-version = "py310"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
EOF

echo "✅ pyproject.toml criado"
```

### 2.6 Criar README.md

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation

cat > README.md << 'EOF'
# DeepBridge Distillation

Model Compression and Knowledge Distillation Toolkit - Extension for [DeepBridge](https://github.com/DeepBridge-Validation/DeepBridge)

## Installation

```bash
pip install deepbridge-distillation
```

This will automatically install `deepbridge>=2.0.0` as a dependency.

## Quick Start

```python
from deepbridge import DBDataset
from deepbridge_distillation import AutoDistiller

# Create dataset with teacher model predictions
dataset = DBDataset(
    data=df,
    target_column='target',
    features=features,
    prob_cols=['prob_0', 'prob_1']
)

# Run automated distillation
distiller = AutoDistiller(
    dataset=dataset,
    output_dir='results',
    n_trials=10
)
results = distiller.run(use_probabilities=True)
```

## Features

- **Automated Distillation**: AutoDistiller with hyperparameter optimization
- **Knowledge Distillation**: Transfer knowledge from teacher to student models
- **Surrogate Models**: Create efficient surrogate models
- **HPM Knowledge Distillation**: Hierarchical Prototype-based Method
- **Multi-framework Support**: Works with scikit-learn, XGBoost, PyTorch

## Documentation

Full documentation: https://deepbridge.readthedocs.io/en/latest/distillation/

## Related Projects

- [deepbridge](https://github.com/DeepBridge-Validation/deepbridge) - Model Validation Toolkit (core)
- [deepbridge-synthetic](https://github.com/DeepBridge-Validation/deepbridge-synthetic) - Synthetic Data Generation

## License

MIT License - see [LICENSE](LICENSE)
EOF

echo "✅ README.md criado"
```

### 2.7 Configurar CI/CD

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation

mkdir -p .github/workflows

cat > .github/workflows/tests.yml << 'EOF'
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install

    - name: Lint with ruff
      run: |
        poetry run ruff check deepbridge_distillation/

    - name: Test with pytest
      run: |
        poetry run pytest tests/ -v --cov=deepbridge_distillation --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: distillation
EOF

echo "✅ CI/CD configurado"
```

### 2.8 Commit e push

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation

# Adicionar tudo
git add -A

# Commit
git commit -m "feat: Initial migration from DeepBridge v1.x

Migrate distillation module from monolithic DeepBridge repo.

Changes:
- Migrate code to deepbridge_distillation/
- Update imports (deepbridge.distillation → deepbridge_distillation)
- Add pyproject.toml with deepbridge>=2.0.0 dependency
- Add README and CI/CD
- Migrate tests

Part of: DeepBridge v2.0 refactoring
Related: DeepBridge-Validation/DeepBridge#XX"

# Push
git push origin main

echo "✅ deepbridge-distillation pronto!"
```

---

## 🔧 PASSO 3: Configurar deepbridge-synthetic

### 3.1 Estrutura de diretórios

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-synthetic

# Criar estrutura
mkdir -p deepbridge_synthetic
mkdir -p tests
mkdir -p examples
mkdir -p docs
```

### 3.2 Migrar código

```bash
# Copiar código do backup
cp -r /tmp/deepbridge-migration/synthetic/* deepbridge_synthetic/

# Copiar testes
if [ -d "/tmp/deepbridge-migration/tests/test_synthetic" ]; then
    cp -r /tmp/deepbridge-migration/tests/test_synthetic/* tests/
fi

echo "✅ Código copiado para deepbridge_synthetic/"
```

### 3.3 Ajustar imports

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-synthetic

# Substituir imports
find deepbridge_synthetic -name "*.py" -type f -exec sed -i 's/from deepbridge\.synthetic/from deepbridge_synthetic/g' {} +
find deepbridge_synthetic -name "*.py" -type f -exec sed -i 's/import deepbridge\.synthetic/import deepbridge_synthetic/g' {} +

# Substituir nos testes
find tests -name "*.py" -type f -exec sed -i 's/from deepbridge\.synthetic/from deepbridge_synthetic/g' {} +
find tests -name "*.py" -type f -exec sed -i 's/import deepbridge\.synthetic/import deepbridge_synthetic/g' {} +

echo "✅ Imports ajustados"
```

### 3.4 Criar __init__.py

```bash
cat > deepbridge_synthetic/__init__.py << 'EOF'
"""
DeepBridge Synthetic - Privacy-Preserving Synthetic Data Generation

This package provides tools for generating high-quality synthetic data
while preserving statistical properties and privacy.

Note: This is a standalone library and does NOT require deepbridge core.
"""

__version__ = '2.0.0-alpha.1'
__author__ = 'Team DeepBridge'

from deepbridge_synthetic.synthesizer import Synthesize
from deepbridge_synthetic.base_generator import BaseGenerator
from deepbridge_synthetic.standard_generator import StandardGenerator

__all__ = [
    'Synthesize',
    'BaseGenerator',
    'StandardGenerator',
]
EOF

echo "✅ __init__.py criado"
```

### 3.5 Criar pyproject.toml

```bash
cat > pyproject.toml << 'EOF'
[tool.poetry]
name = "deepbridge-synthetic"
version = "2.0.0-alpha.1"
description = "Privacy-Preserving Synthetic Data Generation - Standalone library"
authors = ["Team DeepBridge <gustavo.haase@gmail.com>"]
readme = "README.md"
license = "MIT"
homepage = "https://github.com/DeepBridge-Validation/deepbridge-synthetic"
repository = "https://github.com/DeepBridge-Validation/deepbridge-synthetic"
keywords = ["synthetic-data", "privacy", "data-generation", "gaussian-copula"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
packages = [{include = "deepbridge_synthetic"}]

[tool.poetry.dependencies]
python = ">=3.10,<3.13"
# NOTA: NÃO depende de deepbridge!
numpy = ">=1.24.0"
pandas = ">=2.0.0"
scipy = ">=1.11.0"
dask = {extras = ["distributed"], version = ">=2023.5.0"}
statsmodels = ">=0.14.0"
matplotlib = ">=3.7.0"
seaborn = ">=0.12.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.4"
pytest-cov = "^6.0.0"
mypy = "^1.19.1"
ruff = "^0.15.0"
black = "^24.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
pythonpath = "."
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
line-length = 79
target-version = "py310"
EOF

echo "✅ pyproject.toml criado"
```

### 3.6 Criar README.md

```bash
cat > README.md << 'EOF'
# DeepBridge Synthetic

Privacy-Preserving Synthetic Data Generation

**Note:** This is a standalone library and can be used without installing DeepBridge.

## Installation

```bash
pip install deepbridge-synthetic
```

## Quick Start

```python
from deepbridge_synthetic import Synthesize

# Generate synthetic data
synthesizer = Synthesize(
    data=original_df,
    method='gaussian_copula'
)

synthetic_df = synthesizer.generate(n_samples=10000)
```

## Features

- **Gaussian Copula**: Statistical modeling for synthetic data
- **Privacy-Preserving**: Generate data while protecting privacy
- **Quality Metrics**: Evaluate synthetic data quality
- **Distributed Computing**: Uses Dask for large datasets
- **Multiple Methods**: Various generation algorithms

## Documentation

Full documentation: https://deepbridge.readthedocs.io/en/latest/synthetic/

## Related Projects

- [deepbridge](https://github.com/DeepBridge-Validation/deepbridge) - Model Validation Toolkit
- [deepbridge-distillation](https://github.com/DeepBridge-Validation/deepbridge-distillation) - Model Distillation

## License

MIT License - see [LICENSE](LICENSE)
EOF

echo "✅ README.md criado"
```

### 3.7 Configurar CI/CD

```bash
mkdir -p .github/workflows

cat > .github/workflows/tests.yml << 'EOF'
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install

    - name: Lint with ruff
      run: |
        poetry run ruff check deepbridge_synthetic/

    - name: Test with pytest
      run: |
        poetry run pytest tests/ -v --cov=deepbridge_synthetic --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: synthetic
EOF

echo "✅ CI/CD configurado"
```

### 3.8 Commit e push

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-synthetic

git add -A

git commit -m "feat: Initial migration from DeepBridge v1.x

Migrate synthetic module from monolithic DeepBridge repo.

Changes:
- Migrate code to deepbridge_synthetic/
- Update imports (deepbridge.synthetic → deepbridge_synthetic)
- Add pyproject.toml (standalone, no deepbridge dependency)
- Add README and CI/CD
- Migrate tests

Note: This is a standalone library and does NOT require deepbridge.

Part of: DeepBridge v2.0 refactoring
Related: DeepBridge-Validation/DeepBridge#XX"

git push origin main

echo "✅ deepbridge-synthetic pronto!"
```

---

## 🧪 PASSO 4: Testar Instalação

### 4.1 Testar deepbridge-distillation

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate

# Instalar
pip install --upgrade pip
pip install -e .

# Testar import
python << 'EOF'
try:
    import deepbridge_distillation
    print(f"✅ deepbridge_distillation importado - versão: {deepbridge_distillation.__version__}")

    from deepbridge_distillation import AutoDistiller
    print("✅ AutoDistiller importado")

    # Verificar que deepbridge foi instalado como dependência
    import deepbridge
    print(f"✅ deepbridge core instalado como dependência - versão: {deepbridge.__version__}")

except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()
EOF

deactivate
```

### 4.2 Testar deepbridge-synthetic

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-synthetic

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate

# Instalar
pip install --upgrade pip
pip install -e .

# Testar import
python << 'EOF'
try:
    import deepbridge_synthetic
    print(f"✅ deepbridge_synthetic importado - versão: {deepbridge_synthetic.__version__}")

    from deepbridge_synthetic import Synthesize
    print("✅ Synthesize importado")

    # Verificar que NÃO instalou deepbridge (deve dar erro)
    try:
        import deepbridge
        print("⚠️  deepbridge instalado (não deveria)")
    except ImportError:
        print("✅ deepbridge NÃO instalado (correto - standalone)")

except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()
EOF

deactivate
```

---

## ✅ Checklist da Fase 2

### deepbridge-distillation
- [ ] Código migrado de /tmp/deepbridge-migration/distillation/
- [ ] Imports ajustados (deepbridge.distillation → deepbridge_distillation)
- [ ] pyproject.toml criado com dependência deepbridge>=2.0.0
- [ ] README.md criado
- [ ] CI/CD configurado (.github/workflows/tests.yml)
- [ ] Commit e push realizados
- [ ] Testes de import passando

### deepbridge-synthetic
- [ ] Código migrado de /tmp/deepbridge-migration/synthetic/
- [ ] Imports ajustados (deepbridge.synthetic → deepbridge_synthetic)
- [ ] pyproject.toml criado (SEM dependência de deepbridge)
- [ ] README.md criado
- [ ] CI/CD configurado
- [ ] Commit e push realizados
- [ ] Testes de import passando

---

## 🎯 Próximos Passos

Após completar a Fase 2:

**FASE_3_MIGRACAO_TESTES.md** - Migrar e ajustar testes completos

---

## 🆘 Troubleshooting

### Imports falhando

**Problema:** `ModuleNotFoundError: No module named 'deepbridge_distillation'`

**Solução:**
```bash
pip install -e . --force-reinstall
```

### Dependência circular

**Problema:** deepbridge-distillation tenta importar algo que não existe no deepbridge core

**Solução:** Verificar CORE_API_SPEC.md para ver o que está disponível no core.

### CI/CD falhando

Verificar:
1. pyproject.toml tem dependências corretas
2. Imports foram ajustados
3. Poetry está instalado no workflow

---

**Status Final da Fase 2:** ⬜ NÃO INICIADA | 🚧 EM ANDAMENTO | ✅ CONCLUÍDA

**Última atualização:** 2026-02-16
