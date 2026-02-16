# FAQ - DeepBridge v2.0

Perguntas frequentes sobre a migração para DeepBridge v2.0 e problemas comuns.

---

## 📦 Instalação

### Como instalar o DeepBridge v2.0?

```bash
pip install deepbridge>=2.0.0
```

### Como instalar os módulos opcionais?

```bash
# Para destilação de modelos
pip install deepbridge-distillation

# Para geração de dados sintéticos
pip install deepbridge-synthetic

# Instalar tudo
pip install deepbridge[all]
```

### Como verificar a versão instalada?

```bash
python -c "import deepbridge; print(deepbridge.__version__)"
```

---

## 🔄 Migração da v1.x para v2.0

### Quais são as principais mudanças?

1. **Estrutura modular:** código separado em 3 pacotes
2. **Novos imports:** `deepbridge.core`, `deepbridge_distillation`, `deepbridge_synthetic`
3. **APIs simplificadas:** menos parâmetros, mais defaults inteligentes
4. **Melhor tipagem:** suporte completo a type hints
5. **Performance:** otimizações em processamento de dados

### Como migrar meus imports?

**Antes (v1.x):**
```python
from deepbridge import DistillationTrainer
from deepbridge import SyntheticDataGenerator
from deepbridge.utils import load_config
```

**Depois (v2.0):**
```python
# Core sempre disponível
from deepbridge.core import BridgeConfig

# Módulos opcionais
from deepbridge_distillation import DistillationTrainer
from deepbridge_synthetic import SyntheticDataGenerator
```

### Meu código v1.x ainda funciona?

Depende. As principais mudanças:

- ✅ **APIs core:** majoritariamente compatíveis
- ⚠️ **Distillation:** requer `deepbridge-distillation`
- ⚠️ **Synthetic:** requer `deepbridge-synthetic`
- ❌ **Imports antigos:** não funcionam, precisa atualizar

### Existe guia de migração?

Sim! Consulte:
- `refatoracao/GUIA_MIGRACAO_V2.md` - Guia completo
- `refatoracao/CHECKLIST_MIGRACAO.md` - Checklist passo a passo

---

## 🐛 Problemas Comuns

### ModuleNotFoundError: No module named 'deepbridge_distillation'

**Problema:**
```python
from deepbridge_distillation import DistillationTrainer
# ModuleNotFoundError: No module named 'deepbridge_distillation'
```

**Solução:**
```bash
pip install deepbridge-distillation
```

**Explicação:** A partir da v2.0, destilação é um módulo opcional separado.

---

### ModuleNotFoundError: No module named 'deepbridge_synthetic'

**Problema:**
```python
from deepbridge_synthetic import SyntheticDataGenerator
# ModuleNotFoundError: No module named 'deepbridge_synthetic'
```

**Solução:**
```bash
pip install deepbridge-synthetic
```

**Explicação:** A partir da v2.0, geração sintética é um módulo opcional separado.

---

### ImportError: cannot import name 'DistillationTrainer' from 'deepbridge'

**Problema:**
```python
from deepbridge import DistillationTrainer
# ImportError: cannot import name 'DistillationTrainer' from 'deepbridge'
```

**Solução:**
Atualize o import:
```python
from deepbridge_distillation import DistillationTrainer
```

**Explicação:** Na v2.0, os módulos opcionais têm seus próprios pacotes.

---

### ImportError: cannot import name 'SyntheticDataGenerator' from 'deepbridge'

**Problema:**
```python
from deepbridge import SyntheticDataGenerator
# ImportError: cannot import name 'SyntheticDataGenerator' from 'deepbridge'
```

**Solução:**
Atualize o import:
```python
from deepbridge_synthetic import SyntheticDataGenerator
```

---

### Dependências faltando após instalar deepbridge

**Problema:**
```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'transformers'
```

**Solução:**
```bash
# Instalar dependências completas
pip install deepbridge[all]

# Ou instalar dependências específicas
pip install torch transformers
```

**Explicação:** Algumas dependências pesadas são opcionais na v2.0 para reduzir o tamanho da instalação base.

---

### Código lento após migração

**Problema:** O código ficou mais lento após atualizar para v2.0.

**Diagnóstico:**
1. Verifique se está usando caching:
```python
from deepbridge.core import enable_cache
enable_cache()
```

2. Verifique configuração de batch size:
```python
# Ajuste batch_size conforme sua GPU
trainer = DistillationTrainer(batch_size=32)  # ou 16, 64, etc.
```

3. Use profile para identificar gargalos:
```bash
python -m cProfile -o profile.stats seu_script.py
```

---

### Erro ao carregar modelo pré-treinado

**Problema:**
```
ValueError: Model checkpoint not compatible with v2.0
```

**Solução:**
1. Re-treinar o modelo com v2.0
2. Ou usar script de conversão (se disponível):
```bash
python scripts/convert_checkpoint_v1_to_v2.py --input old_model.pt --output new_model.pt
```

---

### Warnings sobre deprecated features

**Problema:**
```
DeprecationWarning: 'old_parameter' is deprecated, use 'new_parameter' instead
```

**Solução:**
Atualize seu código conforme as mensagens de warning. Exemplo:
```python
# Antes
trainer = DistillationTrainer(old_parameter=True)

# Depois
trainer = DistillationTrainer(new_parameter=True)
```

---

## 🔧 Troubleshooting

### Como depurar problemas?

1. **Ative logs detalhados:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Verifique versões:**
```bash
pip list | grep deepbridge
```

3. **Reproduza em ambiente limpo:**
```bash
python -m venv test_env
source test_env/bin/activate
pip install deepbridge[all]
python seu_script.py
```

### Como reportar um bug?

1. Abra uma issue no GitHub
2. Use o template de bug report
3. Inclua:
   - Versão do DeepBridge
   - Versão do Python
   - Sistema operacional
   - Código para reproduzir o bug
   - Mensagem de erro completa

### Onde encontrar mais ajuda?

- **Documentação:** `refatoracao/`
- **Issues:** GitHub Issues
- **Guias:** `GUIA_MIGRACAO_V2.md`, `CHECKLIST_MIGRACAO.md`
- **Changelog:** `CHANGELOG.md`

---

## 📊 Performance

### Como otimizar o treinamento?

```python
from deepbridge_distillation import DistillationTrainer

trainer = DistillationTrainer(
    batch_size=32,  # Ajuste conforme GPU
    num_workers=4,  # Paralelização de dados
    pin_memory=True,  # Acelera transferência GPU
    mixed_precision=True,  # FP16 para GPUs modernas
)
```

### Como reduzir uso de memória?

```python
trainer = DistillationTrainer(
    batch_size=16,  # Reduzir batch size
    gradient_accumulation_steps=2,  # Simula batch maior
    max_sequence_length=128,  # Reduzir se possível
)
```

---

## 🔍 Exemplos

### Exemplo básico de destilação

```python
from deepbridge.core import BridgeConfig
from deepbridge_distillation import DistillationTrainer

config = BridgeConfig(
    teacher_model="bert-base-uncased",
    student_model="distilbert-base-uncased",
)

trainer = DistillationTrainer(config)
trainer.train(train_dataset)
```

### Exemplo de geração sintética

```python
from deepbridge_synthetic import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    model="gpt2",
    num_samples=1000,
)

synthetic_data = generator.generate(prompts=["exemplo 1", "exemplo 2"])
```

---

## 📝 Notas Adicionais

### Compatibilidade com Python

- ✅ Python 3.8+
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11
- ⚠️ Python 3.12 (algumas dependências podem ter issues)

### Compatibilidade com PyTorch

- ✅ PyTorch 1.10+
- ✅ PyTorch 1.13
- ✅ PyTorch 2.0+

---

**Última atualização:** 2025-02-16

Para mais informações, consulte a documentação completa em `refatoracao/`.
