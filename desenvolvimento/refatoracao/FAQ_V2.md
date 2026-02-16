# FAQ - DeepBridge v2.0 - Perguntas Frequentes

**Última atualização:** 2026-02-16

Este FAQ cobre problemas comuns de instalação, migração e uso do DeepBridge v2.0 após a reestruturação modular.

---

## 📦 Instalação

### P: Como instalar o DeepBridge v2.0?

**R:** Existem três pacotes separados agora:

```bash
# Pacote core (obrigatório)
pip install deepbridge

# Módulo de destilação (opcional)
pip install deepbridge-distillation

# Módulo de dados sintéticos (opcional)
pip install deepbridge-synthetic
```

**Instalar tudo de uma vez:**

```bash
pip install deepbridge deepbridge-distillation deepbridge-synthetic
```

---

### P: Posso instalar apenas o que preciso?

**R:** Sim! A arquitetura modular permite instalar apenas os pacotes necessários:

- **Apenas funcionalidades core:** `pip install deepbridge`
- **Core + Distillation:** `pip install deepbridge deepbridge-distillation`
- **Core + Synthetic:** `pip install deepbridge deepbridge-synthetic`
- **Tudo:** `pip install deepbridge deepbridge-distillation deepbridge-synthetic`

---

### P: Quais são as dependências de cada pacote?

**R:**

- **deepbridge:** numpy, pandas, scikit-learn, torch (core dependencies)
- **deepbridge-distillation:** deepbridge>=2.0.0, torch, torchvision
- **deepbridge-synthetic:** deepbridge>=2.0.0, faker, sdv (optional)

Consulte os arquivos `requirements.txt` de cada repositório para detalhes completos.

---

## 🔄 Migração de v1.x para v2.0

### P: Meu código v1.x parou de funcionar. O que mudou?

**R:** A principal mudança são os imports. Veja a tabela de migração:

| **v1.x (Antigo)** | **v2.0 (Novo)** |
|------------------|----------------|
| `from DeepBridge.distillation import KnowledgeDistiller` | `from deepbridge.distillation import KnowledgeDistiller` |
| `from DeepBridge.synthetic import SyntheticDataGenerator` | `from deepbridge.synthetic import SyntheticDataGenerator` |
| `from DeepBridge.utils import setup_logger` | `from deepbridge.utils import setup_logger` |
| `from DeepBridge import Bridge` | `from deepbridge import Bridge` |

**Principais mudanças:**
1. Nome do pacote: `DeepBridge` → `deepbridge` (lowercase)
2. Estrutura modular: funcionalidades separadas em pacotes independentes
3. Imports explícitos: submodules precisam ser importados explicitamente

---

### P: Recebi `ModuleNotFoundError: No module named 'DeepBridge'`

**R:** Você está usando imports da v1.x. Siga estes passos:

**1. Desinstale a versão antiga:**
```bash
pip uninstall DeepBridge
```

**2. Instale a v2.0:**
```bash
pip install deepbridge
# E módulos opcionais conforme necessário
pip install deepbridge-distillation deepbridge-synthetic
```

**3. Atualize seus imports:**
```python
# ❌ Antigo (v1.x)
from DeepBridge.distillation import KnowledgeDistiller

# ✅ Novo (v2.0)
from deepbridge.distillation import KnowledgeDistiller
```

---

### P: Recebi `ModuleNotFoundError: No module named 'deepbridge.distillation'`

**R:** Você instalou apenas o pacote core. Instale o módulo de destilação:

```bash
pip install deepbridge-distillation
```

**Verificar instalação:**
```bash
pip list | grep deepbridge
```

Você deve ver:
```
deepbridge               2.0.0
deepbridge-distillation  2.0.0
```

---

### P: Recebi `ModuleNotFoundError: No module named 'deepbridge.synthetic'`

**R:** Você instalou apenas o pacote core. Instale o módulo de dados sintéticos:

```bash
pip install deepbridge-synthetic
```

**Verificar instalação:**
```bash
pip list | grep deepbridge
```

Você deve ver:
```
deepbridge              2.0.0
deepbridge-synthetic    2.0.0
```

---

### P: Como migrar meu código automaticamente?

**R:** Use nosso script de migração automática:

```bash
# Baixar script (disponível no repo)
python scripts/migrate_imports.py <seu_arquivo.py>

# Ou para um diretório inteiro
python scripts/migrate_imports.py <seu_diretorio> --recursive
```

O script irá:
- Substituir `DeepBridge` → `deepbridge`
- Atualizar imports de submódulos
- Criar backup do arquivo original (.bak)

---

## 🏗️ Problemas Comuns

### P: `ImportError: cannot import name 'X' from 'deepbridge'`

**R:** Verifique de qual módulo a funcionalidade faz parte:

```python
# ❌ Errado
from deepbridge import KnowledgeDistiller  # Não está no core

# ✅ Correto
from deepbridge.distillation import KnowledgeDistiller
```

**Mapeamento de módulos:**
- `deepbridge.*` → Funcionalidades core (Bridge, utils, base)
- `deepbridge.distillation.*` → Conhecimento/destilação (requer deepbridge-distillation)
- `deepbridge.synthetic.*` → Dados sintéticos (requer deepbridge-synthetic)

---

### P: Recebi `AttributeError: module 'deepbridge' has no attribute 'X'`

**R:** Você precisa importar explicitamente de submódulos:

```python
# ❌ Errado
import deepbridge
model = deepbridge.KnowledgeDistiller()

# ✅ Correto
from deepbridge.distillation import KnowledgeDistiller
model = KnowledgeDistiller()
```

**Nota:** Na v2.0, imports devem ser explícitos para reduzir overhead.

---

### P: Meu ambiente virtual tem versões conflitantes

**R:** Recrie o ambiente virtual:

```bash
# Desativar e remover ambiente antigo
deactivate
rm -rf venv/

# Criar novo ambiente
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar v2.0
pip install --upgrade pip
pip install deepbridge deepbridge-distillation deepbridge-synthetic

# Verificar
pip list | grep deepbridge
python -c "import deepbridge; print(deepbridge.__version__)"
```

---

### P: Recebi erro relacionado a dependências (numpy, torch, etc.)

**R:** Certifique-se de ter as versões compatíveis:

```bash
# Atualizar dependências
pip install --upgrade numpy pandas scikit-learn torch

# Ou reinstalar tudo
pip uninstall deepbridge deepbridge-distillation deepbridge-synthetic
pip install deepbridge deepbridge-distillation deepbridge-synthetic
```

**Versões recomendadas:**
- Python: 3.8-3.12
- NumPy: >=1.19.0
- PyTorch: >=1.9.0
- Pandas: >=1.2.0

---

## 🧪 Uso e Desenvolvimento

### P: Como verifico a versão instalada?

**R:**

```python
import deepbridge
print(deepbridge.__version__)  # Exemplo: '2.0.0'

# Para módulos específicos
import deepbridge.distillation
import deepbridge.synthetic
print(deepbridge.distillation.__version__)
print(deepbridge.synthetic.__version__)
```

**Via CLI:**
```bash
pip show deepbridge
pip show deepbridge-distillation
pip show deepbridge-synthetic
```

---

### P: Como reportar um bug?

**R:** Use nossos templates de issue no GitHub:

1. Acesse o repositório correspondente:
   - Core: https://github.com/guhaase/DeepBridge/issues
   - Distillation: https://github.com/guhaase/deepbridge-distillation/issues
   - Synthetic: https://github.com/guhaase/deepbridge-synthetic/issues

2. Clique em "New Issue"
3. Escolha o template "Bug Report"
4. Preencha todas as seções (ambiente, código, erro)

**Informações importantes:**
- Versão de todos os pacotes deepbridge instalados
- Python version
- Sistema operacional
- Código mínimo para reproduzir o erro
- Mensagem de erro completa

---

### P: Como contribuir com o projeto?

**R:**

1. **Fork** do repositório desejado
2. **Clone** seu fork localmente
3. **Crie branch** para sua feature: `git checkout -b feature/minha-feature`
4. **Faça commit** das mudanças: `git commit -m "feat: adiciona X"`
5. **Push** para o branch: `git push origin feature/minha-feature`
6. **Abra Pull Request** no GitHub

Consulte `CONTRIBUTING.md` em cada repositório para guidelines detalhadas.

---

## 📚 Recursos e Documentação

### P: Onde encontro a documentação completa?

**R:**

- **Documentação principal:** https://deepbridge.readthedocs.io/
- **Exemplos:** `/examples` em cada repositório
- **Guias de migração:** `desenvolvimento/refatoracao/GUIA_MIGRACAO.md`
- **Changelog:** `CHANGELOG.md` em cada repositório

---

### P: Onde encontro exemplos de código?

**R:**

Cada repositório tem uma pasta `examples/`:

```bash
# Clonar repositórios
git clone https://github.com/guhaase/DeepBridge.git
git clone https://github.com/guhaase/deepbridge-distillation.git
git clone https://github.com/guhaase/deepbridge-synthetic.git

# Explorar exemplos
cd DeepBridge/examples/
cd deepbridge-distillation/examples/
cd deepbridge-synthetic/examples/
```

**Exemplos comuns:**
- `examples/basic_usage.py` - Uso básico do core
- `examples/distillation/knowledge_distillation.py` - Destilação de conhecimento
- `examples/synthetic/generate_data.py` - Geração de dados sintéticos

---

### P: A v1.x ainda recebe suporte?

**R:**

- **Manutenção:** Não. A v1.x não recebe mais atualizações.
- **Bugfixes críticos:** Apenas em casos extremos (segurança).
- **Recomendação:** Migrar para v2.0 o quanto antes.

**Motivo:** A v2.0 oferece:
- Arquitetura modular (instale apenas o necessário)
- Melhor organização de código
- Instalação via PyPI
- CI/CD automatizado
- Documentação aprimorada

---

## 🔧 Troubleshooting Avançado

### P: Instalação falha com erro de permissão

**R:**

```bash
# Opção 1: Usar --user
pip install --user deepbridge

# Opção 2: Usar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate
pip install deepbridge
```

**Nunca use `sudo pip install`!** Isso pode quebrar o Python do sistema.

---

### P: Como limpar cache do pip e reinstalar?

**R:**

```bash
# Limpar cache
pip cache purge

# Desinstalar completamente
pip uninstall -y deepbridge deepbridge-distillation deepbridge-synthetic

# Reinstalar
pip install --no-cache-dir deepbridge deepbridge-distillation deepbridge-synthetic
```

---

### P: Erro ao importar no Jupyter Notebook

**R:**

Certifique-se de que o kernel do Jupyter está usando o ambiente virtual correto:

```bash
# Instalar ipykernel no ambiente virtual
pip install ipykernel

# Registrar kernel
python -m ipykernel install --user --name=deepbridge-env --display-name "Python (DeepBridge)"

# Abrir Jupyter e selecionar o kernel "Python (DeepBridge)"
jupyter notebook
```

**Verificar no notebook:**
```python
import sys
print(sys.executable)  # Deve apontar para seu venv

import deepbridge
print(deepbridge.__version__)
```

---

### P: Como desenvolver/editar código localmente?

**R:**

```bash
# Clonar repositório
git clone https://github.com/guhaase/DeepBridge.git
cd DeepBridge

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate

# Instalar em modo editable
pip install -e .

# Agora mudanças no código são refletidas imediatamente
```

Repita para `deepbridge-distillation` e `deepbridge-synthetic` conforme necessário.

---

## 🆘 Ainda Precisa de Ajuda?

Se sua dúvida não foi respondida:

1. **Pesquise issues existentes:** https://github.com/guhaase/DeepBridge/issues
2. **Abra uma issue:** Use o template "Question"
3. **Discord/Slack:** (se disponível, adicionar link)
4. **Email:** (se disponível, adicionar email de contato)

---

## 📝 Contribuindo com o FAQ

Encontrou uma solução para um problema comum? Ajude a comunidade:

1. Abra um PR adicionando a pergunta/resposta neste FAQ
2. Ou crie uma issue com tag `documentation`

**Formato sugerido:**

```markdown
### P: [Sua pergunta]

**R:** [Sua resposta com código se aplicável]
```

---

**DeepBridge v2.0** - Construindo pontes entre dados e inteligência artificial.
