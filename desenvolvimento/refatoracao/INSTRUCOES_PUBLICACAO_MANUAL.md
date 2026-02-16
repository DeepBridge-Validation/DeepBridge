# Instruções para Publicação Manual - v2.0.0

Este documento contém as instruções para completar as tarefas de publicação que requerem autenticação manual.

---

## 📋 Status das Tarefas

### ✅ Concluído Automaticamente

- [x] Versões atualizadas para 2.0.0
- [x] Tags v2.0.0 criadas e pushed
- [x] Builds criados (poetry build)
- [x] Release notes criados
- [x] Deprecação v1.x implementada

### ⚠️ Pendente (Requer Ação Manual)

- [ ] Publicação no Test PyPI
- [ ] Teste de instalação do Test PyPI
- [ ] Publicação no PyPI oficial
- [ ] Criação de GitHub Releases
- [ ] Publicação v1.63.0 no PyPI
- [ ] Anúncios

---

## 🔑 1. Configurar Tokens PyPI

### Test PyPI (Recomendado testar primeiro)

1. Crie uma conta no Test PyPI: https://test.pypi.org/account/register/
2. Gere um API token: https://test.pypi.org/manage/account/token/
3. Configure o Poetry:

```bash
# Configurar repositório Test PyPI
poetry config repositories.testpypi https://test.pypi.org/legacy/

# Configurar token (substitua YOUR_TEST_TOKEN)
poetry config pypi-token.testpypi pypi-YOUR_TEST_TOKEN_HERE
```

### PyPI Oficial

1. Crie uma conta no PyPI: https://pypi.org/account/register/
2. Gere um API token: https://pypi.org/manage/account/token/
3. Configure o Poetry:

```bash
# Configurar token PyPI oficial (substitua YOUR_TOKEN)
poetry config pypi-token.pypi pypi-YOUR_TOKEN_HERE
```

---

## 📦 2. Publicar no Test PyPI

Execute para cada pacote:

### deepbridge (core)

```bash
cd /home/guhaase/projetos/DeepBridge
poetry publish -r testpypi
```

### deepbridge-distillation

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation
poetry publish -r testpypi
```

### deepbridge-synthetic

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-synthetic
poetry publish -r testpypi
```

### Verificar publicação

Visite os URLs:
- https://test.pypi.org/project/deepbridge/
- https://test.pypi.org/project/deepbridge-distillation/
- https://test.pypi.org/project/deepbridge-synthetic/

---

## 🧪 3. Testar Instalação do Test PyPI

Crie um ambiente virtual temporário e teste:

```bash
# Criar venv temporário
python -m venv /tmp/test_deepbridge_v2
source /tmp/test_deepbridge_v2/bin/activate

# Testar deepbridge
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple deepbridge
python -c "import deepbridge; print(deepbridge.__version__)"
python -c "from deepbridge import DBDataset, Experiment; print('OK')"

# Testar deepbridge-distillation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple deepbridge-distillation
python -c "import deepbridge_distillation; print(deepbridge_distillation.__version__)"
python -c "from deepbridge_distillation import AutoDistiller; print('OK')"

# Testar deepbridge-synthetic
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple deepbridge-synthetic
python -c "import deepbridge_synthetic; print(deepbridge_synthetic.__version__)"
python -c "from deepbridge_synthetic import Synthesize; print('OK')"

# Limpar
deactivate
rm -rf /tmp/test_deepbridge_v2
```

---

## 🚀 4. Publicar no PyPI Oficial

**⚠️ ATENÇÃO: Só execute após validar no Test PyPI!**

### deepbridge (core)

```bash
cd /home/guhaase/projetos/DeepBridge
poetry publish
```

### deepbridge-distillation

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation
poetry publish
```

### deepbridge-synthetic

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-synthetic
poetry publish
```

### Verificar publicação

Visite os URLs:
- https://pypi.org/project/deepbridge/
- https://pypi.org/project/deepbridge-distillation/
- https://pypi.org/project/deepbridge-synthetic/

---

## 🏷️ 5. Criar GitHub Releases

### Autenticar GitHub CLI

```bash
gh auth login
# Siga as instruções interativas
```

### Criar Releases

Os release notes já foram preparados em:
- `RELEASE_NOTES_v2.0.0.md` (deepbridge)
- `RELEASE_NOTES_DISTILLATION_v2.0.0.md` (deepbridge-distillation)
- `RELEASE_NOTES_SYNTHETIC_v2.0.0.md` (deepbridge-synthetic)

#### deepbridge

```bash
cd /home/guhaase/projetos/DeepBridge
gh release create v2.0.0 \
  --title "DeepBridge v2.0.0 - Major Release" \
  --notes-file desenvolvimento/refatoracao/RELEASE_NOTES_v2.0.0.md \
  --latest
```

#### deepbridge-distillation

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation
gh release create v2.0.0 \
  --title "deepbridge-distillation v2.0.0 - Initial Release" \
  --notes-file /home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/RELEASE_NOTES_DISTILLATION_v2.0.0.md \
  --latest
```

#### deepbridge-synthetic

```bash
cd /home/guhaase/projetos/deepbridge_toolkit/deepbridge-synthetic
gh release create v2.0.0 \
  --title "deepbridge-synthetic v2.0.0 - Initial Standalone Release" \
  --notes-file /home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/RELEASE_NOTES_SYNTHETIC_v2.0.0.md \
  --latest
```

---

## 📢 6. Publicar v1.63.0 (Deprecação)

A versão v1.63.0 já foi criada com o deprecation warning. Para publicá-la:

```bash
cd /home/guhaase/projetos/DeepBridge
git checkout master  # Ou v1.63.0 tag
poetry publish
```

Isso notificará usuários da v1.x sobre a migração para v2.0.

---

## 📣 7. Criar Anúncios

### GitHub Discussions

Crie um post em GitHub Discussions anunciando o v2.0:

**Título**: "DeepBridge v2.0.0 Released - Package Split & Focus on Validation"

**Conteúdo**: Use o template em `ANUNCIO_v2.0.0.md` (será criado automaticamente)

### Atualizar README

O README principal já foi atualizado com informações do v2.0.

### Opcional: Redes Sociais

Se o projeto tiver presença em redes sociais, anuncie:
- Twitter/X
- LinkedIn
- Blog técnico
- Reddit (r/MachineLearning, r/Python)

---

## ✅ Checklist de Verificação Final

Após completar todas as etapas acima, verifique:

- [ ] `pip install deepbridge` funciona
- [ ] `pip install deepbridge-distillation` funciona
- [ ] `pip install deepbridge-synthetic` funciona
- [ ] PyPI mostra versão 2.0.0 para todos os pacotes
- [ ] GitHub Releases criados e visíveis
- [ ] v1.63.0 publicado com deprecation warning
- [ ] Anúncio no GitHub Discussions criado
- [ ] README atualizado

---

## 🆘 Troubleshooting

### Erro: "Repository already exists"

Se o pacote já foi publicado anteriormente, você pode ter conflitos de versão. Verifique:
1. A versão no PyPI: `pip show <package>`
2. A versão local: `poetry version`

### Erro: "Invalid credentials"

Verifique se o token está configurado corretamente:
```bash
poetry config --list | grep pypi-token
```

### Erro: "Package not found" no Test PyPI

Ao instalar do Test PyPI, sempre use `--extra-index-url https://pypi.org/simple` para que as dependências sejam encontradas.

---

**Data de criação**: 2026-02-16
**Criado para**: DeepBridge v2.0.0 Release
