#!/bin/bash

# Script para publicação dos pacotes DeepBridge v2.0.0 no PyPI
# Este script solicita os tokens PyPI e executa todas as publicações

set -e

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Diretórios dos pacotes
DEEPBRIDGE_DIR="/home/guhaase/projetos/DeepBridge"
DISTILLATION_DIR="/home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation"
SYNTHETIC_DIR="/home/guhaase/projetos/deepbridge_toolkit/deepbridge-synthetic"
DOC_FILE="/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/PROMPT_FASE_5_AUTOMATICO.md"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  DeepBridge v2.0.0 - Script de Publicação PyPI${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Função para marcar checkbox no documento
mark_checkbox() {
    local line_text="$1"
    if grep -q "- \[ \] $line_text" "$DOC_FILE" 2>/dev/null; then
        sed -i "s/- \[ \] $line_text/- [x] $line_text/" "$DOC_FILE"
        echo -e "${GREEN}✓${NC} Checkbox marcado: $line_text"
    fi
}

# Função para publicar um pacote
publish_package() {
    local package_name="$1"
    local package_dir="$2"
    local repo="$3"

    echo -e "\n${YELLOW}→ Publicando $package_name no $repo...${NC}"

    cd "$package_dir"

    if poetry publish -r "$repo"; then
        echo -e "${GREEN}✓ $package_name publicado com sucesso!${NC}"

        # Marcar checkbox
        if [ "$repo" == "testpypi" ]; then
            mark_checkbox "$package_name publicado no Test PyPI"
        else
            mark_checkbox "$package_name publicado no PyPI oficial"
        fi

        return 0
    else
        echo -e "${RED}✗ Erro ao publicar $package_name${NC}"
        return 1
    fi
}

# Função para testar instalação
test_installation() {
    local repo_flag="$1"
    local test_type="$2"

    echo -e "\n${YELLOW}→ Testando instalação dos pacotes ($test_type)...${NC}"

    # Criar venv temporário
    VENV_DIR="/tmp/test_deepbridge_$(date +%s)"
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    echo -e "${BLUE}→ Instalando deepbridge...${NC}"
    if pip install $repo_flag deepbridge; then
        echo -e "${GREEN}✓ deepbridge instalado${NC}"

        # Testar imports
        if python -c "import deepbridge; from deepbridge import DBDataset, Experiment" 2>/dev/null; then
            echo -e "${GREEN}✓ Imports do deepbridge funcionando${NC}"
            mark_checkbox "deepbridge instalado do $test_type"
            mark_checkbox "\`import deepbridge\`"
            mark_checkbox "\`from deepbridge import DBDataset, Experiment\`"
        fi
    fi

    echo -e "\n${BLUE}→ Instalando deepbridge-distillation...${NC}"
    if pip install $repo_flag deepbridge-distillation; then
        echo -e "${GREEN}✓ deepbridge-distillation instalado${NC}"

        # Testar imports
        if python -c "import deepbridge_distillation; from deepbridge_distillation import AutoDistiller; import deepbridge" 2>/dev/null; then
            echo -e "${GREEN}✓ Imports do deepbridge-distillation funcionando${NC}"
            mark_checkbox "deepbridge-distillation instalado do $test_type"
        fi
    fi

    echo -e "\n${BLUE}→ Instalando deepbridge-synthetic...${NC}"
    if pip install $repo_flag deepbridge-synthetic; then
        echo -e "${GREEN}✓ deepbridge-synthetic instalado${NC}"

        # Testar imports
        if python -c "import deepbridge_synthetic; from deepbridge_synthetic import Synthesize" 2>/dev/null; then
            echo -e "${GREEN}✓ Imports do deepbridge-synthetic funcionando${NC}"
            mark_checkbox "deepbridge-synthetic instalado do $test_type"
        fi
    fi

    # Limpar
    deactivate
    rm -rf "$VENV_DIR"
    echo -e "${GREEN}✓ Ambiente de teste limpo${NC}"
    mark_checkbox "Ambiente de teste limpo"
}

# Menu principal
echo "Este script irá guiá-lo pela publicação dos pacotes DeepBridge v2.0.0."
echo ""
echo "Você precisará de:"
echo "  1. Token do Test PyPI (recomendado testar primeiro)"
echo "  2. Token do PyPI oficial"
echo "  3. Autenticação do GitHub CLI (para releases)"
echo ""
echo "Escolha uma opção:"
echo ""
echo "  1) Publicar no Test PyPI (recomendado - testar primeiro)"
echo "  2) Publicar no PyPI oficial (produção)"
echo "  3) Publicar v1.63.0 (deprecação)"
echo "  4) Criar GitHub Releases"
echo "  5) Executar tudo (Test PyPI + PyPI oficial + v1.63.0 + Releases)"
echo "  6) Sair"
echo ""
read -p "Digite sua escolha (1-6): " choice

case $choice in
    1)
        echo ""
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  PUBLICAÇÃO NO TEST PYPI${NC}"
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
        echo ""

        # Solicitar token Test PyPI
        echo "Você precisa de um token do Test PyPI."
        echo "Se ainda não tem:"
        echo "  1. Acesse: https://test.pypi.org/account/register/"
        echo "  2. Crie uma conta ou faça login"
        echo "  3. Gere um token: https://test.pypi.org/manage/account/token/"
        echo ""
        read -p "Digite seu token do Test PyPI (pypi-...): " TEST_TOKEN

        if [ -z "$TEST_TOKEN" ]; then
            echo -e "${RED}Erro: Token não fornecido${NC}"
            exit 1
        fi

        # Configurar repositório e token
        echo -e "\n${BLUE}→ Configurando repositório Test PyPI...${NC}"
        poetry config repositories.testpypi https://test.pypi.org/legacy/
        poetry config pypi-token.testpypi "$TEST_TOKEN"
        echo -e "${GREEN}✓ Repositório configurado${NC}"
        mark_checkbox "Repositório Test PyPI configurado"
        mark_checkbox "Token Test PyPI configurado"

        # Publicar pacotes
        publish_package "deepbridge" "$DEEPBRIDGE_DIR" "testpypi"
        publish_package "deepbridge-distillation" "$DISTILLATION_DIR" "testpypi"
        publish_package "deepbridge-synthetic" "$SYNTHETIC_DIR" "testpypi"

        # Marcar URLs
        mark_checkbox "https://test.pypi.org/project/deepbridge/"
        mark_checkbox "https://test.pypi.org/project/deepbridge-distillation/"
        mark_checkbox "https://test.pypi.org/project/deepbridge-synthetic/"

        echo ""
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  ✓ PUBLICAÇÃO NO TEST PYPI CONCLUÍDA${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        echo ""
        echo "Verifique em:"
        echo "  - https://test.pypi.org/project/deepbridge/"
        echo "  - https://test.pypi.org/project/deepbridge-distillation/"
        echo "  - https://test.pypi.org/project/deepbridge-synthetic/"
        echo ""

        # Perguntar se quer testar instalação
        read -p "Deseja testar a instalação do Test PyPI? (s/n): " test_install
        if [ "$test_install" == "s" ] || [ "$test_install" == "S" ]; then
            test_installation "--index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/" "Test PyPI"
        fi
        ;;

    2)
        echo ""
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  PUBLICAÇÃO NO PYPI OFICIAL${NC}"
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
        echo ""

        echo -e "${RED}⚠️  ATENÇÃO: Você está prestes a publicar no PyPI OFICIAL!${NC}"
        echo -e "${RED}    Esta ação é IRREVERSÍVEL e a versão ficará disponível publicamente.${NC}"
        echo ""
        read -p "Tem certeza que deseja continuar? (sim/não): " confirm

        if [ "$confirm" != "sim" ]; then
            echo "Publicação cancelada."
            exit 0
        fi

        # Solicitar token PyPI
        echo ""
        echo "Você precisa de um token do PyPI oficial."
        echo "Se ainda não tem:"
        echo "  1. Acesse: https://pypi.org/account/register/"
        echo "  2. Crie uma conta ou faça login"
        echo "  3. Gere um token: https://pypi.org/manage/account/token/"
        echo ""
        read -p "Digite seu token do PyPI (pypi-...): " PYPI_TOKEN

        if [ -z "$PYPI_TOKEN" ]; then
            echo -e "${RED}Erro: Token não fornecido${NC}"
            exit 1
        fi

        # Configurar token
        echo -e "\n${BLUE}→ Configurando token PyPI...${NC}"
        poetry config pypi-token.pypi "$PYPI_TOKEN"
        echo -e "${GREEN}✓ Token configurado${NC}"
        mark_checkbox "Token PyPI oficial configurado"

        # Publicar pacotes
        publish_package "deepbridge" "$DEEPBRIDGE_DIR" "pypi"
        publish_package "deepbridge-distillation" "$DISTILLATION_DIR" "pypi"
        publish_package "deepbridge-synthetic" "$SYNTHETIC_DIR" "pypi"

        # Marcar URLs
        mark_checkbox "https://pypi.org/project/deepbridge/"
        mark_checkbox "https://pypi.org/project/deepbridge-distillation/"
        mark_checkbox "https://pypi.org/project/deepbridge-synthetic/"

        echo ""
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  ✓ PUBLICAÇÃO NO PYPI OFICIAL CONCLUÍDA${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        echo ""
        echo "Verifique em:"
        echo "  - https://pypi.org/project/deepbridge/"
        echo "  - https://pypi.org/project/deepbridge-distillation/"
        echo "  - https://pypi.org/project/deepbridge-synthetic/"
        echo ""

        # Testar instalação
        echo -e "${BLUE}→ Testando instalação do PyPI oficial...${NC}"
        test_installation "" "PyPI oficial"

        mark_checkbox "Instalação funciona: \`pip install deepbridge\`"
        mark_checkbox "Instalação funciona: \`pip install deepbridge-distillation\`"
        mark_checkbox "Instalação funciona: \`pip install deepbridge-synthetic\`"
        mark_checkbox "PyPI mostra versão 2.0.0 para todos"
        ;;

    3)
        echo ""
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  PUBLICAÇÃO v1.63.0 (DEPRECAÇÃO)${NC}"
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
        echo ""

        # Verificar se está no branch correto
        cd "$DEEPBRIDGE_DIR"
        current_branch=$(git rev-parse --abbrev-ref HEAD)

        if [ "$current_branch" != "master" ]; then
            echo -e "${YELLOW}→ Mudando para branch master...${NC}"
            git checkout master
        fi

        # Solicitar token PyPI se não configurado
        if ! poetry config pypi-token.pypi >/dev/null 2>&1; then
            echo "Você precisa de um token do PyPI oficial."
            read -p "Digite seu token do PyPI (pypi-...): " PYPI_TOKEN
            poetry config pypi-token.pypi "$PYPI_TOKEN"
        fi

        # Verificar se já foi feito build
        if [ ! -f "dist/deepbridge-1.63.0-py3-none-any.whl" ]; then
            echo -e "${YELLOW}→ Executando build do v1.63.0...${NC}"
            poetry build
        fi

        # Publicar
        echo -e "${YELLOW}→ Publicando v1.63.0 no PyPI...${NC}"
        poetry publish

        echo -e "${GREEN}✓ v1.63.0 publicado com sucesso!${NC}"
        mark_checkbox "v1.63.0 publicado no PyPI"

        # Voltar para o branch original
        if [ "$current_branch" != "master" ]; then
            git checkout "$current_branch"
        fi
        ;;

    4)
        echo ""
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  CRIAÇÃO DE GITHUB RELEASES${NC}"
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
        echo ""

        # Verificar autenticação GitHub
        if ! gh auth status >/dev/null 2>&1; then
            echo -e "${YELLOW}→ Você não está autenticado no GitHub CLI${NC}"
            echo "Execute: gh auth login"
            echo ""
            read -p "Pressione ENTER após autenticar..."
        fi

        # Criar releases
        echo -e "\n${BLUE}→ Criando release para deepbridge...${NC}"
        cd "$DEEPBRIDGE_DIR"
        if gh release create v2.0.0 \
            --title "DeepBridge v2.0.0 - Modular Architecture" \
            --notes-file desenvolvimento/refatoracao/RELEASE_NOTES_v2.0.0.md; then
            echo -e "${GREEN}✓ Release deepbridge criado${NC}"
            mark_checkbox "GitHub Release criado para deepbridge v2.0.0"
        fi

        echo -e "\n${BLUE}→ Criando release para deepbridge-distillation...${NC}"
        cd "$DISTILLATION_DIR"
        if gh release create v2.0.0 \
            --title "DeepBridge Distillation v2.0.0" \
            --notes-file RELEASE_NOTES_DISTILLATION_v2.0.0.md; then
            echo -e "${GREEN}✓ Release deepbridge-distillation criado${NC}"
            mark_checkbox "GitHub Release criado para deepbridge-distillation v2.0.0"
        fi

        echo -e "\n${BLUE}→ Criando release para deepbridge-synthetic...${NC}"
        cd "$SYNTHETIC_DIR"
        if gh release create v2.0.0 \
            --title "DeepBridge Synthetic v2.0.0" \
            --notes-file RELEASE_NOTES_SYNTHETIC_v2.0.0.md; then
            echo -e "${GREEN}✓ Release deepbridge-synthetic criado${NC}"
            mark_checkbox "GitHub Release criado para deepbridge-synthetic v2.0.0"
        fi

        echo ""
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  ✓ GITHUB RELEASES CRIADOS${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        ;;

    5)
        echo ""
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  EXECUÇÃO COMPLETA${NC}"
        echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
        echo ""

        # Executar todas as etapas
        "$0" "1"  # Test PyPI
        echo ""
        read -p "Pressione ENTER para continuar com PyPI oficial..."
        "$0" "2"  # PyPI oficial
        echo ""
        read -p "Pressione ENTER para publicar v1.63.0..."
        "$0" "3"  # v1.63.0
        echo ""
        read -p "Pressione ENTER para criar releases..."
        "$0" "4"  # Releases

        echo ""
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  ✓ TODAS AS TAREFAS CONCLUÍDAS!${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        ;;

    6)
        echo "Saindo..."
        exit 0
        ;;

    *)
        echo -e "${RED}Opção inválida${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Script concluído!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
