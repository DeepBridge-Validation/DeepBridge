#!/bin/bash

# Script para criar GitHub Releases para DeepBridge v2.0.0

set -e

echo "=================================================="
echo "Script de Criação de GitHub Releases v2.0.0"
echo "=================================================="
echo ""

# Verificar autenticação gh
echo "Verificando autenticação do GitHub CLI..."
if ! gh auth status > /dev/null 2>&1; then
    echo "❌ Você não está autenticado no GitHub CLI."
    echo ""
    echo "Por favor, execute:"
    echo "  gh auth login"
    echo ""
    exit 1
fi

echo "✅ Autenticado no GitHub CLI"
echo ""

# Função para criar release
criar_release() {
    local REPO_PATH=$1
    local TAG=$2
    local TITLE=$3
    local NOTES_FILE=$4

    echo "------------------------------------------------"
    echo "Criando release para: $TITLE"
    echo "Repositório: $REPO_PATH"
    echo "Tag: $TAG"
    echo "------------------------------------------------"

    cd "$REPO_PATH"

    if gh release view "$TAG" > /dev/null 2>&1; then
        echo "⚠️  Release $TAG já existe. Pulando..."
    else
        gh release create "$TAG" \
            --title "$TITLE" \
            --notes-file "$NOTES_FILE" \
            --verify-tag
        echo "✅ Release criado com sucesso!"
    fi

    echo ""
}

# Criar release do deepbridge
criar_release \
    "/home/guhaase/projetos/DeepBridge" \
    "v2.0.0" \
    "DeepBridge v2.0.0 - Modular Architecture" \
    "/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/RELEASE_NOTES_v2.0.0.md"

# Criar release do deepbridge-distillation
criar_release \
    "/home/guhaase/projetos/deepbridge_toolkit/deepbridge-distillation" \
    "v2.0.0" \
    "DeepBridge Distillation v2.0.0" \
    "/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/RELEASE_NOTES_DISTILLATION_v2.0.0.md"

# Criar release do deepbridge-synthetic
criar_release \
    "/home/guhaase/projetos/deepbridge_toolkit/deepbridge-synthetic" \
    "v2.0.0" \
    "DeepBridge Synthetic v2.0.0" \
    "/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/RELEASE_NOTES_SYNTHETIC_v2.0.0.md"

echo "=================================================="
echo "✅ Todos os releases foram criados com sucesso!"
echo "=================================================="
echo ""
echo "Verifique os releases em:"
echo "  - https://github.com/DeepBridge-Validation/DeepBridge/releases"
echo "  - https://github.com/DeepBridge-Validation/deepbridge-distillation/releases"
echo "  - https://github.com/DeepBridge-Validation/deepbridge-synthetic/releases"
echo ""
