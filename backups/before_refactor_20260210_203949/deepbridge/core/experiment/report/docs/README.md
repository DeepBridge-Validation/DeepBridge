# Documentação - Sistema de Reports

**Versão:** 1.0
**Data:** 05/11/2025

---

## 📚 Documentos Disponíveis

Este diretório contém toda a documentação técnica do sistema de geração de reports do DeepBridge.

### 📖 Para Desenvolvedores

1. **[PADROES_CODIGO.md](./PADROES_CODIGO.md)** - ⭐ LEIA PRIMEIRO
   - Padrões de código obrigatórios
   - Como usar CSSManager
   - Serialização JSON segura
   - Tratamento de erros e logging
   - Exemplos completos
   - **Quando ler:** Antes de modificar qualquer renderer

2. **[GUIA_RENDERER.md](./GUIA_RENDERER.md)** - Passo-a-passo
   - Como criar um novo tipo de renderer
   - Tutorial completo com exemplo
   - Troubleshooting comum
   - Checklist de implementação
   - **Quando ler:** Ao criar um novo tipo de report

---

## 🚀 Quick Start

### Você quer...

#### 1. Modificar um renderer existente?
→ Leia: `PADROES_CODIGO.md`

**Principais pontos:**
- Use `CSSManager` para CSS
- Use `json_utils` para JSON
- Herde de `BaseRenderer`
- Siga nomenclatura padrão

#### 2. Criar um novo tipo de report?
→ Leia: `GUIA_RENDERER.md`

**Passos:**
1. Criar transformer
2. Criar template
3. Implementar renderer
4. Testar
5. Integrar

#### 3. Entender a arquitetura?
→ Leia: `/planejamento_report/ANALISE_ARQUITETURA_REPORTS.md`

---

## 📋 Checklist Rápido

### Antes de Commitar

- [ ] Código segue `PADROES_CODIGO.md`
- [ ] Usa `CSSManager` para CSS
- [ ] Usa `json_utils` para JSON
- [ ] Tem docstrings completos
- [ ] Tem logging apropriado
- [ ] Trata erros corretamente
- [ ] Testes passando
- [ ] Code review solicitado

### Antes de Criar PR

- [ ] Todos os checklist acima ✅
- [ ] Testes de integração passando
- [ ] Documentação atualizada
- [ ] Changelog atualizado
- [ ] Performance verificada
- [ ] Sem duplicação de código

---

## 🎓 Materiais de Referência

### Código de Exemplo

**Simple Renderers (Interactive):**
- `uncertainty_renderer_simple.py` - Exemplo completo
- `robustness_renderer_simple.py` - Bom exemplo
- `resilience_renderer_simple.py` - Padrão atual

**Static Renderers:**
- `static/static_uncertainty_renderer.py`
- `static/static_robustness_renderer.py`
- `static/base_static_renderer.py` - Base class

### Módulos Importantes

**Core:**
- `base_renderer.py` - Classe base para todos renderers
- `css_manager.py` - Gerenciamento de CSS
- `utils/json_utils.py` - Utilities JSON seguras
- `template_manager.py` - Gerenciamento de templates

**Transformers:**
- `transformers/*_simple.py` - Transformadores de dados

---

## 📊 Estrutura do Sistema

```
deepbridge/core/experiment/report/
├── docs/                          ← Você está aqui
│   ├── README.md                  ← Este arquivo
│   ├── PADROES_CODIGO.md          ← Padrões obrigatórios
│   └── GUIA_RENDERER.md           ← Tutorial passo-a-passo
│
├── renderers/                     ← Renderers
│   ├── base_renderer.py           ← Base class
│   ├── *_renderer_simple.py       ← Simple renderers
│   └── static/                    ← Static renderers
│       ├── base_static_renderer.py
│       └── static_*_renderer.py
│
├── transformers/                  ← Data transformers
│   ├── *_simple.py                ← Simple transformers
│   └── static/                    ← Static transformers
│
├── utils/                         ← Utilities
│   ├── json_utils.py              ← JSON serialization ⭐
│   ├── json_formatter.py          ← JSON formatting
│   └── seaborn_utils.py           ← Chart generation
│
├── css_manager.py                 ← CSS management ⭐
├── asset_manager.py               ← Asset management
├── template_manager.py            ← Template management
└── report_manager.py              ← Orchestration
```

---

## 🔍 Como Encontrar Informação

### Procuro informação sobre...

**CSS e estilos:**
- Ver: `PADROES_CODIGO.md` → Seção "Uso de CSSManager"
- Código: `css_manager.py`

**JSON e serialização:**
- Ver: `PADROES_CODIGO.md` → Seção "Serialização JSON"
- Código: `utils/json_utils.py`

**Criar novo renderer:**
- Ver: `GUIA_RENDERER.md` → Tutorial completo
- Exemplo: `uncertainty_renderer_simple.py`

**Padrões de código:**
- Ver: `PADROES_CODIGO.md` → Todos os padrões
- Checklist: Seção "Checklist de Code Review"

**Arquitetura geral:**
- Ver: `/planejamento_report/ANALISE_ARQUITETURA_REPORTS.md`
- Ver: `/planejamento_report/ROADMAP_GERAL.md`

**Troubleshooting:**
- Ver: `GUIA_RENDERER.md` → Seção "Troubleshooting"

---

## 🤝 Contribuindo

### Workflow

1. **Ler documentação**
   - `PADROES_CODIGO.md` (obrigatório)
   - `GUIA_RENDERER.md` (se criar novo)

2. **Implementar**
   - Seguir padrões documentados
   - Escrever testes
   - Adicionar logging

3. **Code Review**
   - Auto-review com checklist
   - Solicitar review do time
   - Endereçar feedback

4. **Merge**
   - Após aprovação
   - Atualizar changelog
   - Atualizar docs se necessário

### Atualizando Documentação

Se você modificou algo significativo:

- [ ] Atualize `PADROES_CODIGO.md` se mudou padrões
- [ ] Atualize `GUIA_RENDERER.md` se mudou processo
- [ ] Atualize este README se mudou estrutura
- [ ] Atualize data de "Última Atualização"
- [ ] Incremente versão se breaking change

---

## 📞 Suporte

### Canais

- **Dúvidas gerais:** Slack #deepbridge-reports
- **Issues técnicos:** GitHub Issues
- **Code review:** Pull Requests
- **Discussões:** GitHub Discussions

### Pessoas de Contato

- **Tech Lead:** [A definir]
- **Report System Owner:** [A definir]
- **Code Reviewers:** [A definir]

---

## 🎯 Metas do Refatoramento

Este sistema está em processo de refatoramento (Fase 1 - Quick Wins).

**Objetivos:**
- ✅ Eliminar duplicação de código
- ✅ Padronizar uso de CSS (CSSManager)
- ✅ Melhorar serialização JSON
- ✅ Documentar padrões
- ⏳ Aumentar cobertura de testes
- ⏳ Melhorar performance

**Status Atual:**
- Fase 1: 🟢 Em andamento
- Documentação: ✅ Completa

Ver mais detalhes em: `/planejamento_report/`

---

## 📝 Histórico de Versões

| Versão | Data | Mudanças |
|--------|------|----------|
| 1.0 | 2025-11-05 | Documentação inicial criada |

---

## 📖 Leituras Recomendadas

**Livros:**
- "Refactoring" - Martin Fowler
- "Clean Code" - Robert C. Martin

**Artigos:**
- [Jinja2 Template Designer Documentation](https://jinja.palletsprojects.com/)
- [Plotly Python Documentation](https://plotly.com/python/)

---

**Última Atualização:** 05/11/2025
**Mantido por:** Tech Lead
**Versão:** 1.0

---

**Boa documentação = Código melhor!** 📚✨
