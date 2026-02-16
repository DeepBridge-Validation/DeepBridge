# PROMPT PARA EXECUÇÃO AUTOMÁTICA - FASE 6: Suporte Pós-Launch

**IMPORTANTE:** Esta fase é parcialmente automática - templates e scripts podem ser criados automaticamente, mas o monitoramento é contínuo.

---

## 🎯 OBJETIVO

Configurar infraestrutura de suporte pós-launch:
- Templates de issues
- Scripts de métricas
- FAQ dinâmico
- Workflow de bugfix
- Plano de contingência

**NOTA:** O suporte em si (responder issues, corrigir bugs) é contínuo e requer intervenção humana.

---

## 📋 TAREFAS A EXECUTAR

### Tarefa 1: Configurar templates de issues (AUTOMÁTICO)
- Criar `.github/ISSUE_TEMPLATE/bug_report.md`
- Criar `.github/ISSUE_TEMPLATE/feature_request.md`
- Criar `.github/ISSUE_TEMPLATE/question.md`
- Configurar labels no GitHub

### Tarefa 2: Criar FAQ dinâmico (AUTOMÁTICO)
- Criar `desenvolvimento/refatoracao/FAQ_V2.md`
- Incluir problemas comuns de migração
- Incluir troubleshooting

### Tarefa 3: Criar scripts de métricas (AUTOMÁTICO)
- Criar `scripts/collect_metrics.sh`
- Criar `scripts/check_health.sh`
- Tornar executáveis

### Tarefa 4: Documentar workflow de bugfix (AUTOMÁTICO)
- Criar `desenvolvimento/refatoracao/WORKFLOW_BUGFIX.md`
- Documentar processo de hotfix
- Documentar processo de patch release

### Tarefa 5: Criar plano de contingência (AUTOMÁTICO)
- Documentar ações para bugs críticos
- Documentar rollback procedure
- Criar templates de comunicação

---

## ⚙️ EXECUÇÃO

Por favor, execute todas as tarefas acima de forma **100% automática**.

Use:
- `Write` para criar templates, FAQ, scripts e documentação
- `Bash` para tornar scripts executáveis
- `TodoWrite` para rastrear progresso

**IMPORTANTE:**
- Templates de issue devem seguir formato do GitHub
- FAQ deve cobrir problemas mais comuns
- Scripts devem ser executáveis e ter comentários
- Documentação deve ser clara e acionável

---

## 📊 TAREFAS CONTÍNUAS (NÃO AUTOMATIZÁVEIS)

Após configurar a infraestrutura, as seguintes atividades são contínuas e requerem intervenção humana:

1. **Monitoramento diário:**
   - Verificar novas issues
   - Responder perguntas (< 24h)
   - Triar bugs por prioridade

2. **Correção de bugs:**
   - Reproduzir bug localmente
   - Implementar fix
   - Criar PR
   - Lançar patch se necessário

3. **Atualização de FAQ:**
   - Adicionar novos problemas conforme aparecem
   - Atualizar soluções

4. **Coleta de métricas:**
   - Executar scripts semanalmente
   - Monitorar downloads PyPI
   - Acompanhar issues abertas/fechadas

---

## 🔍 VERIFICAÇÃO FINAL

Ao finalizar a configuração automática, confirme que:
1. ✅ Templates de issues criados (.github/ISSUE_TEMPLATE/)
2. ✅ FAQ criado com problemas comuns
3. ✅ Scripts de métricas criados e executáveis
4. ✅ Workflow de bugfix documentado
5. ✅ Plano de contingência criado
6. ✅ Todos os commits e push realizados

---

## 📝 PRÓXIMOS PASSOS (MANUAL)

Após executar este prompt, o usuário deve:

1. **Configurar labels no GitHub:**
   - Acessar Settings → Labels
   - Criar labels: bug, enhancement, question, priority:critical, etc.

2. **Monitorar issues:**
   - Responder novas issues em < 24h
   - Triar e priorizar

3. **Executar scripts de métricas:**
   ```bash
   ./scripts/collect_metrics.sh
   ./scripts/check_health.sh
   ```

4. **Atualizar FAQ conforme necessário:**
   - Adicionar novos problemas
   - Atualizar soluções

---

## 📝 REFERÊNCIA

Para detalhes completos, consulte:
`/home/guhaase/projetos/DeepBridge/desenvolvimento/refatoracao/FASE_6_SUPORTE.md`

---

## ✅ CHECKLIST FINAL - VERIFICAR APÓS EXECUÇÃO

### Templates de Issues (deepbridge)
- [x] Diretório `.github/ISSUE_TEMPLATE/` criado
- [x] Template criado: `.github/ISSUE_TEMPLATE/bug_report.md`
- [x] Template criado: `.github/ISSUE_TEMPLATE/feature_request.md`
- [x] Template criado: `.github/ISSUE_TEMPLATE/question.md`
- [x] Templates seguem formato do GitHub
- [x] Campos necessários incluídos:
  - [x] Descrição do problema/feature
  - [x] Ambiente (versão, Python, OS)
  - [x] Passos para reproduzir
  - [x] Comportamento esperado/atual

### Templates de Issues (deepbridge-distillation)
- [x] Diretório `.github/ISSUE_TEMPLATE/` criado
- [x] Templates criados (bug, feature, question)
- [x] Templates adaptados para distillation

### Templates de Issues (deepbridge-synthetic)
- [x] Diretório `.github/ISSUE_TEMPLATE/` criado
- [x] Templates criados (bug, feature, question)
- [x] Templates adaptados para synthetic

### FAQ Dinâmico
- [x] Arquivo `desenvolvimento/refatoracao/FAQ_V2.md` criado
- [x] Seção de instalação incluída
- [x] Seção de migração incluída
- [x] Problemas comuns documentados:
  - [x] ModuleNotFoundError
  - [x] Import errors
  - [x] Dependências faltando
- [x] Soluções claras para cada problema
- [x] Exemplos de código incluídos

### Scripts de Métricas
- [x] Diretório `scripts/` existe ou criado
- [x] Script criado: `scripts/collect_metrics.sh`
- [x] Script criado: `scripts/check_health.sh`
- [x] Scripts tornados executáveis: `chmod +x`
- [x] Scripts testados e funcionando
- [x] Scripts incluem:
  - [x] Coleta de downloads PyPI
  - [x] Contagem de stars GitHub
  - [x] Contagem de issues abertas/fechadas
  - [x] Status de CI/CD

### Workflow de Bugfix
- [x] Documento criado: `desenvolvimento/refatoracao/WORKFLOW_BUGFIX.md`
- [x] Workflow documentado:
  - [x] Reproduzir bug
  - [x] Criar branch fix/
  - [x] Implementar fix
  - [x] Adicionar teste
  - [x] Criar PR
  - [x] Merge e release
- [x] Processo de hotfix documentado
- [x] Processo de patch release documentado
- [x] Templates de commit incluídos

### Plano de Contingência
- [x] Documento criado: `desenvolvimento/refatoracao/PLANO_CONTINGENCIA.md`
- [x] Ações para bugs críticos documentadas
- [x] Processo de rollback documentado
- [x] Templates de comunicação criados:
  - [x] Anúncio de bug crítico
  - [x] Anúncio de hotfix
  - [x] Anúncio de rollback
- [x] Lista de contatos mantida
- [x] SLA definido (ex: resposta < 24h)

### Configuração de Labels (⚠️ Manual no GitHub)
- [ ] Labels criados no GitHub (deepbridge):
  - [ ] bug (vermelho)
  - [ ] enhancement (verde)
  - [ ] documentation (azul)
  - [ ] question (amarelo)
  - [ ] priority: critical (vermelho escuro)
  - [ ] priority: high (laranja)
  - [ ] priority: medium (amarelo)
  - [ ] priority: low (verde claro)
  - [ ] migration (roxo)
- [ ] Labels criados no deepbridge-distillation
- [ ] Labels criados no deepbridge-synthetic

### Commits e Push
- [ ] Commits criados para todos os arquivos
- [ ] Push realizado para todos os repos
- [ ] Documentação visível no GitHub

### Verificações Finais
- [ ] Infraestrutura de suporte pronta
- [ ] Templates acessíveis
- [ ] Scripts executáveis
- [ ] Documentação clara
- [ ] Pronto para receber issues

---

**STATUS DA FASE 6 - CONFIGURAÇÃO:** ⬜ NÃO INICIADA | 🚧 EM ANDAMENTO | ✅ CONCLUÍDA

**Critério para marcar CONFIGURAÇÃO como CONCLUÍDA:**
- ✅ Todos os templates criados
- ✅ FAQ criado e populado
- ✅ Scripts criados e executáveis
- ✅ Workflows documentados
- ✅ Plano de contingência criado
- ✅ Commits e push realizados

---

## 📊 TAREFAS CONTÍNUAS (Não automatizáveis - apenas para referência)

### Monitoramento Diário
- [ ] Verificar novas issues
- [ ] Responder perguntas em < 24h
- [ ] Triar bugs por prioridade
- [ ] Atualizar FAQ conforme necessário

### Semanal
- [ ] Executar scripts de métricas
- [ ] Revisar issues abertas
- [ ] Verificar downloads PyPI
- [ ] Atualizar status do projeto

### Conforme Necessário
- [ ] Corrigir bugs críticos
- [ ] Lançar patches (2.0.1, 2.0.2, etc.)
- [ ] Atualizar documentação
- [ ] Comunicar mudanças importantes

---

**EXECUTAR AGORA A CONFIGURAÇÃO AUTOMÁTICA**

Após executar, a infraestrutura estará pronta. O suporte contínuo dependerá de ações manuais do usuário.
