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

**EXECUTAR AGORA A CONFIGURAÇÃO AUTOMÁTICA**

Após executar, a infraestrutura estará pronta. O suporte contínuo dependerá de ações manuais do usuário.
