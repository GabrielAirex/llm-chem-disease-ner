# Pipeline de Limpeza e Verificação de PMIDs

Esta pasta contém scripts para limpeza e verificação de PMIDs nos indícios encontrados.

## Scripts Disponíveis

### 1. `verify_and_clean_pmids.py`
Script completo que verifica PMIDs faltantes, detecta duplicados e remove os mais antigos.

**Funcionalidades:**
- Verifica se todos os 1500 PMIDs do gold standard estão presentes
- Detecta arquivos JSON duplicados (mesmo PMID)
- Remove duplicados, mantendo apenas o mais recente
- Gera arquivos `pmids_faltantes_{strategy}.txt` para cada estratégia

**Uso:**
```bash
# Modo dry-run (apenas simular)
python src/pipeline/verify_and_clean_pmids.py --dry-run --prompt-type type2

# Processar todos os modelos
python src/pipeline/verify_and_clean_pmids.py --prompt-type type2

# Processar modelo específico
python src/pipeline/verify_and_clean_pmids.py --model qwen2.5-7b-instruct --prompt-type type2
```

### 2. `remove_duplicates_pmids.py`
Script focado apenas na remoção de duplicados.

**Funcionalidades:**
- Remove arquivos JSON duplicados (mesmo PMID)
- Mantém apenas o arquivo mais recente (baseado na data de modificação)
- Suporta processar um modelo específico ou todos

**Uso:**
```bash
# Modo dry-run
python src/pipeline/remove_duplicates_pmids.py --dry-run

# Remover duplicados de todos os modelos
python src/pipeline/remove_duplicates_pmids.py --prompt-type type2

# Remover duplicados de um modelo específico
python src/pipeline/remove_duplicates_pmids.py --model qwen2.5-7b-instruct --prompt-type type2

# Remover duplicados de uma estratégia específica
python src/pipeline/remove_duplicates_pmids.py --model qwen2.5-7b-instruct --strategy examples_1 --prompt-type type2
```

### 3. `check_missing_pmids.py`
Script para verificar PMIDs faltantes.

**Funcionalidades:**
- Verifica PMIDs faltantes para um modelo específico ou todos
- Dois modos: usando arquivos `pmids_faltantes_*.txt` (rápido) ou verificação direta dos JSONs (preciso)
- Mostra resumo por modelo e por estratégia

**Uso:**
```bash
# Verificar todos os modelos (usando arquivos já gerados - rápido)
python src/pipeline/check_missing_pmids.py --use-files --prompt-type type2

# Verificar um modelo específico
python src/pipeline/check_missing_pmids.py --model yi-1.5-9b-chat --use-files

# Verificar uma estratégia específica (verificação direta dos JSONs)
python src/pipeline/check_missing_pmids.py --model qwen2.5-7b-instruct --strategy examples_2

# Modo detalhado (mostra todos os PMIDs faltantes)
python src/pipeline/check_missing_pmids.py --model yi-1.5-9b-chat --detailed --use-files
```

## Opções Comuns

Todos os scripts suportam as seguintes opções:

- `--model`: Nome do modelo específico (padrão: processa todos)
- `--prompt-type`: Tipo de prompt (type1 ou type2, padrão: type2)
- `--strategy`: Estratégia específica (padrão: processa todas)
- `--dry-run`: Apenas simular, não fazer alterações (quando aplicável)
- `--base-dir`: Diretório base do projeto (padrão: detecta automaticamente)

## Estrutura de Arquivos Gerados

Os scripts geram arquivos de faltantes em:
```
indicios_encontrados/{model}/type2/pmids_faltantes_{strategy}.txt
```

Cada arquivo contém uma lista de PMIDs faltantes (um por linha), ordenados numericamente.

## Fluxo Recomendado

1. **Verificar faltantes:**
   ```bash
   python src/pipeline/check_missing_pmids.py --use-files --prompt-type type2
   ```

2. **Remover duplicados (se necessário):**
   ```bash
   python src/pipeline/remove_duplicates_pmids.py --dry-run --prompt-type type2  # Verificar primeiro
   python src/pipeline/remove_duplicates_pmids.py --prompt-type type2  # Executar
   ```

3. **Verificação completa e limpeza:**
   ```bash
   python src/pipeline/verify_and_clean_pmids.py --dry-run --prompt-type type2  # Verificar primeiro
   python src/pipeline/verify_and_clean_pmids.py --prompt-type type2  # Executar
   ```

