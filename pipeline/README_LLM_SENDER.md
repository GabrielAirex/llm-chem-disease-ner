# LLM Sender - Processamento de CSV para API BioNER_llm

Script para processar o dataset CDR (`cdr_ner_dataset.csv`) e enviar textos para a API de extração de entidades biomédicas.

## Características

- ✅ **Modelos configuráveis**: Suporta diferentes modelos via argumentos ou `config.yaml`
- ✅ **Processamento parcial**: Permite processar apenas uma parte do dataset (`--start` e `--end`)
- ✅ **Múltiplas estratégias**: Zero-shot, Few-shot, Chain-of-thought
- ✅ **Salvamento automático**: Salva resultados individuais e resumo em JSON
- ✅ **Estatísticas detalhadas**: Mostra químicos, doenças e tempo de processamento

## Uso Básico

### Processar todo o dataset com modelo padrão

```bash
cd /home/airex/Desktop/projects/onconavegation/onconavegation-BioNER_llm
source venv/bin/activate
python3 src/preprocessing/llm_sender.py
```

O script usa automaticamente o modelo configurado em `config/config.yaml` (`model_name`).

### Processar apenas uma parte do dataset

```bash
# Processar apenas primeiros 10 textos (índices 0-9)
python3 src/preprocessing/llm_sender.py --start 0 --end 10

# Processar textos de 100 a 200
python3 src/preprocessing/llm_sender.py --start 100 --end 200
```

## Configuração de Modelos

### Usar modelo específico via argumento

```bash
# Usar modelo Llama-3.2-3B-Instruct
python3 src/preprocessing/llm_sender.py --model meta-llama/Llama-3.2-3B-Instruct

# Usar múltiplos modelos (se API suportar)
python3 src/preprocessing/llm_sender.py --model meta-llama/Llama-3.2-3B-Instruct --model mistralai/Mistral-7B-Instruct-v0.2
```

### Modelos disponíveis (vLLM)

- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `microsoft/Phi-3-mini-4k-instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `google/gemma-2-2b-it`

## Estratégias de Prompt

### Zero-shot (sem exemplos)

```bash
python3 src/preprocessing/llm_sender.py --prompt-strategy zero-shot
```

### Few-shot (com exemplos)

```bash
# Few-shot padrão (3 exemplos)
python3 src/preprocessing/llm_sender.py --prompt-strategy few-shot

# Few-shot com 5 exemplos
python3 src/preprocessing/llm_sender.py --prompt-strategy few-shot --num-examples 5
```

### Chain-of-thought

```bash
python3 src/preprocessing/llm_sender.py --prompt-strategy chain-of-thought
```

## Configurações Avançadas

### Ajustar temperatura e max_tokens

```bash
python3 src/preprocessing/llm_sender.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --temperature 0.2 \
  --max-tokens 2000
```

### Sem extrair posições (apenas texto)

```bash
python3 src/preprocessing/llm_sender.py --no-positions
```

### Ajustar delay entre requisições

```bash
# Delay de 0.5 segundos (mais rápido)
python3 src/preprocessing/llm_sender.py --delay 0.5

# Delay de 2 segundos (mais conservador)
python3 src/preprocessing/llm_sender.py --delay 2.0
```

### Não salvar resultados individuais (apenas resumo)

```bash
python3 src/preprocessing/llm_sender.py --no-save
```

### Especificar diretório de saída

```bash
python3 src/preprocessing/llm_sender.py --output-dir /caminho/para/resultados
```

## Exemplos Completos

### Processar 50 textos com Llama usando zero-shot

```bash
python3 src/preprocessing/llm_sender.py \
  --start 0 \
  --end 50 \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --prompt-strategy zero-shot \
  --delay 0.5
```

### Processar dataset completo com configurações customizadas

```bash
python3 src/preprocessing/llm_sender.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --prompt-strategy few-shot \
  --num-examples 5 \
  --temperature 0.1 \
  --max-tokens 1500 \
  --delay 1.0
```

### Processar com modelo diferente (Mistral)

```bash
python3 src/preprocessing/llm_sender.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --prompt-strategy few-shot \
  --start 0 \
  --end 100
```

## Saída

### Arquivos Gerados

1. **Resultados Individuais**: `indicios_encontrados/extraction_YYYYMMDD_HHMMSS_PMID_IDX.json`
   - Contém: texto original, entidades extraídas, metadados

2. **Resumo**: `indicios_encontrados/summary_YYYYMMDD_HHMMSS.json`
   - Contém: estatísticas gerais, configurações usadas, lista de resultados

### Formato dos Resultados

```json
{
  "pmid": "6794356",
  "original_text": "...",
  "extraction": {
    "chemicals": [
      {
        "text": "lithium carbonate",
        "start": 34,
        "end": 51,
        "type": "Chemical",
        "confidence": 0.95,
        "mesh_id": "D016651"
      }
    ],
    "diseases": [...]
  },
  "processing_time": 2.34,
  "timestamp": "20241112_132045",
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "prompt_strategy": "few-shot"
}
```

## Integração com config.yaml

O script lê automaticamente `config/config.yaml` e usa:
- `llm.model_name`: Modelo padrão (se `--model` não for especificado)
- `prompts.num_examples`: Número de exemplos (se `--num-examples` não for especificado)
- `prompts.max_text_length`: Comprimento máximo (se `--max-text-length` não for especificado)
- `prompts.use_positions`: Extrair posições (se `--no-positions` não for usado)
- `llm_defaults.temperature`: Temperatura padrão
- `llm_defaults.max_tokens`: Max tokens padrão

## Troubleshooting

### Erro de conexão com API

```bash
# Verificar se API está rodando
curl http://localhost:8001/health

# Se não estiver, iniciar API
python3 run_api.py
```

### Erro de modelo não encontrado

Certifique-se de que o modelo está disponível no vLLM:
```bash
curl http://localhost:8000/v1/models
```

### Processar em lotes grandes

Para datasets grandes, processe em partes:
```bash
# Lote 1: textos 0-500
python3 src/preprocessing/llm_sender.py --start 0 --end 500

# Lote 2: textos 500-1000
python3 src/preprocessing/llm_sender.py --start 500 --end 1000

# Lote 3: textos 1000-1500
python3 src/preprocessing/llm_sender.py --start 1000 --end 1500
```

## Ver todas as opções

```bash
python3 src/preprocessing/llm_sender.py --help
```

