#!/usr/bin/env python3
"""
Script para verificar PMIDs faltantes nos indícios encontrados.
Pode verificar um modelo específico ou todos os modelos.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
import sys

def get_pmid_from_json(json_file):
    """Extrai o PMID de um arquivo JSON."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pmid = data.get('pmid')
            if pmid:
                return str(pmid)
    except Exception as e:
        return None
    return None

def check_missing_from_files(model_dir, prompt_type='type2'):
    """
    Verifica PMIDs faltantes a partir dos arquivos pmids_faltantes_*.txt já gerados.
    Retorna: {strategy: [lista de PMIDs faltantes]}
    """
    type_dir = model_dir / prompt_type
    
    if not type_dir.exists():
        return {}
    
    missing_by_strategy = {}
    faltantes_files = list(type_dir.glob("pmids_faltantes_*.txt"))
    
    for faltantes_file in faltantes_files:
        # Extrair nome da estratégia do nome do arquivo
        strategy = faltantes_file.stem.replace("pmids_faltantes_", "")
        
        try:
            with open(faltantes_file, 'r', encoding='utf-8') as f:
                pmids = [line.strip() for line in f if line.strip()]
                if pmids:
                    missing_by_strategy[strategy] = pmids
        except Exception as e:
            print(f"   ⚠️  Erro ao ler {faltantes_file.name}: {e}")
    
    return missing_by_strategy

def check_missing_from_json(model_dir, gold_pmids, prompt_type='type2', strategy=None):
    """
    Verifica PMIDs faltantes diretamente dos arquivos JSON.
    Retorna: {strategy: [lista de PMIDs faltantes]}
    """
    type_dir = model_dir / prompt_type
    
    if not type_dir.exists():
        return {}
    
    strategies = ['zero_shot', 'examples_1', 'examples_2', 'examples_4', 
                  'examples_8', 'examples_16', 'examples_32']
    
    if strategy:
        strategies = [strategy]
    
    missing_by_strategy = {}
    
    for strat in strategies:
        strategy_dir = type_dir / strat
        
        if not strategy_dir.exists():
            continue
        
        # Coletar PMIDs presentes
        present_pmids = set()
        json_files = list(strategy_dir.glob("extraction_*.json"))
        
        for json_file in json_files:
            pmid = get_pmid_from_json(json_file)
            if pmid:
                present_pmids.add(pmid)
        
        # Identificar faltantes
        missing_pmids = gold_pmids - present_pmids
        missing_pmids_sorted = sorted(missing_pmids, key=lambda x: int(x) if x.isdigit() else 0)
        
        if missing_pmids_sorted:
            missing_by_strategy[strat] = missing_pmids_sorted
    
    return missing_by_strategy

def load_gold_pmids(gold_csv):
    """Carrega todos os PMIDs do gold standard."""
    if not gold_csv.exists():
        return None
    
    df_gold = pd.read_csv(gold_csv)
    return set(df_gold['pmid'].astype(str))

def print_summary(model_name, missing_by_strategy, use_files=False):
    """Imprime resumo dos faltantes para um modelo."""
    if not missing_by_strategy:
        print(f"   ✅ {model_name}: Nenhum faltante encontrado")
        return
    
    total_missing = sum(len(pmids) for pmids in missing_by_strategy.values())
    print(f"   ⚠️  {model_name}: {total_missing} PMIDs faltantes em {len(missing_by_strategy)} estratégia(s)")
    
    for strategy, pmids in sorted(missing_by_strategy.items()):
        source = "arquivo" if use_files else "verificação direta"
        print(f"      - {strategy}: {len(pmids)} faltantes ({source})")

def main():
    parser = argparse.ArgumentParser(
        description="Verifica PMIDs faltantes nos indícios encontrados"
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Nome do modelo específico (padrão: processa todos)'
    )
    parser.add_argument(
        '--prompt-type',
        type=str,
        default='type2',
        choices=['type1', 'type2'],
        help='Tipo de prompt (padrão: type2)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        help='Estratégia específica (padrão: processa todas)'
    )
    parser.add_argument(
        '--use-files',
        action='store_true',
        help='Usar arquivos pmids_faltantes_*.txt já gerados (mais rápido)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Mostrar lista completa de PMIDs faltantes'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Diretório base do projeto (padrão: detecta automaticamente)'
    )
    
    args = parser.parse_args()
    
    # Detectar diretório base
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(__file__).parent.parent
    
    indicios_dir = base_dir / "indicios_encontrados"
    gold_csv = base_dir / "src" / "preprocessing" / "cdr_gold.csv"
    
    print("=" * 80)
    print("🔍 VERIFICAÇÃO DE PMIDs FALTANTES")
    print("=" * 80)
    print()
    
    # Carregar gold standard (se não usar arquivos)
    gold_pmids = None
    if not args.use_files:
        print("📄 Carregando gold standard...")
        gold_pmids = load_gold_pmids(gold_csv)
        if gold_pmids is None:
            print(f"❌ Erro ao carregar gold standard: {gold_csv}")
            sys.exit(1)
        print(f"   ✅ Total de PMIDs no gold standard: {len(gold_pmids)}")
        print()
    
    # Determinar quais modelos processar
    if args.model:
        models_to_process = [args.model]
    else:
        models_to_process = [d.name for d in indicios_dir.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')]
    
    all_missing = defaultdict(dict)
    models_with_missing = 0
    total_missing_count = 0
    
    # Processar cada modelo
    for model in sorted(models_to_process):
        model_dir = indicios_dir / model
        
        if not model_dir.exists():
            continue
        
        if args.use_files:
            missing_by_strategy = check_missing_from_files(model_dir, args.prompt_type)
        else:
            missing_by_strategy = check_missing_from_json(
                model_dir, gold_pmids, args.prompt_type, args.strategy
            )
        
        if missing_by_strategy:
            all_missing[model] = missing_by_strategy
            models_with_missing += 1
            total_missing_count += sum(len(pmids) for pmids in missing_by_strategy.values())
    
    # Mostrar resultados
    if not all_missing:
        print("✅ Nenhum PMID faltante encontrado em nenhum modelo!")
        return
    
    print(f"\n📊 RESULTADOS")
    print("=" * 80)
    print(f"Modelos com faltantes: {models_with_missing} de {len(models_to_process)}")
    print(f"Total de PMIDs faltantes: {total_missing_count}")
    print()
    
    # Mostrar detalhes por modelo
    for model, missing_by_strategy in sorted(all_missing.items()):
        print(f"\n📦 {model}")
        print("-" * 80)
        
        for strategy, pmids in sorted(missing_by_strategy.items()):
            print(f"   📊 {strategy}: {len(pmids)} PMIDs faltantes")
            
            if args.detailed:
                print(f"      PMIDs faltantes:")
                # Mostrar em colunas para melhor visualização
                for i in range(0, len(pmids), 10):
                    chunk = pmids[i:i+10]
                    print(f"      {', '.join(chunk)}")
                    if i + 10 < len(pmids):
                        print()
            else:
                # Mostrar apenas primeiros e últimos
                if len(pmids) <= 10:
                    print(f"      {', '.join(pmids)}")
                else:
                    print(f"      Primeiros 5: {', '.join(pmids[:5])}")
                    print(f"      Últimos 5: {', '.join(pmids[-5:])}")
                    print(f"      ... e mais {len(pmids) - 10} PMIDs")
    
    # Resumo por estratégia
    print("\n" + "=" * 80)
    print("📊 RESUMO POR ESTRATÉGIA")
    print("=" * 80)
    
    strategy_summary = defaultdict(int)
    for model, missing_by_strategy in all_missing.items():
        for strategy, pmids in missing_by_strategy.items():
            strategy_summary[strategy] += len(pmids)
    
    for strategy in sorted(strategy_summary.keys()):
        count = strategy_summary[strategy]
        models_count = sum(1 for m in all_missing.values() if strategy in m)
        print(f"   {strategy}: {count} faltantes em {models_count} modelo(s)")
    
    # Resumo final
    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL")
    print("=" * 80)
    print(f"Total de modelos verificados: {len(models_to_process)}")
    print(f"Modelos com faltantes: {models_with_missing}")
    print(f"Total de PMIDs faltantes: {total_missing_count}")
    print(f"Estratégias afetadas: {len(strategy_summary)}")
    
    if args.use_files:
        print("\n💡 Dica: Use sem --use-files para verificação direta dos JSONs")

if __name__ == "__main__":
    main()

