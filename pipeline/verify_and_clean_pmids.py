#!/usr/bin/env python3
"""
Script para verificar se todos os indícios encontrados têm a lista de PMIDs dos 1500,
verificar duplicados e remover os mais antigos.
Gera arquivo com PMIDs faltantes para cada modelo/strategy.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
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

def find_duplicates(strategy_dir):
    """
    Encontra arquivos duplicados (mesmo PMID) e retorna um dicionário:
    {pmid: [lista de arquivos ordenados por data (mais antigo primeiro)]}
    """
    pmid_to_files = defaultdict(list)
    
    json_files = list(strategy_dir.glob("extraction_*.json"))
    
    for json_file in json_files:
        pmid = get_pmid_from_json(json_file)
        if pmid:
            # Obter data de modificação do arquivo
            mtime = json_file.stat().st_mtime
            pmid_to_files[pmid].append((mtime, json_file))
    
    # Ordenar por data (mais antigo primeiro)
    duplicates = {}
    for pmid, files in pmid_to_files.items():
        if len(files) > 1:
            files_sorted = sorted(files, key=lambda x: x[0])  # Ordenar por mtime
            duplicates[pmid] = [f[1] for f in files_sorted]  # Apenas os paths
    
    return duplicates

def remove_duplicates(strategy_dir, dry_run=False):
    """
    Remove duplicados, mantendo apenas o arquivo mais recente.
    Retorna: (num_removidos, duplicados_info)
    """
    duplicates = find_duplicates(strategy_dir)
    
    if not duplicates:
        return 0, {}
    
    removed_count = 0
    duplicates_info = {}
    
    for pmid, files in duplicates.items():
        # Manter o último (mais recente), remover os anteriores
        files_to_remove = files[:-1]
        files_to_keep = files[-1]
        
        duplicates_info[pmid] = {
            'total': len(files),
            'removed': len(files_to_remove),
            'kept': files_to_keep.name,
            'removed_files': [f.name for f in files_to_remove]
        }
        
        if not dry_run:
            for file_to_remove in files_to_remove:
                try:
                    file_to_remove.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"   ⚠️  Erro ao remover {file_to_remove.name}: {e}")
    
    return removed_count, duplicates_info

def verify_pmids_for_strategy(strategy_dir, gold_pmids, strategy_name, dry_run=False):
    """
    Verifica PMIDs para uma estratégia específica.
    Retorna: (present_pmids, missing_pmids, duplicates_info)
    """
    print(f"\n{'='*80}")
    print(f"📊 Verificando: {strategy_name}")
    print(f"{'='*80}")
    
    if not strategy_dir.exists():
        print(f"   ⚠️  Diretório não existe: {strategy_dir}")
        return set(), gold_pmids, {}
    
    # 1. Encontrar e remover duplicados
    print("   🔍 Procurando duplicados...")
    removed_count, duplicates_info = remove_duplicates(strategy_dir, dry_run=dry_run)
    
    if duplicates_info:
        print(f"   ⚠️  Encontrados {len(duplicates_info)} PMIDs duplicados")
        if not dry_run:
            print(f"   🗑️  Removidos {removed_count} arquivos duplicados (mantidos os mais recentes)")
        else:
            print(f"   🔍 [DRY RUN] Seriam removidos {removed_count} arquivos duplicados")
    else:
        print("   ✅ Nenhum duplicado encontrado")
    
    # 2. Coletar PMIDs presentes
    print("   📚 Coletando PMIDs presentes...")
    present_pmids = set()
    json_files = list(strategy_dir.glob("extraction_*.json"))
    
    for json_file in json_files:
        pmid = get_pmid_from_json(json_file)
        if pmid:
            present_pmids.add(pmid)
    
    print(f"   ✅ PMIDs encontrados: {len(present_pmids)}")
    
    # 3. Identificar faltantes
    missing_pmids = gold_pmids - present_pmids
    missing_pmids_sorted = sorted(missing_pmids, key=lambda x: int(x) if x.isdigit() else 0)
    
    print(f"   📊 Total esperado (gold): {len(gold_pmids)}")
    print(f"   ✅ Presentes: {len(present_pmids)}")
    print(f"   ❌ Faltantes: {len(missing_pmids)}")
    
    return present_pmids, missing_pmids_sorted, duplicates_info

def main():
    parser = argparse.ArgumentParser(
        description="Verifica PMIDs faltantes e remove duplicados dos indícios encontrados"
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
        '--dry-run',
        action='store_true',
        help='Apenas simular, não remover arquivos'
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
    
    # Caminhos
    gold_csv = base_dir / "src" / "preprocessing" / "cdr_gold.csv"
    indicios_dir = base_dir / "indicios_encontrados"
    
    print("=" * 80)
    print("🔍 VERIFICAÇÃO DE PMIDs E REMOÇÃO DE DUPLICADOS")
    print("=" * 80)
    
    if args.dry_run:
        print("⚠️  MODO DRY RUN - Nenhum arquivo será removido")
        print()
    
    # 1. Ler todos os PMIDs do gold standard
    print("📄 Lendo CSV gold standard...")
    if not gold_csv.exists():
        print(f"❌ Arquivo gold standard não encontrado: {gold_csv}")
        sys.exit(1)
    
    df_gold = pd.read_csv(gold_csv)
    all_pmids = set(df_gold['pmid'].astype(str))
    print(f"   ✅ Total de PMIDs no gold standard: {len(all_pmids)}")
    print()
    
    # 2. Determinar quais modelos/estratégias processar
    if args.model:
        models_to_process = [args.model]
    else:
        # Processar todos os modelos
        models_to_process = [d.name for d in indicios_dir.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')]
    
    strategies = ['zero_shot', 'examples_1', 'examples_2', 'examples_4', 
                  'examples_8', 'examples_16', 'examples_32']
    
    if args.strategy:
        strategies = [args.strategy]
    
    # 3. Processar cada modelo/estratégia
    total_duplicates_removed = 0
    total_missing = 0
    
    for model in sorted(models_to_process):
        model_dir = indicios_dir / model / args.prompt_type
        
        if not model_dir.exists():
            continue
        
        print(f"\n{'#'*80}")
        print(f"📦 MODELO: {model}")
        print(f"{'#'*80}")
        
        for strategy in strategies:
            strategy_dir = model_dir / strategy
            
            if not strategy_dir.exists():
                continue
            
            # Verificar PMIDs
            present_pmids, missing_pmids, duplicates_info = verify_pmids_for_strategy(
                strategy_dir, all_pmids, f"{model}/{strategy}", args.dry_run
            )
            
            # Salvar lista de faltantes
            if missing_pmids:
                output_file = model_dir / f"pmids_faltantes_{strategy}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for pmid in missing_pmids:
                        f.write(f"{pmid}\n")
                print(f"   💾 Lista de faltantes salva em: {output_file}")
                total_missing += len(missing_pmids)
            
            # Contar duplicados removidos
            if duplicates_info:
                for pmid, info in duplicates_info.items():
                    total_duplicates_removed += info['removed']
    
    # Resumo final
    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL")
    print("=" * 80)
    print(f"Total de duplicados removidos: {total_duplicates_removed}")
    print(f"Total de PMIDs faltantes encontrados: {total_missing}")
    
    if args.dry_run:
        print("\n⚠️  Este foi um DRY RUN. Execute sem --dry-run para aplicar as mudanças.")
    else:
        print("\n✅ Verificação e limpeza concluídas!")

if __name__ == "__main__":
    main()

