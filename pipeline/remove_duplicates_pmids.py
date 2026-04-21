#!/usr/bin/env python3
"""
Script para remover arquivos JSON duplicados (mesmo PMID) dos indícios encontrados.
Mantém apenas o arquivo mais recente baseado na data de modificação.
"""

import json
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

def find_and_remove_duplicates(strategy_dir, dry_run=False):
    """
    Encontra e remove arquivos duplicados (mesmo PMID) em um diretório de estratégia.
    Mantém apenas o arquivo mais recente.
    Retorna: (num_duplicados_encontrados, num_arquivos_removidos, detalhes)
    """
    if not strategy_dir.exists():
        return 0, 0, {}
    
    pmid_to_files = defaultdict(list)
    json_files = list(strategy_dir.glob("extraction_*.json"))
    
    # Agrupar arquivos por PMID
    for json_file in json_files:
        pmid = get_pmid_from_json(json_file)
        if pmid:
            mtime = json_file.stat().st_mtime
            pmid_to_files[pmid].append((mtime, json_file))
    
    # Identificar duplicados (PMID com mais de 1 arquivo)
    duplicates = {pmid: files for pmid, files in pmid_to_files.items() if len(files) > 1}
    
    if not duplicates:
        return 0, 0, {}
    
    removed_count = 0
    details = {}
    
    for pmid, files in duplicates.items():
        # Ordenar por data (mais antigo primeiro)
        files_sorted = sorted(files, key=lambda x: x[0])
        
        # Manter o último (mais recente), remover os anteriores
        files_to_remove = files_sorted[:-1]
        file_to_keep = files_sorted[-1]
        
        details[pmid] = {
            'total': len(files),
            'removed': len(files_to_remove),
            'kept': file_to_keep[1].name,
            'removed_files': [f[1].name for f in files_to_remove]
        }
        
        if not dry_run:
            for mtime, file_to_remove in files_to_remove:
                try:
                    file_to_remove.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"   ⚠️  Erro ao remover {file_to_remove.name}: {e}")
    
    return len(duplicates), removed_count, details

def process_model(model_dir, prompt_type='type2', dry_run=False, verbose=False):
    """
    Processa um modelo específico, removendo duplicados de todas as estratégias.
    """
    type_dir = model_dir / prompt_type
    
    if not type_dir.exists():
        if verbose:
            print(f"   ⚠️  Diretório não existe: {type_dir}")
        return 0, 0
    
    strategies = ['zero_shot', 'examples_1', 'examples_2', 'examples_4', 
                  'examples_8', 'examples_16', 'examples_32']
    
    total_duplicates = 0
    total_removed = 0
    
    for strategy in strategies:
        strategy_dir = type_dir / strategy
        
        if not strategy_dir.exists():
            continue
        
        num_duplicates, num_removed, details = find_and_remove_duplicates(
            strategy_dir, dry_run=dry_run
        )
        
        if num_duplicates > 0:
            total_duplicates += num_duplicates
            total_removed += num_removed
            
            if verbose:
                print(f"   📊 {strategy}: {num_duplicates} PMIDs duplicados, {num_removed} arquivos removidos")
                if details:
                    # Mostrar alguns exemplos
                    for i, (pmid, info) in enumerate(list(details.items())[:3]):
                        print(f"      - PMID {pmid}: mantido {info['kept']}, removidos {info['removed']} arquivo(s)")
                    if len(details) > 3:
                        print(f"      ... e mais {len(details) - 3} PMIDs duplicados")
    
    return total_duplicates, total_removed

def main():
    parser = argparse.ArgumentParser(
        description="Remove arquivos JSON duplicados (mesmo PMID) dos indícios encontrados"
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
        '--verbose',
        action='store_true',
        help='Mostrar detalhes de cada estratégia'
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
    
    print("=" * 80)
    print("🗑️  REMOÇÃO DE DUPLICADOS")
    print("=" * 80)
    
    if args.dry_run:
        print("⚠️  MODO DRY RUN - Nenhum arquivo será removido")
        print()
    
    # Determinar quais modelos processar
    if args.model:
        models_to_process = [args.model]
    else:
        models_to_process = [d.name for d in indicios_dir.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')]
    
    total_models_processed = 0
    total_duplicates_found = 0
    total_files_removed = 0
    
    # Processar cada modelo
    for model in sorted(models_to_process):
        model_dir = indicios_dir / model
        
        if not model_dir.exists():
            continue
        
        # Se strategy específica foi fornecida, processar apenas ela
        if args.strategy:
            type_dir = model_dir / args.prompt_type
            strategy_dir = type_dir / args.strategy
            
            if not strategy_dir.exists():
                if args.verbose:
                    print(f"   ⚠️  Diretório não existe: {strategy_dir}")
                continue
            
            print(f"\n📦 {model} / {args.strategy}")
            num_duplicates, num_removed, details = find_and_remove_duplicates(
                strategy_dir, dry_run=args.dry_run
            )
            
            if num_duplicates > 0:
                total_duplicates_found += num_duplicates
                total_files_removed += num_removed
                print(f"   ✅ {num_duplicates} PMIDs duplicados encontrados, {num_removed} arquivos removidos")
                
                if args.verbose and details:
                    for i, (pmid, info) in enumerate(list(details.items())[:5]):
                        print(f"      - PMID {pmid}: mantido {info['kept']}, removidos {info['removed']} arquivo(s)")
                    if len(details) > 5:
                        print(f"      ... e mais {len(details) - 5} PMIDs duplicados")
            else:
                print(f"   ✅ Nenhum duplicado encontrado")
            
            total_models_processed += 1
        else:
            # Processar todas as estratégias do modelo
            print(f"\n📦 {model}")
            num_duplicates, num_removed = process_model(
                model_dir, args.prompt_type, args.dry_run, args.verbose
            )
            
            if num_duplicates > 0:
                total_duplicates_found += num_duplicates
                total_files_removed += num_removed
                print(f"   ✅ Total: {num_duplicates} PMIDs duplicados, {num_removed} arquivos removidos")
            else:
                print(f"   ✅ Nenhum duplicado encontrado")
            
            total_models_processed += 1
    
    # Resumo final
    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL")
    print("=" * 80)
    print(f"Modelos processados: {total_models_processed}")
    print(f"PMIDs duplicados encontrados: {total_duplicates_found}")
    print(f"Arquivos removidos: {total_files_removed}")
    
    if args.dry_run:
        print("\n⚠️  Este foi um DRY RUN. Execute sem --dry-run para remover os arquivos.")
    else:
        print("\n✅ Remoção de duplicados concluída!")

if __name__ == "__main__":
    main()

