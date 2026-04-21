#!/usr/bin/env python3
"""
Script para criar datasets de validação a partir dos arquivos JSON extraídos
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

def find_text_positions(text, entity_text):
    """Encontra todas as posições de uma entidade no texto"""
    positions = []
    # Escapar caracteres especiais para regex
    pattern = re.escape(entity_text)
    # Buscar todas as ocorrências (case-insensitive)
    for match in re.finditer(pattern, text, re.IGNORECASE):
        positions.append({
            "start": match.start(),
            "end": match.end(),
            "text": match.group()
        })
    return positions

def convert_entities_to_format(entities, text):
    """Converte entidades do formato JSON para o formato do CSV"""
    formatted_entities = []
    for entity in entities:
        entity_text = entity.get("text", "")
        if not entity_text:
            continue
        
        # Encontrar posições no texto
        positions = find_text_positions(text, entity_text)
        
        for pos in positions:
            formatted_entities.append({
                "start": pos["start"],
                "end": pos["end"],
                "text": pos["text"],
                "mesh_id": ""  # Os JSONs não têm mesh_id
            })
    
    return formatted_entities

def process_json_directory(json_dir: Path, reference_csv: pd.DataFrame) -> pd.DataFrame:
    """Processa um diretório de JSONs e retorna um DataFrame"""
    json_files = sorted(json_dir.glob("extraction_*.json"))
    
    rows = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            pmid = str(data.get('pmid', ''))
            if not pmid:
                continue
            
            # Buscar informações do CSV de referência
            ref_row = reference_csv[reference_csv['pmid'] == pmid]
            
            if ref_row.empty:
                # Se não encontrar no CSV, usar apenas os dados do JSON
                text = data.get('text', '')
                title = ''
                abstract = ''
            else:
                text = ref_row.iloc[0]['text']
                title = ref_row.iloc[0]['title']
                abstract = ref_row.iloc[0]['abstract']
            
            # Extrair entidades
            entities = data.get('entities', {})
            chemicals = entities.get('chemicals', [])
            diseases = entities.get('diseases', [])
            
            # Converter para formato do CSV
            chemicals_formatted = convert_entities_to_format(chemicals, text)
            diseases_formatted = convert_entities_to_format(diseases, text)
            
            # Criar listas simples
            chemicals_list = [c.get('text', '') for c in chemicals if c.get('text')]
            diseases_list = [d.get('text', '') for d in diseases if d.get('text')]
            
            # Contar
            num_chemicals = len(chemicals_list)
            num_diseases = len(diseases_list)
            num_entities = num_chemicals + num_diseases
            
            # Converter para JSON string (formato do CSV)
            chemicals_json = json.dumps(chemicals_formatted, ensure_ascii=False)
            diseases_json = json.dumps(diseases_formatted, ensure_ascii=False)
            
            rows.append({
                'pmid': pmid,
                'text': text,
                'title': title,
                'abstract': abstract,
                'chemicals': chemicals_json,
                'diseases': diseases_json,
                'chemicals_list': json.dumps(chemicals_list, ensure_ascii=False),
                'diseases_list': json.dumps(diseases_list, ensure_ascii=False),
                'num_chemicals': num_chemicals,
                'num_diseases': num_diseases,
                'num_entities': num_entities
            })
        
        except Exception as e:
            print(f"⚠️  Erro ao processar {json_file.name}: {e}")
            continue
    
    return pd.DataFrame(rows)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cria datasets de validação a partir dos arquivos JSON extraídos"
    )
    parser.add_argument(
        '--reference-csv',
        type=str,
        default='cdr_ner_dataset.csv',
        help='CSV de referência com title e abstract (padrão: cdr_ner_dataset.csv)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='../../indicios_encontrados/llama3.2:3b',
        help='Diretório base com os JSONs (padrão: ../../indicios_encontrados/llama3.2:3b)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../../dataset',
        help='Diretório de saída (padrão: ../../dataset)'
    )
    
    args = parser.parse_args()
    
    # Caminhos
    script_dir = Path(__file__).parent
    reference_csv_path = script_dir / args.reference_csv
    base_dir = script_dir / args.base_dir
    output_dir = script_dir / args.output_dir
    
    # Extrair nome do modelo do diretório base
    # Ex: indicios_encontrados/gemma-1.1-2b-it -> gemma-1.1-2b-it
    # Ex: indicios_encontrados/llama3.2:3b -> llama3.2-3b
    model_name = base_dir.name.replace(':', '-').replace('_', '-')
    
    # Criar diretório de saída
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("📊 CRIANDO DATASETS DE VALIDAÇÃO")
    print("=" * 80)
    print(f"📁 Diretório base: {base_dir}")
    print(f"📁 Diretório de saída: {output_dir}")
    print(f"📄 CSV de referência: {reference_csv_path}")
    print()
    
    # Carregar CSV de referência
    print("📖 Carregando CSV de referência...")
    reference_df = pd.read_csv(reference_csv_path)
    reference_df['pmid'] = reference_df['pmid'].astype(str)
    print(f"   ✅ {len(reference_df)} artigos carregados")
    print()
    
    # Diretórios para processar
    directories = [
        ("zero_shot", "zero_shot"),
        ("examples_1", "examples_1"),
        ("examples_2", "examples_2"),
        ("examples_4", "examples_4"),
        ("examples_8", "examples_8"),
        ("examples_16", "examples_16"),
        ("examples_32", "examples_32"),
    ]
    
    results = []
    
    for dir_name, output_suffix in directories:
        json_dir = base_dir / dir_name
        
        if not json_dir.exists():
            print(f"⚠️  Diretório não encontrado: {json_dir}")
            continue
        
        print(f"📁 Processando: {dir_name}")
        print(f"   📂 {json_dir}")
        
        # Processar JSONs
        df = process_json_directory(json_dir, reference_df)
        
        if df.empty:
            print(f"   ⚠️  Nenhum dado encontrado")
            continue
        
        # Ordenar por PMID
        df = df.sort_values('pmid')
        
        # Nome do arquivo de saída
        output_filename = f"{model_name}_{output_suffix}.csv"
        output_path = output_dir / output_filename
        
        # Salvar CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"   ✅ {len(df)} artigos processados")
        print(f"   💾 Salvo em: {output_path}")
        
        # Estatísticas
        total_chemicals = df['num_chemicals'].sum()
        total_diseases = df['num_diseases'].sum()
        total_entities = df['num_entities'].sum()
        
        print(f"   📊 Estatísticas:")
        print(f"      - Chemicals: {total_chemicals}")
        print(f"      - Diseases: {total_diseases}")
        print(f"      - Total entities: {total_entities}")
        print()
        
        results.append({
            'dataset': output_filename,
            'articles': len(df),
            'chemicals': total_chemicals,
            'diseases': total_diseases,
            'entities': total_entities
        })
    
    # Resumo final
    print("=" * 80)
    print("📊 RESUMO")
    print("=" * 80)
    print(f"{'Dataset':<30} {'Artigos':<10} {'Chemicals':<12} {'Diseases':<12} {'Total':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['dataset']:<30} {r['articles']:<10} {r['chemicals']:<12} {r['diseases']:<12} {r['entities']:<10}")
    
    print("=" * 80)
    print(f"✅ Todos os datasets foram criados em: {output_dir}")

if __name__ == "__main__":
    main()

