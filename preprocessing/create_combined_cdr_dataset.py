#!/usr/bin/env python3
"""
Script para criar dataset combinado dos 3 conjuntos do CDR (Development, Test, Training)
Formato de saída: igual ao cdr_ner_dataset.csv
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

def parse_pubtator_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse arquivo PubTator.txt do CDR
    
    Formato:
    PMID|t|title
    PMID|a|abstract
    PMID	start	end	text	type	mesh_id
    
    Args:
        file_path: Caminho para arquivo PubTator.txt
        
    Returns:
        Lista de dicionários com os dados parseados
    """
    articles = []
    current_article = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                # Linha vazia = fim do artigo
                if current_article:
                    articles.append(current_article)
                    current_article = None
                continue
            
            if '|t|' in line:
                # Linha de título
                parts = line.split('|t|')
                pmid = parts[0]
                title = parts[1] if len(parts) > 1 else ''
                current_article = {
                    'pmid': pmid,
                    'title': title,
                    'abstract': '',
                    'chemicals': [],
                    'diseases': []
                }
            elif '|a|' in line:
                # Linha de abstract
                parts = line.split('|a|')
                abstract = parts[1] if len(parts) > 1 else ''
                if current_article:
                    current_article['abstract'] = abstract
            elif '\t' in line:
                # Linha de anotação
                parts = line.split('\t')
                if len(parts) >= 6 and current_article:
                    pmid, start, end, text, entity_type, mesh_id = parts[:6]
                    
                    entity = {
                        'start': int(start),
                        'end': int(end),
                        'text': text,
                        'mesh_id': mesh_id
                    }
                    
                    if entity_type.lower() == 'chemical':
                        current_article['chemicals'].append(entity)
                    elif entity_type.lower() == 'disease':
                        current_article['diseases'].append(entity)
    
    # Adicionar último artigo se houver
    if current_article:
        articles.append(current_article)
    
    return articles

def article_to_csv_row(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converte artigo parseado para formato CSV
    
    Args:
        article: Dicionário com dados do artigo
        
    Returns:
        Dicionário no formato do CSV
    """
    # Combinar título e abstract em text
    title = article.get('title', '')
    abstract = article.get('abstract', '')
    text = f"{title} {abstract}" if title and abstract else title or abstract
    
    chemicals = article.get('chemicals', [])
    diseases = article.get('diseases', [])
    
    # Criar listas de textos
    chemicals_list = [c['text'] for c in chemicals]
    diseases_list = [d['text'] for d in diseases]
    
    return {
        'pmid': article.get('pmid', ''),
        'text': text,
        'title': title,
        'abstract': abstract,
        'chemicals': json.dumps(chemicals),
        'diseases': json.dumps(diseases),
        'chemicals_list': json.dumps(chemicals_list),
        'diseases_list': json.dumps(diseases_list),
        'num_chemicals': len(chemicals),
        'num_diseases': len(diseases),
        'num_entities': len(chemicals) + len(diseases)
    }

def main():
    """Função principal"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Cria dataset combinado dos 3 splits do BC5CDR (dev + test + train)."
                    " Para gerar o gold de avaliação (test only), use text_to_df.py."
    )
    parser.add_argument(
        '--corpus-dir',
        type=str,
        required=True,
        help='Diretório com os arquivos PubTator do BC5CDR '
             '(ex: /path/to/CDR.Corpus.v010516)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "cdr_combined.csv"),
        help='Caminho do CSV de saída (padrão: data/cdr_combined.csv)'
    )
    args = parser.parse_args()

    print("🔄 Criando dataset combinado do CDR...")
    print("=" * 80)

    base_path = Path(args.corpus_dir)

    files = {
        'development': base_path / "CDR_DevelopmentSet.PubTator.txt",
        'test': base_path / "CDR_TestSet.PubTator.txt",
        'training': base_path / "CDR_TrainingSet.PubTator.txt"
    }
    
    # Verificar se arquivos existem
    for name, path in files.items():
        if not path.exists():
            print(f"❌ Arquivo não encontrado: {path}")
            return 1
        print(f"✅ Arquivo encontrado: {name} ({path.name})")
    
    print("\n" + "=" * 80)
    print("📥 Lendo e parseando arquivos...")
    print("=" * 80)
    
    all_articles = []
    
    for name, path in files.items():
        print(f"\n{name.upper()}:")
        articles = parse_pubtator_file(path)
        print(f"  Artigos: {len(articles)}")
        all_articles.extend(articles)
    
    print("\n" + "=" * 80)
    print(f"📊 Total de artigos combinados: {len(all_articles)}")
    print("=" * 80)
    
    # Converter para DataFrame
    print("\n🔄 Convertendo para DataFrame...")
    rows = [article_to_csv_row(article) for article in all_articles]
    df = pd.DataFrame(rows)
    
    # Estatísticas
    print(f"\n📈 Estatísticas:")
    print(f"  Total de artigos: {len(df)}")
    print(f"  Total de químicos: {df['num_chemicals'].sum()}")
    print(f"  Total de doenças: {df['num_diseases'].sum()}")
    print(f"  Total de entidades: {df['num_entities'].sum()}")
    print(f"  Média de entidades por artigo: {df['num_entities'].mean():.2f}")
    
    # Salvar CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset salvo em: {output_path}")
    print(f"   Linhas: {len(df)}")
    print(f"   Colunas: {len(df.columns)}")
    print("\n" + "=" * 80)
    
    # Mostrar primeiras linhas
    print("\n📋 Primeiras 3 linhas:")
    print(df[['pmid', 'num_chemicals', 'num_diseases', 'num_entities']].head(3))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

