import pandas as pd
from typing import List, Dict
import json

def parse_pubtator_for_ner(file_path: str) -> pd.DataFrame:
    """
    Parseia arquivo PubTator criando dataset para NER.
    
    Cada linha contém:
    - pmid: ID do artigo
    - text: Texto completo (título + abstract)
    - chemicals: Lista de químicos anotados
    - diseases: Lista de doenças anotadas
    - chemicals_list: Lista simples de textos de químicos (para validação)
    - diseases_list: Lista simples de textos de doenças (para validação)
    """
    
    articles = []
    current_pmid = None
    current_title = None
    current_abstract = None
    current_entities = []  # Lista de entidades do artigo atual
    
    def process_article():
        """Processa artigo completo quando termina"""
        nonlocal current_pmid, current_title, current_abstract, current_entities
        
        if not current_pmid or not current_title or not current_abstract:
            return
        
        # Concatenar título e abstract
        full_text = current_title + " " + current_abstract
        title_length = len(current_title) + 1  # +1 para o espaço
        
        # Separar entidades por tipo e ajustar posições
        chemicals = []
        diseases = []
        
        for entity in current_entities:
            entity_type = entity['type']
            start = entity['start']
            end = entity['end']
            text = entity['text']
            mesh_id = entity['mesh_id']
            
            # Posições no PubTator são absolutas (relativas ao início do título).
            # Como full_text = title + " " + abstract, entidades do abstract ficam
            # deslocadas em +1 pelo espaço separador. Entidades do título ficam iguais.
            # Nota: a avaliação usa texto (chemicals_list/diseases_list), não posições.
            if start < len(current_title):  # Entidade no título
                adjusted_start = start
                adjusted_end = end
            else:  # Entidade no abstract: +1 pelo espaço separador
                adjusted_start = start + 1
                adjusted_end = end + 1
            
            entity_data = {
                'start': adjusted_start,
                'end': adjusted_end,
                'text': text,
                'mesh_id': mesh_id
            }
            
            if entity_type == 'Chemical':
                chemicals.append(entity_data)
            elif entity_type == 'Disease':
                diseases.append(entity_data)
        
        # Criar listas simples de textos (útil para comparação)
        chemicals_list = [c['text'] for c in chemicals]
        diseases_list = [d['text'] for d in diseases]
        
        articles.append({
            'pmid': current_pmid,
            'text': full_text,
            'title': current_title,
            'abstract': current_abstract,
            'chemicals': json.dumps(chemicals),  # Salvar como JSON string
            'diseases': json.dumps(diseases),
            'chemicals_list': json.dumps(chemicals_list),
            'diseases_list': json.dumps(diseases_list),
            'num_chemicals': len(chemicals),
            'num_diseases': len(diseases),
            'num_entities': len(chemicals) + len(diseases)
        })
        
        # Reset
        current_pmid = None
        current_title = None
        current_abstract = None
        current_entities = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Linha vazia = fim do artigo
            if not line:
                process_article()
                continue
            
            # Título: PMID|t|Título
            if '|t|' in line:
                process_article()  # Processar artigo anterior se houver
                parts = line.split('|t|', 1)
                current_pmid = parts[0]
                current_title = parts[1] if len(parts) > 1 else ''
                current_entities = []
                continue
            
            # Abstract: PMID|a|Abstract
            if '|a|' in line:
                parts = line.split('|a|', 1)
                current_pmid = parts[0]
                current_abstract = parts[1] if len(parts) > 1 else ''
                continue
            
            # Entidades: PMID\tstart\tend\ttext\tType\tMeSH_ID
            # Ignorar linhas CID (relações)
            if '\t' in line and current_pmid:
                parts = line.split('\t')
                
                # Pular relações CID
                if len(parts) == 4 and parts[1] == 'CID':
                    continue
                
                # Processar entidade
                if len(parts) >= 6:
                    try:
                        current_entities.append({
                            'start': int(parts[1]),
                            'end': int(parts[2]),
                            'text': parts[3],
                            'type': parts[4],
                            'mesh_id': parts[5]
                        })
                    except (ValueError, IndexError):
                        continue
    
    # Processar último artigo
    process_article()
    
    return pd.DataFrame(articles)


def create_ner_dataset(file_path: str, output_path: str = None):
    """
    Cria dataset completo para NER.
    
    Args:
        file_path: Caminho do arquivo PubTator
        output_path: Caminho para salvar CSV (opcional)
    """
    print("📚 Parseando arquivo PubTator...")
    df = parse_pubtator_for_ner(file_path)
    
    print(f"\n✅ Dataset criado!")
    print(f"   Total de artigos: {len(df)}")
    print(f"   Total de químicos: {df['num_chemicals'].sum()}")
    print(f"   Total de doenças: {df['num_diseases'].sum()}")
    print(f"   Média de químicos por artigo: {df['num_chemicals'].mean():.2f}")
    print(f"   Média de doenças por artigo: {df['num_diseases'].mean():.2f}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n💾 Dataset salvo em: {output_path}")
    
    return df


# Exemplo de uso
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python text_to_df.py <pubtator_file> [output_csv]")
        print("  Ex: python text_to_df.py CDR_TestSet.PubTator.txt data/cdr_gold.csv")
        sys.exit(1)
    file_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "cdr_ner_dataset.csv"

    df = create_ner_dataset(file_path, output_path=output_csv)
    
    # Mostrar exemplos
    print("\n📄 Exemplo de artigo:")
    print(f"PMID: {df.iloc[0]['pmid']}")
    print(f"Texto: {df.iloc[0]['text'][:200]}...")
    print(f"Químicos: {json.loads(df.iloc[0]['chemicals_list'])}")
    print(f"Doenças: {json.loads(df.iloc[0]['diseases_list'])}")