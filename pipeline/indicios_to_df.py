"""
Script para converter indícios encontrados (JSONs) em dataframe no formato similar ao BC5CDR.
"""

import pandas as pd
from typing import Optional
import json
from pathlib import Path


def parse_indicios_to_df(indicios_dir: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Converte indícios encontrados (JSONs) em dataframe no formato similar ao PubTator.
    
    Args:
        indicios_dir: Diretório contendo os arquivos JSON de indícios
        output_path: Caminho para salvar CSV (opcional)
    
    Returns:
        DataFrame com a mesma estrutura do parse_pubtator_for_ner
    """
    indicios_path = Path(indicios_dir)
    
    if not indicios_path.exists():
        raise ValueError(f"Diretório não encontrado: {indicios_dir}")
    
    articles = []
    
    # Buscar todos os arquivos JSON de extração
    json_files = list(indicios_path.glob("extraction_*.json"))
    
    print(f"📚 Encontrados {len(json_files)} arquivos de indícios...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extrair informações básicas
            # Suportar tanto 'text' quanto 'original_text' (formato antigo)
            text = data.get('text', data.get('original_text', ''))
            text_hash = data.get('text_hash', '')
            timestamp = data.get('timestamp', '')
            pmid = data.get('pmid', '')  # Tentar obter pmid salvo
            
            # Usar pmid se disponível, senão usar text_hash, senão timestamp + hash
            article_id = pmid if pmid else (text_hash if text_hash else f"{timestamp}_{json_file.stem}")
            
            # Extrair entidades - suportar ambos os formatos (antigo e novo)
            chemicals = []
            diseases = []
            
            # Formato novo: entities é um objeto com chemicals e diseases
            if 'entities' in data and isinstance(data['entities'], dict):
                entities_dict = data['entities']
                chemicals_raw = entities_dict.get('chemicals', [])
                diseases_raw = entities_dict.get('diseases', [])
            # Formato antigo: entities é uma lista plana ou extraction contém chemicals/diseases
            elif 'extraction' in data:
                extraction = data['extraction']
                chemicals_raw = extraction.get('chemicals', [])
                diseases_raw = extraction.get('diseases', [])
            elif 'entities' in data and isinstance(data['entities'], list):
                # Formato muito antigo: entities é lista plana
                entities_list = data['entities']
                chemicals_raw = [e for e in entities_list if isinstance(e, dict) and e.get('type') == 'Chemical']
                diseases_raw = [e for e in entities_list if isinstance(e, dict) and e.get('type') == 'Disease']
            else:
                chemicals_raw = []
                diseases_raw = []
            
            # Processar chemicals
            for chem in chemicals_raw:
                if not isinstance(chem, dict):
                    continue
                # Remover confidence e mesh_id se existirem
                chem_data = {
                    'start': chem.get('start', 0),
                    'end': chem.get('end', 0),
                    'text': chem.get('text', ''),
                    'mesh_id': ''  # Sempre vazio, não temos mesh_id nos indícios
                }
                chemicals.append(chem_data)
            
            # Processar diseases
            for dis in diseases_raw:
                if not isinstance(dis, dict):
                    continue
                # Remover confidence e mesh_id se existirem
                dis_data = {
                    'start': dis.get('start', 0),
                    'end': dis.get('end', 0),
                    'text': dis.get('text', ''),
                    'mesh_id': ''  # Sempre vazio, não temos mesh_id nos indícios
                }
                diseases.append(dis_data)
            
            # Criar listas simples de textos
            chemicals_list = [c['text'] for c in chemicals]
            diseases_list = [d['text'] for d in diseases]
            
            # Tentar separar título e abstract (se o texto contiver "|t|" ou "|a|")
            # Caso contrário, usar o texto completo como abstract e título vazio
            title = ''
            abstract = text
            
            # Se o texto parece conter título e abstract separados
            if '|t|' in text or '|a|' in text:
                # Tentar extrair título e abstract se estiverem no formato PubTator
                if '|t|' in text:
                    parts = text.split('|t|', 1)
                    if len(parts) > 1:
                        title_abstract = parts[1]
                        if '|a|' in title_abstract:
                            title, abstract = title_abstract.split('|a|', 1)
                        else:
                            title = title_abstract
                            abstract = ''
                elif '|a|' in text:
                    parts = text.split('|a|', 1)
                    abstract = parts[1] if len(parts) > 1 else text
            else:
                # Se não há separação clara, tentar identificar título (primeira frase)
                # ou usar tudo como abstract
                sentences = text.split('. ')
                if len(sentences) > 1:
                    # Primeira frase como título potencial
                    title = sentences[0] + '.' if not sentences[0].endswith('.') else sentences[0]
                    abstract = '. '.join(sentences[1:])
                else:
                    title = ''
                    abstract = text
            
            # Criar registro do artigo no formato do cdr_ner_dataset
            article = {
                'pmid': article_id,
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
            
            articles.append(article)
            
        except Exception as e:
            print(f"⚠️ Erro ao processar {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    df = pd.DataFrame(articles)
    
    print(f"\n✅ Dataset criado a partir dos indícios!")
    print(f"   Total de artigos: {len(df)}")
    if len(df) > 0:
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
        print("Uso: python indicios_to_df.py <indicios_dir> [output_csv]")
        sys.exit(1)
    indicios_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "indicios_ner_dataset.csv"
    
    df = parse_indicios_to_df(indicios_dir, output_path=output_file)
    
    # Mostrar exemplos
    if len(df) > 0:
        print("\n📄 Exemplo de artigo dos indícios:")
        print(f"ID: {df.iloc[0]['pmid']}")
        print(f"Texto: {df.iloc[0]['text'][:200]}...")
        print(f"Químicos: {json.loads(df.iloc[0]['chemicals_list'])}")
        print(f"Doenças: {json.loads(df.iloc[0]['diseases_list'])}")

