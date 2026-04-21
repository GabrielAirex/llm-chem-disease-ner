"""
Script para enviar textos do CSV para a API BioNER_llm via POST
Suporta modelos configuráveis via argumentos ou arquivo de configuração
"""

import pandas as pd
import requests
from requests.exceptions import Timeout as RequestsTimeout
import json
import time
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carrega configuração do arquivo YAML ou usa padrões.
    
    Args:
        config_path: Caminho para config.yaml (None = usar padrão)
        
    Returns:
        Dicionário com configurações
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        return {}


def send_text_to_api(
    text: str,
    api_url: str = "http://localhost:8001",
    models: Optional[List[str]] = None,
    prompt_strategy: str = "few-shot",
    use_positions: bool = True,
    num_examples: Optional[int] = None,
    max_text_length: Optional[int] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    prompt_type: str = "type1",
    timeout: int = 120
) -> Optional[dict]:
    """
    Envia um texto para a API via POST.
    
    Args:
        text: Texto para processar
        api_url: URL da API (ex: http://localhost:8001)
        models: Lista de modelos (None = usar padrão da API)
        prompt_strategy: Estratégia de prompt (zero-shot, few-shot, chain-of-thought)
        use_positions: Extrair posições (start/end)
        num_examples: Número de exemplos para few-shot
        max_text_length: Comprimento máximo do texto
        llm_config: Configuração de LLM (temperature, max_tokens, etc)
        timeout: Timeout em segundos
        
    Returns:
        Resposta da API ou None
    """
    url = f"{api_url}/extract"
    
    payload = {
        "text": text,
        "prompt_strategy": prompt_strategy,
        "use_positions": use_positions,
        "prompt_type": prompt_type
    }
    
    # Adicionar modelos apenas se especificado
    if models:
        payload["models"] = models
    
    # Adicionar parâmetros opcionais
    if num_examples is not None:
        payload["num_examples"] = num_examples
    
    if max_text_length is not None:
        payload["max_text_length"] = max_text_length
    
    # Adicionar configuração de LLM
    if llm_config:
        payload["llm_config"] = llm_config
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Erro {response.status_code}: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        # Relançar timeout para ser capturado no process_csv
        print(f"⏱️  Timeout na requisição (limite: {timeout}s)")
        raise RequestsTimeout(f"Request timeout após {timeout}s")
    except requests.exceptions.ConnectionError:
        print(f"❌ Erro: Não foi possível conectar em {api_url}")
        print("   Verifique se a API está rodando (python run_api.py)")
        return None
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None


def process_csv(
    csv_path: str,
    api_url: str = "http://localhost:800",
    delay: float = 1.0,
    start_index: int = 0,
    end_index: Optional[int] = None,
    models: Optional[List[str]] = None,
    prompt_strategy: str = "few-shot",
    use_positions: bool = True,
    num_examples: Optional[int] = None,
    max_text_length: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    save_individual: bool = True,
    save_summary: bool = True,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    max_consecutive_timeouts: int = 2,
    prompt_type: str = "type1"
):
    """
    Lê CSV e envia cada texto para a API via POST.
    
    Args:
        csv_path: Caminho do CSV
        api_url: URL da API (padrão: http://localhost:8001)
        delay: Tempo de espera entre requisições (segundos)
        start_index: Índice inicial (para processar em partes)
        end_index: Índice final (None = até o fim)
        models: Lista de modelos a usar (None = usar padrão da API)
        prompt_strategy: Estratégia de prompt
        use_positions: Extrair posições (start/end)
        num_examples: Número de exemplos para few-shot
        max_text_length: Comprimento máximo do texto
        temperature: Temperatura para LLM
        max_tokens: Máximo de tokens
        save_individual: Salvar resultados individuais em JSON
        save_summary: Salvar resumo em JSON
        output_dir: Diretório para salvar resultados (padrão: indicios_encontrados)
        config_path: Caminho para config.yaml (None = usar padrão)
        resume_from_checkpoint: Caminho para arquivo de checkpoint para retomar execução
        max_consecutive_timeouts: Número máximo de timeouts consecutivos antes de parar (padrão: 2)
    """
    # Carregar configuração
    config = load_config(config_path)
    
    # Carregar checkpoint se fornecido
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        if checkpoint_path.exists():
            print(f"\n📂 Carregando checkpoint de: {checkpoint_path}")
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            # Começar do último índice processado (499) em vez do próximo (500)
            # Isso permite reprocessar o último item caso tenha havido algum problema
            start_index = checkpoint.get('last_processed_index', start_index)
            if start_index is None:
                # Fallback: usar next_index - 1 se last_processed_index não existir
                next_idx = checkpoint.get('next_index')
                if next_idx is not None:
                    start_index = next_idx - 1
                else:
                    start_index = 0
            print(f"   ✅ Retomando do índice: {start_index} (último processado no checkpoint)")
            print(f"   📊 Checkpoint: último={checkpoint.get('last_processed_index', 'N/A')}, próximo={checkpoint.get('next_index', 'N/A')}")
            
            # Também usar end_index do checkpoint se disponível e não foi especificado
            if end_index is None and 'end_index' in checkpoint:
                end_index = checkpoint.get('end_index')
                print(f"   📊 End index do checkpoint: {end_index}")
            
            # Usar configurações do checkpoint se não foram fornecidas
            if api_url == "http://localhost:800":  # valor padrão
                api_url = checkpoint.get('api_url', api_url)
            if models is None:
                models = checkpoint.get('models', models)
            if prompt_strategy == "few-shot":  # valor padrão
                prompt_strategy = checkpoint.get('prompt_strategy', prompt_strategy)
            if use_positions:  # padrão True
                use_positions = checkpoint.get('use_positions', use_positions)
            if num_examples is None:
                num_examples = checkpoint.get('num_examples', num_examples)
            if prompt_type == "type1":  # valor padrão
                prompt_type = checkpoint.get('prompt_type', prompt_type)
        else:
            print(f"⚠️  Checkpoint não encontrado: {checkpoint_path}")
            print(f"   Iniciando do início...")
    
    # Usar valores do config se não especificados
    if models is None:
        # Tentar obter modelo do config
        model_name = config.get('llm', {}).get('model_name')
        if model_name:
            models = [model_name]
        else:
            models = None  # API usará padrão
    
    if num_examples is None:
        num_examples = config.get('prompts', {}).get('num_examples')
    
    if max_text_length is None:
        max_text_length = config.get('prompts', {}).get('max_text_length')
    
    if use_positions is None:
        use_positions = config.get('prompts', {}).get('use_positions', True)
    
    # Preparar llm_config
    llm_config = None
    if temperature is not None or max_tokens is not None:
        llm_config = {}
        if temperature is not None:
            llm_config['temperature'] = temperature
        elif 'llm_defaults' in config:
            llm_config['temperature'] = config['llm_defaults'].get('temperature')
        
        if max_tokens is not None:
            llm_config['max_tokens'] = max_tokens
        elif 'llm_defaults' in config:
            llm_config['max_tokens'] = config['llm_defaults'].get('max_tokens')
    
    # Ler CSV
    print("=" * 80)
    print("🚀 Processando Dataset CDR")
    print("=" * 80)
    print(f"📁 CSV: {csv_path}")
    print(f"🌐 API: {api_url}")
    print(f"🤖 Modelos: {models or 'padrão da API'}")
    print(f"📝 Estratégia: {prompt_strategy}")
    print(f"📊 Índices: {start_index} até {end_index or 'fim'}")
    print("=" * 80)
    
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    if end_index is None:
        end_index = total_rows
    
    df_subset = df.iloc[start_index:end_index]
    print(f"\n📥 Total de textos no CSV: {total_rows}")
    print(f"📥 Processando: {len(df_subset)} textos (índices {start_index} a {end_index-1})")
    
    # Verificar se a API está rodando
    try:
        health_check = requests.get(f"{api_url}/health", timeout=5)
        if health_check.status_code == 200:
            print(f"✅ API está rodando em {api_url}")
        else:
            print(f"⚠️  API respondeu com status {health_check.status_code}")
    except:
        print(f"⚠️  Não foi possível verificar saúde da API")
    
    # Criar diretório de saída
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "indicios_encontrados"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Variáveis para controle de timeouts
    consecutive_timeouts = 0
    checkpoint_file = None
    
    # Criar arquivo de checkpoint se não for retomada
    if not resume_from_checkpoint and output_dir:
        checkpoint_file = Path(output_dir) / f"checkpoint_{timestamp}.json"
    
    # Processar cada texto
    results = []
    
    print("\n" + "=" * 80)
    print("🔄 Processando textos...")
    print("=" * 80)
    
    for idx, row in df_subset.iterrows():
        pmid = str(row.get('pmid', ''))
        text = str(row.get('text', ''))
        
        if not text or text == 'nan':
            print(f"\n[{idx - start_index + 1}/{len(df_subset)}] ⚠️  PMID {pmid}: Texto vazio, pulando...")
            results.append({
                'pmid': pmid,
                'success': False,
                'error': 'Texto vazio',
                'num_chemicals': 0,
                'num_diseases': 0,
                'num_entities': 0
            })
            continue
        
        print(f"\n[{idx - start_index + 1}/{len(df_subset)}] 📤 PMID {pmid} ({len(text)} chars)...")
        
        # Fazer POST para API
        start_time = time.time()
        try:
            result = send_text_to_api(
                text=text,
                api_url=api_url,
                models=models,
                prompt_strategy=prompt_strategy,
                use_positions=use_positions,
                num_examples=num_examples,
                max_text_length=max_text_length,
                llm_config=llm_config,
                prompt_type=prompt_type
            )
            processing_time = time.time() - start_time
            
            # Resetar contador de timeouts em caso de sucesso
            if result:
                consecutive_timeouts = 0
        except RequestsTimeout as e:
            # Tratamento específico de timeout
            processing_time = time.time() - start_time
            consecutive_timeouts += 1
            print(f"   ⏱️  TIMEOUT ({processing_time:.2f}s) - Requisição demorou mais que o limite")
            print(f"   ⚠️  Timeouts consecutivos: {consecutive_timeouts}/{max_consecutive_timeouts}")
            
            # Verificar se excedeu o limite de timeouts consecutivos
            if consecutive_timeouts >= max_consecutive_timeouts:
                print(f"\n{'='*80}")
                print(f"🛑 PARANDO EXECUÇÃO: {consecutive_timeouts} timeouts consecutivos detectados")
                print(f"{'='*80}")
                
                # Salvar checkpoint
                if checkpoint_file:
                    checkpoint_data = {
                        'timestamp': timestamp,
                        'csv_path': csv_path,
                        'api_url': api_url,
                        'models': models,
                        'prompt_strategy': prompt_strategy,
                        'use_positions': use_positions,
                        'num_examples': num_examples,
                        'prompt_type': prompt_type,
                        'start_index': start_index,
                        'end_index': end_index,
                        'last_processed_index': idx - 1,  # Último índice processado com sucesso
                        'next_index': idx,  # Próximo índice a processar
                        'consecutive_timeouts': consecutive_timeouts,
                        'total_processed': len(results),
                        'success_count': sum(1 for r in results if r.get('success', False)),
                        'error_count': len(results) - sum(1 for r in results if r.get('success', False))
                    }
                    
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"\n💾 Checkpoint salvo em: {checkpoint_file}")
                    print(f"\n📋 Para retomar a execução, use:")
                    print(f"   python llm_sender.py --resume-from-checkpoint {checkpoint_file}")
                    print(f"\n   Ou com run_multiple_examples.py:")
                    print(f"   python run_multiple_examples.py --resume-from-checkpoint {checkpoint_file}")
                
                # Adicionar resultado de erro
                results.append({
                    'pmid': pmid,
                    'success': False,
                    'error': f'Timeout (consecutivos: {consecutive_timeouts})',
                    'num_chemicals': 0,
                    'num_diseases': 0,
                    'num_entities': 0,
                    'processing_time': processing_time
                })
                
                # Salvar resumo parcial
                if save_summary and output_dir:
                    summary_file = Path(output_dir) / f"summary_partial_{timestamp}.json"
                    summary = {
                        'timestamp': timestamp,
                        'csv_path': csv_path,
                        'api_url': api_url,
                        'models': models,
                        'prompt_strategy': prompt_strategy,
                        'prompt_type': prompt_type,
                        'start_index': start_index,
                        'end_index': end_index,
                        'total_processed': len(results),
                        'success_count': sum(1 for r in results if r.get('success', False)),
                        'error_count': len(results) - sum(1 for r in results if r.get('success', False)),
                        'stopped_due_to_timeouts': True,
                        'consecutive_timeouts': consecutive_timeouts,
                        'last_processed_index': idx - 1,
                        'results': results
                    }
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    print(f"💾 Resumo parcial salvo em: {summary_file}")
                
                # Parar execução
                break
            
            # Se não excedeu, continuar tentando
            result = None
            results.append({
                'pmid': pmid,
                'success': False,
                'error': 'Timeout',
                'num_chemicals': 0,
                'num_diseases': 0,
                'num_entities': 0,
                'processing_time': processing_time
            })
        except Exception as e:
            # Resetar contador de timeouts para outros erros
            consecutive_timeouts = 0
            processing_time = time.time() - start_time
            print(f"   ❌ Erro: {e}")
            result = None
            results.append({
                'pmid': pmid,
                'success': False,
                'error': str(e),
                'num_chemicals': 0,
                'num_diseases': 0,
                'num_entities': 0,
                'processing_time': processing_time
            })
        
        if result:
            chemicals = result.get('chemicals', [])
            diseases = result.get('diseases', [])
            num_chemicals = len(chemicals)
            num_diseases = len(diseases)
            num_entities = num_chemicals + num_diseases
            
            print(f"   ✅ Sucesso! {num_chemicals} químicos, {num_diseases} doenças ({processing_time:.2f}s)")
            
            results.append({
                'pmid': pmid,
                'success': True,
                'num_chemicals': num_chemicals,
                'num_diseases': num_diseases,
                'num_entities': num_entities,
                'processing_time': processing_time
            })
            
            # Salvar resultado individual
            if save_individual:
                # Limpar entidades: remover confidence e mesh_id
                cleaned_chemicals = []
                for chem in chemicals:
                    cleaned_chem = {
                        'text': chem.get('text', ''),
                        'type': chem.get('type', 'Chemical')
                    }
                    # Incluir posições apenas se use_positions=True
                    if use_positions and 'start' in chem and 'end' in chem:
                        cleaned_chem['start'] = chem['start']
                        cleaned_chem['end'] = chem['end']
                    cleaned_chemicals.append(cleaned_chem)
                
                cleaned_diseases = []
                for dis in diseases:
                    cleaned_dis = {
                        'text': dis.get('text', ''),
                        'type': dis.get('type', 'Disease')
                    }
                    # Incluir posições apenas se use_positions=True
                    if use_positions and 'start' in dis and 'end' in dis:
                        cleaned_dis['start'] = dis['start']
                        cleaned_dis['end'] = dis['end']
                    cleaned_diseases.append(cleaned_dis)
                
                # Preparar dados para salvar
                entities_data = {
                    'chemicals': cleaned_chemicals,
                    'diseases': cleaned_diseases
                }
                
                # Preparar metadados
                save_data = {
                    'pmid': pmid,
                    'text': text,
                    'entities': entities_data,
                    'processing_time': processing_time,
                    'timestamp': timestamp,
                    'model': models[0] if models else 'default',
                    'prompt_strategy': prompt_strategy
                }
                
                # Incluir num_examples apenas se for few-shot
                if prompt_strategy == 'few-shot':
                    # Usar num_examples se fornecido, senão usar padrão do config
                    if num_examples is not None:
                        save_data['num_examples'] = num_examples
                    else:
                        # Usar valor do config já carregado
                        default_num_examples = config.get('prompts', {}).get('num_examples', 3)
                        save_data['num_examples'] = default_num_examples
                
                output_file = output_dir / f"extraction_{timestamp}_{pmid}_{idx}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
        elif result is None and (not results or 'error' not in results[-1].get('error', '')):
            # Só adicionar erro se não foi adicionado no tratamento de exceção
            print(f"   ❌ Falha ao processar")
            results.append({
                'pmid': pmid,
                'success': False,
                'error': 'Falha ao processar',
                'num_chemicals': 0,
                'num_diseases': 0,
                'num_entities': 0,
                'processing_time': processing_time
            })
        
        # Atualizar checkpoint periodicamente (a cada 10 itens)
        if checkpoint_file and (idx - start_index + 1) % 10 == 0:
            checkpoint_data = {
                'timestamp': timestamp,
                'csv_path': csv_path,
                'api_url': api_url,
                'models': models,
                'prompt_strategy': prompt_strategy,
                'use_positions': use_positions,
                'num_examples': num_examples,
                'prompt_type': prompt_type,
                'start_index': start_index,
                'end_index': end_index,
                'last_processed_index': idx,
                'next_index': idx + 1,
                'consecutive_timeouts': consecutive_timeouts,
                'total_processed': len(results),
                'success_count': sum(1 for r in results if r.get('success', False)),
                'error_count': len(results) - sum(1 for r in results if r.get('success', False))
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        # Aguardar entre requisições
        if delay > 0 and idx < (end_index - 1 if end_index else len(df_subset) - 1):
            time.sleep(delay)
    
    # Resumo
    success_count = sum(1 for r in results if r['success'])
    error_count = len(results) - success_count
    total_chemicals = sum(r.get('num_chemicals', 0) for r in results)
    total_diseases = sum(r.get('num_diseases', 0) for r in results)
    total_entities = sum(r.get('num_entities', 0) for r in results)
    avg_time = sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0
    
    print("\n" + "=" * 80)
    print("📊 RESUMO")
    print("=" * 80)
    print(f"   Total processado: {len(results)}")
    print(f"   ✅ Sucessos: {success_count}")
    print(f"   ❌ Erros: {error_count}")
    print(f"   🧪 Total de químicos: {total_chemicals}")
    print(f"   🏥 Total de doenças: {total_diseases}")
    print(f"   📦 Total de entidades: {total_entities}")
    print(f"   ⏱️  Tempo médio: {avg_time:.2f}s")
    print("=" * 80)
    
    # Salvar resumo
    if save_summary:
        summary_file = output_dir / f"summary_{timestamp}.json"
        summary = {
            'timestamp': timestamp,
            'csv_path': csv_path,
            'api_url': api_url,
            'models': models,
            'prompt_strategy': prompt_strategy,
            'prompt_type': prompt_type,
            'start_index': start_index,
            'end_index': end_index,
            'total_processed': len(df_subset),
            'success_count': success_count,
            'error_count': error_count,
            'total_chemicals': total_chemicals,
            'total_diseases': total_diseases,
            'total_entities': total_entities,
            'avg_processing_time': avg_time,
            'results': results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resumo salvo em: {summary_file}")
    
    # Remover checkpoint se completou com sucesso
    if checkpoint_file and checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"✅ Checkpoint removido (execução concluída)")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enviar textos do CSV para API BioNER_llm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  
  # Processar todo o dataset com modelo padrão
  python llm_sender.py
  
  # Processar apenas primeiros 10 textos
  python llm_sender.py --start 0 --end 10
  
  # Usar modelo específico
  python llm_sender.py --model meta-llama/Llama-3.2-3B-Instruct
  
  # Usar zero-shot
  python llm_sender.py --prompt-strategy zero-shot
  
  # Processar com configurações customizadas
  python llm_sender.py --model meta-llama/Llama-3.2-3B-Instruct --temperature 0.2 --max-tokens 2000
        """
    )
    
    parser.add_argument('--csv', type=str,
                       default=str(Path(__file__).parent.parent / "data" / "cdr_ner_dataset.csv"),
                       help='Caminho do CSV')
    parser.add_argument('--api-url', type=str, default="http://localhost:8001",
                       help='URL da API')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay entre requisições (segundos)')
    parser.add_argument('--start', type=int, default=0,
                       help='Índice inicial')
    parser.add_argument('--end', type=int, default=None,
                       help='Índice final')
    parser.add_argument('--model', type=str, action='append', dest='models',
                       help='Modelo a usar (pode ser usado múltiplas vezes para múltiplos modelos)')
    parser.add_argument('--prompt-strategy', type=str, default='few-shot',
                       choices=['zero-shot', 'few-shot', 'chain-of-thought'],
                       help='Estratégia de prompt')
    # use_positions sempre False - estratégia with_positions foi removida
    # parser.add_argument removido - sempre sem posições
    parser.add_argument('--num-examples', type=int, default=None,
                       help='Número de exemplos para few-shot')
    parser.add_argument('--max-text-length', type=int, default=None,
                       help='Comprimento máximo do texto')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Temperatura para LLM')
    parser.add_argument('--max-tokens', type=int, default=None,
                       help='Máximo de tokens')
    parser.add_argument('--no-save', action='store_false', dest='save_individual',
                       help='Não salvar resultados individuais')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Diretório para salvar resultados')
    parser.add_argument('--config', type=str, default=None,
                       help='Caminho para config.yaml (padrão: config/config.yaml)')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                       help='Caminho para arquivo de checkpoint para retomar execução')
    parser.add_argument('--max-consecutive-timeouts', type=int, default=2,
                       help='Número máximo de timeouts consecutivos antes de parar (padrão: 2)')
    
    args = parser.parse_args()
    
    process_csv(
        csv_path=args.csv,
        api_url=args.api_url,
        delay=args.delay,
        start_index=args.start,
        end_index=args.end,
        models=args.models,
        prompt_strategy=args.prompt_strategy,
        use_positions=False,  # Sempre False - with_positions foi removido
        num_examples=args.num_examples,
        max_text_length=args.max_text_length,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        save_individual=args.save_individual,
        output_dir=args.output_dir,
        config_path=args.config,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_consecutive_timeouts=args.max_consecutive_timeouts
    )

