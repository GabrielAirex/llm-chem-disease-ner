#!/usr/bin/env python3
"""
Script para executar llm_sender com diferentes números de exemplos.
Executa com: 1, 2, 4, 6, 16, 32 exemplos e salva resultados em diretórios separados.
"""

import sys
from pathlib import Path
from llm_sender import process_csv
import argparse
import atexit
from api_launcher import APILauncher


def main():
    parser = argparse.ArgumentParser(
        description="Executa llm_sender com diferentes números de exemplos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python run_multiple_examples.py
  python run_multiple_examples.py --csv dataset.csv --start 0 --end 100
  python run_multiple_examples.py --examples 1 2 4
  python run_multiple_examples.py --run-all
  python run_multiple_examples.py --run-all --examples 2 4 8 16 32
  python run_multiple_examples.py --model google/gemma-1.1-2b-it --run-all
  python run_multiple_examples.py --model meta-llama/Llama-3.2-3B-Instruct --api-url http://localhost:8000 --run-all
  
  # Usar modelo específico (a API direciona automaticamente para a porta correta via config.yaml)
  python run_multiple_examples.py --model internlm/internlm2_5-1_8b-chat --run-all --prompt-type type2
  python run_multiple_examples.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --run-all --prompt-type type2
  
  # Usar ambos os modelos simultaneamente
  python run_multiple_examples.py --use-both --run-all --prompt-type type2
  python run_multiple_examples.py --models "HuggingFaceTB/SmolLM2-1.7B-Instruct" "internlm/internlm2_5-1_8b-chat" --run-all --prompt-type type2
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "cdr_ner_dataset.csv"),
        help='Caminho do CSV'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8001',
        help='URL da API BioNER_llm (padrão: http://localhost:8001). A API direciona automaticamente para as portas corretas dos modelos baseado no config.yaml'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-3.2-3B-Instruct',
        help='Modelo a usar (ex: google/gemma-1.1-2b-it). Será usado para criar estrutura de pastas: indicios_encontrados/nome_do_modelo/)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Lista de modelos para usar simultaneamente (ex: --models "HuggingFaceTB/SmolLM2-1.7B-Instruct" "internlm/internlm2_5-1_8b-chat"). Se especificado, sobrescreve --model'
    )
    parser.add_argument(
        '--use-both',
        action='store_true',
        help='Usa ambos os modelos simultaneamente (SmolLM2 na porta 8002 e InternLM2.5 na porta 8003). Equivale a --models "HuggingFaceTB/SmolLM2-1.7B-Instruct" "internlm/internlm2_5-1_8b-chat"'
    )
    parser.add_argument(
        '--prompt-strategy',
        type=str,
        default='few-shot',
        choices=['zero-shot', 'few-shot', 'chain-of-thought'],
        help='Estratégia de prompt'
    )
    # use_positions sempre False - estratégia with_positions foi removida
    # Não há mais opção --use-positions ou --no-positions
    parser.add_argument(
        '--examples',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8, 16, 32],
        help='Números de exemplos a testar (padrão: 1 2 4 8 16 32)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Índice inicial'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Índice final'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay entre requisições (segundos)'
    )
    parser.add_argument(
        '--base-output-dir',
        type=str,
        default=None,
        help='Diretório base para salvar resultados (padrão: indicios_encontrados)'
    )
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help='Caminho para arquivo de checkpoint para retomar execução'
    )
    parser.add_argument(
        '--max-consecutive-timeouts',
        type=int,
        default=2,
        help='Número máximo de timeouts consecutivos antes de parar (padrão: 2)'
    )
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Executa zero-shot e todos os few-shots em sequência (ignora --prompt-strategy)'
    )
    parser.add_argument(
        '--prompt-type',
        type=str,
        default='type1',
        choices=['type1', 'type2'],
        help='Tipo de prompt a usar (type1: mais restritivo, type2: mais inclusivo). Padrão: type1'
    )
    parser.add_argument(
        '--auto-launch-api',
        action='store_true',
        default=True,
        help='Iniciar automaticamente uma instância da API em thread separada (padrão: True)'
    )
    parser.add_argument(
        '--no-auto-launch-api',
        dest='auto_launch_api',
        action='store_false',
        help='Não iniciar API automaticamente (usar API externa via --api-url)'
    )
    
    args = parser.parse_args()
    
    # Processar --use-both (define ambos os modelos)
    if args.use_both:
        args.models = ["HuggingFaceTB/SmolLM2-1.7B-Instruct", "internlm/internlm2_5-1_8b-chat"]
    
    # Determinar modelos a usar
    if args.models:
        models_to_use = args.models
        # Para nome de diretório, usar o primeiro modelo ou criar nome combinado
        if len(models_to_use) == 1:
            model_name = models_to_use[0].split('/')[-1].lower().replace('_', '-')
        else:
            # Múltiplos modelos: criar nome combinado
            model_names = [m.split('/')[-1].lower().replace('_', '-') for m in models_to_use]
            model_name = "_and_".join(model_names)
    else:
        models_to_use = [args.model]
        # Extrair nome curto do modelo para criar estrutura de pastas
        # Ex: "google/gemma-1.1-2b-it" -> "gemma-1.1-2b-it"
        # Ex: "meta-llama/Llama-3.2-3B-Instruct" -> "llama-3.2-3b-instruct"
        model_name = args.model.split('/')[-1].lower().replace('_', '-')
    
    # Determinar diretório base de saída
    if args.base_output_dir is None:
        base_indicios_dir = Path(__file__).parent.parent / "indicios_encontrados"
    else:
        base_indicios_dir = Path(args.base_output_dir)
    
    # Criar estrutura: indicios_encontrados/nome_do_modelo/prompt_type/
    base_output_dir = base_indicios_dir / model_name / args.prompt_type
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Iniciar instância da API em thread separada para cada modelo (se habilitado)
    # Cada execução terá sua própria API rodando em uma porta única
    api_launchers = []
    api_urls = {}
    
    if args.auto_launch_api:
        print("=" * 80)
        print("🚀 Iniciando instâncias da API para cada modelo")
        print("=" * 80)
        
        for model in models_to_use:
            try:
                launcher = APILauncher(model_name=model, base_port=9000)
                launcher.start(wait_ready=True, timeout=30)
                api_launchers.append(launcher)
                api_urls[model] = launcher.api_url
                print(f"✅ API iniciada para {model} em {launcher.api_url}")
                # Pequeno delay para garantir que a API está totalmente pronta
                import time
                time.sleep(2)
            except Exception as e:
                print(f"⚠️ Erro ao iniciar API para {model}: {e}")
                print(f"   Continuando com API externa: {args.api_url}")
                import traceback
                traceback.print_exc()
        
        # Registrar cleanup ao sair
        def cleanup_apis():
            for launcher in api_launchers:
                try:
                    launcher.stop()
                except:
                    pass
        
        atexit.register(cleanup_apis)
        
        # Usar a URL da API local do primeiro modelo (ou criar uma URL combinada)
        if api_urls:
            if len(api_urls) == 1:
                args.api_url = list(api_urls.values())[0]
            else:
                # Para múltiplos modelos, usar a primeira API (a API já direciona para portas corretas)
                args.api_url = list(api_urls.values())[0]
            
            print(f"🌐 Usando API local: {args.api_url}")
        else:
            print(f"⚠️ Nenhuma API local iniciada, usando API externa: {args.api_url}")
        
        print("=" * 80)
        print()
    else:
        print(f"🌐 Usando API externa: {args.api_url}")
        print()
    
    # Se --run-all estiver ativo, executar zero-shot e few-shot
    if args.run_all:
        # Preparar lista de execuções: zero-shot primeiro, depois few-shots
        executions = [
            ("zero-shot", None, "zero_shot"),
        ]
        # Adicionar few-shots
        for num_examples in args.examples:
            executions.append(("few-shot", num_examples, f"examples_{num_examples}"))
        
        print("=" * 80)
        print("🚀 Executando TODAS as estratégias: zero-shot + few-shots")
        print("=" * 80)
        print(f"📄 CSV: {args.csv}")
        print(f"🌐 API: {args.api_url}")
        if len(models_to_use) > 1:
            print(f"🤖 Modelos: {', '.join(models_to_use)}")
        else:
            print(f"🤖 Modelo: {models_to_use[0]}")
        print(f"📝 Tipo de prompt: {args.prompt_type}")
        print(f"📁 Diretório de saída: {base_output_dir}")
        print(f"📍 Posições: False (sempre desativado)")
        print(f"🔢 Few-shot exemplos: {args.examples}")
        print(f"📊 Índices: {args.start} até {args.end if args.end else 'fim'}")
        print("=" * 80)
        print()
        
        results = []
        for strategy, num_examples, output_dir_name in executions:
            print()
            print("━" * 80)
            if strategy == "zero-shot":
                print("📊 Executando zero-shot (sem exemplos)")
            else:
                print(f"📊 Executando few-shot com {num_examples} exemplo(s)")
            print("━" * 80)
            
            output_dir = base_output_dir / output_dir_name
            
            try:
                process_csv(
                    csv_path=args.csv,
                    api_url=args.api_url,
                    delay=args.delay,
                    start_index=args.start,
                    end_index=args.end,
                    models=models_to_use,
                    prompt_strategy=strategy,
                    use_positions=False,  # Sempre False - with_positions foi removido
                    num_examples=num_examples,  # None para zero-shot
                    save_individual=True,
                    save_summary=True,
                    output_dir=str(output_dir),
                    resume_from_checkpoint=args.resume_from_checkpoint,
                    max_consecutive_timeouts=args.max_consecutive_timeouts,
                    prompt_type=args.prompt_type
                )
                if strategy == "zero-shot":
                    results.append(("zero-shot", "✅ Sucesso"))
                    print("✅ Concluído com sucesso: zero-shot")
                else:
                    results.append((f"{num_examples} exemplos", "✅ Sucesso"))
                    print(f"✅ Concluído com sucesso: {num_examples} exemplo(s)")
            except Exception as e:
                if strategy == "zero-shot":
                    results.append(("zero-shot", f"❌ Erro: {e}"))
                    print(f"❌ Erro ao executar zero-shot: {e}")
                else:
                    results.append((f"{num_examples} exemplos", f"❌ Erro: {e}"))
                    print(f"❌ Erro ao executar com {num_examples} exemplo(s): {e}")
                import traceback
                traceback.print_exc()
        
        # Resumo final
        print()
        print("=" * 80)
        print("📊 RESUMO DAS EXECUÇÕES")
        print("=" * 80)
        for identifier, status in results:
            print(f"   {identifier}: {status}")
        print("=" * 80)
        print()
        print("📁 Resultados salvos em:")
        print(f"   - {base_output_dir / 'zero_shot'}")
        for num_examples in args.examples:
            print(f"   - {base_output_dir / f'examples_{num_examples}'}")
        print()
        print(f"📂 Estrutura completa: {base_output_dir}")
        print()
    
    else:
        # Comportamento original: executar apenas uma estratégia
        # Detectar se é zero-shot
        is_zero_shot = args.prompt_strategy == 'zero-shot'
        
        # Se for zero-shot, ignorar exemplos e executar apenas uma vez
        if is_zero_shot:
            examples_to_run = [None]  # None indica zero-shot (sem exemplos)
            output_dir_name = "zero_shot"
        else:
            examples_to_run = args.examples
            output_dir_name = None  # Será definido por exemplo
        
        print("=" * 80)
        if is_zero_shot:
            print("🚀 Executando llm_sender com zero-shot (sem exemplos)")
        else:
            print("🚀 Executando llm_sender com diferentes números de exemplos")
        print("=" * 80)
        print(f"📄 CSV: {args.csv}")
        print(f"🌐 API: {args.api_url}")
        if len(models_to_use) > 1:
            print(f"🤖 Modelos: {', '.join(models_to_use)}")
        else:
            print(f"🤖 Modelo: {models_to_use[0]}")
        print(f"📝 Tipo de prompt: {args.prompt_type}")
        print(f"📁 Diretório de saída: {base_output_dir}")
        print(f"📝 Estratégia: {args.prompt_strategy}")
        print(f"📍 Posições: False (sempre desativado)")
        if not is_zero_shot:
            print(f"🔢 Números de exemplos: {args.examples}")
        print(f"📊 Índices: {args.start} até {args.end if args.end else 'fim'}")
        print("=" * 80)
        print()
        
        # Executar para cada número de exemplos (ou apenas uma vez para zero-shot)
        results = []
        for num_examples in examples_to_run:
            print()
            print("━" * 80)
            if is_zero_shot:
                print("📊 Executando zero-shot (sem exemplos)")
            else:
                print(f"📊 Executando com {num_examples} exemplo(s)")
            print("━" * 80)
            
            # Criar diretório de saída específico
            if is_zero_shot:
                output_dir = base_output_dir / output_dir_name
            else:
                output_dir = base_output_dir / f"examples_{num_examples}"
            
            try:
                process_csv(
                    csv_path=args.csv,
                    api_url=args.api_url,
                    delay=args.delay,
                    start_index=args.start,
                    end_index=args.end,
                    models=models_to_use,
                    prompt_strategy=args.prompt_strategy,
                    use_positions=False,  # Sempre False - with_positions foi removido
                    num_examples=num_examples,  # None para zero-shot
                    save_individual=True,
                    save_summary=True,
                    output_dir=str(output_dir),
                    resume_from_checkpoint=args.resume_from_checkpoint,
                    max_consecutive_timeouts=args.max_consecutive_timeouts,
                    prompt_type=args.prompt_type
                )
                if is_zero_shot:
                    results.append(("zero-shot", "✅ Sucesso"))
                    print("✅ Concluído com sucesso: zero-shot")
                else:
                    results.append((num_examples, "✅ Sucesso"))
                    print(f"✅ Concluído com sucesso: {num_examples} exemplo(s)")
            except Exception as e:
                if is_zero_shot:
                    results.append(("zero-shot", f"❌ Erro: {e}"))
                    print(f"❌ Erro ao executar zero-shot: {e}")
                else:
                    results.append((num_examples, f"❌ Erro: {e}"))
                    print(f"❌ Erro ao executar com {num_examples} exemplo(s): {e}")
                import traceback
                traceback.print_exc()
        
        # Resumo final
        print()
        print("=" * 80)
        print("📊 RESUMO DAS EXECUÇÕES")
        print("=" * 80)
        for identifier, status in results:
            if isinstance(identifier, int):
                print(f"   {identifier} exemplo(s): {status}")
            else:
                print(f"   {identifier}: {status}")
        print("=" * 80)
        print()
        print("📁 Resultados salvos em:")
        if is_zero_shot:
            output_dir = base_output_dir / output_dir_name
            print(f"   - {output_dir}")
        else:
            for num_examples in args.examples:
                output_dir = base_output_dir / f"examples_{num_examples}"
                print(f"   - {output_dir}")
        print()
        print(f"📂 Estrutura completa: {base_output_dir}")
        print()


if __name__ == "__main__":
    main()

