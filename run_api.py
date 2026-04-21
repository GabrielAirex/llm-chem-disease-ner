#!/usr/bin/env python3
"""
Script para executar a API BioNER_llm

Uso:
    python utils/run_api.py
    python utils/run_api.py --port 8002
    python utils/run_api.py --host 127.0.0.1 --port 8001 --no-reload
"""

import uvicorn
import sys
import argparse
from pathlib import Path


def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    try:
        import fastapi
        import uvicorn
        return True
    except ImportError as e:
        print(f"❌ Erro: Dependência não encontrada: {e}")
        print("💡 Execute: pip install -r requirements.txt")
        return False


def setup_paths():
    """Configura os paths do projeto"""
    # Obter diretório raiz do projeto (2 níveis acima de utils/)
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    
    # Adicionar ao path
    if src_path.exists():
        sys.path.insert(0, str(project_root))
        return True
    else:
        print(f"❌ Erro: Diretório src não encontrado em {project_root}")
        return False


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Inicia a API BioNER_llm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python utils/run_api.py
  python utils/run_api.py --port 8002
  python utils/run_api.py --host 127.0.0.1 --no-reload
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host para bind da API (padrão: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Porta para a API (padrão: 8001)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        default=True,
        help="Habilitar reload automático (padrão: True)"
    )
    
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Desabilitar reload automático"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Nível de log (padrão: info)"
    )
    
    args = parser.parse_args()
    
    # Verificar dependências
    if not check_dependencies():
        sys.exit(1)
    
    # Configurar paths
    if not setup_paths():
        sys.exit(1)
    
    # Informações de inicialização
    print("🚀 Iniciando BioNER_llm API...")
    print("=" * 60)
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Reload: {args.reload}")
    print(f"   Log Level: {args.log_level}")
    print("=" * 60)
    print(f"   URL: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print(f"   Docs: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
    print(f"   Health: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/health")
    print("\n   Para parar: Ctrl+C")
    print("=" * 60)
    print()
    
    # Iniciar servidor
    try:
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        print("\n\n🛑 API interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro ao iniciar API: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
