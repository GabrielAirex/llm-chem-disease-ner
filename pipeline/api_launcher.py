"""
Módulo para iniciar instâncias da API em threads separadas
Cada instância é configurada para usar apenas um modelo específico
"""

import threading
import uvicorn
import time
import requests
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import logging

# Configurar paths para importar a API
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class APILauncher:
    """Gerencia uma instância da API em uma thread separada"""
    
    def __init__(self, model_name: str, base_port: int = 9000, host: str = "127.0.0.1"):
        """
        Inicializa o launcher da API
        
        Args:
            model_name: Nome do modelo (ex: "internlm/internlm2_5-7b-chat")
            base_port: Porta base (será incrementada para cada modelo)
            host: Host para bind da API
        """
        self.model_name = model_name
        self.host = host
        # Calcular porta baseada no hash do nome do modelo para consistência
        # Usar abs() para garantir valor positivo
        port_offset = abs(hash(model_name)) % 1000
        self.port = base_port + port_offset
        # Garantir que a porta seja única (evitar colisões)
        # Se a porta já estiver em uso, tentar próximas portas
        self.port = self._find_available_port(self.port)
        self.server_thread: Optional[threading.Thread] = None
        self.server: Optional[uvicorn.Server] = None
        self.running = False
        self.api_url = f"http://{host}:{self.port}"
    
    def _find_available_port(self, start_port: int, max_attempts: int = 100) -> int:
        """Encontra uma porta disponível começando de start_port"""
        import socket
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        # Se não encontrou, retornar a porta original (pode dar erro depois)
        return start_port
        
    def _get_model_port(self) -> Optional[int]:
        """Obtém a porta do modelo do config.yaml"""
        try:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                llm_ports = config.get('llm_ports', {})
                if self.model_name in llm_ports:
                    return llm_ports[self.model_name].get('port')
        except Exception as e:
            logger.warning(f"Erro ao ler config.yaml: {e}")
        return None
    
    def _run_server(self):
        """Executa o servidor uvicorn em uma thread separada"""
        try:
            # Configurar uvicorn para rodar em thread separada
            # Cada thread terá seu próprio event loop
            import asyncio
            
            # Criar novo event loop para esta thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            config = uvicorn.Config(
                "src.api.main:app",
                host=self.host,
                port=self.port,
                log_level="warning",  # Reduzir logs para múltiplas instâncias
                access_log=False,
                loop="asyncio"
            )
            self.server = uvicorn.Server(config)
            self.running = True
            logger.info(f"🚀 API iniciada para modelo {self.model_name} em {self.api_url}")
            self.server.run()
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar API para {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    def start(self, wait_ready: bool = True, timeout: int = 30):
        """
        Inicia a API em uma thread separada
        
        Args:
            wait_ready: Se True, aguarda a API estar pronta antes de retornar
            timeout: Timeout em segundos para aguardar API estar pronta
        """
        if self.running:
            logger.warning(f"API já está rodando para {self.model_name}")
            return
        
        # Iniciar thread do servidor
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name=f"API-{self.model_name}"
        )
        self.server_thread.start()
        
        if wait_ready:
            # Aguardar API estar pronta
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(f"{self.api_url}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info(f"✅ API pronta para {self.model_name} em {self.api_url}")
                        return
                except:
                    pass
                time.sleep(0.5)
            
            logger.warning(f"⚠️ Timeout aguardando API para {self.model_name}")
    
    def stop(self):
        """Para a API"""
        if self.server and self.running:
            self.server.should_exit = True
            self.running = False
            logger.info(f"🛑 API parada para {self.model_name}")
    
    def is_running(self) -> bool:
        """Verifica se a API está rodando"""
        if not self.running:
            return False
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False


def launch_api_for_model(model_name: str, base_port: int = 9000) -> APILauncher:
    """
    Função auxiliar para iniciar API para um modelo específico
    
    Args:
        model_name: Nome do modelo
        base_port: Porta base (será ajustada automaticamente)
    
    Returns:
        Instância do APILauncher
    """
    launcher = APILauncher(model_name, base_port=base_port)
    launcher.start()
    return launcher

