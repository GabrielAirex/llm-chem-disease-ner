"""
Gerenciador para múltiplas LLMs em portas diferentes
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .llm_manager import LLMManager, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class LLMInstance:
    """Instância de uma LLM com configurações específicas"""
    name: str
    host: str
    port: int
    manager: Optional[LLMManager] = None
    available: bool = True


class MultiLLMManager:
    """Gerenciador para múltiplas LLMs em portas diferentes"""
    
    def __init__(self, llm_instances: List[Dict[str, Any]], max_workers: Optional[int] = None):
        """
        Inicializa o gerenciador com múltiplas instâncias de LLM
        
        Args:
            llm_instances: Lista de configurações de LLMs
                [
                    {
                        "name": "llama3.2:3b",
                        "host": "localhost",
                        "port": 11434,
                        "models": ["llama3.2:3b"]
                    },
                    {
                        "name": "mistral:7b-instruct", 
                        "host": "localhost",
                        "port": 11435,
                        "models": ["mistral:7b-instruct"]
                    }
                ]
            max_workers: Número máximo de threads para processamento paralelo (padrão: número de instâncias)
        """
        self.instances: Dict[str, LLMInstance] = {}
        # ThreadPoolExecutor para processar cada LLM em thread separada
        # max_workers=None usa min(32, num_instances + 4) por padrão
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LLMThread")
        self._lock = threading.Lock()
        
        for config in llm_instances:
            name = config["name"]
            host = config["host"]
            port = config["port"]
            
            # Criar URL completa
            url = f"http://{host}:{port}"
            
            # Obter api_type e base_url da configuração (padrão: openai para vLLM)
            api_type = config.get("api_type", "openai")
            base_url = config.get("base_url", f"{url}/v1" if api_type == "openai" else None)
            model_name = config.get("model_name", name)
            
            # Criar instância do LLMManager
            try:
                manager = LLMManager(
                    host=url,
                    timeout=config.get("timeout", 60),
                    max_retries=config.get("max_retries", 3),
                    api_type=api_type,
                    base_url=base_url,
                    model_name=model_name
                )
                
                # Verificar se a conexão está funcionando (não falhar se não estiver)
                try:
                    available_models = manager.get_available_models()
                    available = True
                    logger.info(f"✅ Instância {name} configurada e disponível em {url}")
                    if available_models:
                        logger.info(f"   Modelos disponíveis: {available_models}")
                except Exception as e:
                    available = False
                    logger.warning(f"⚠️  Instância {name} configurada em {url} mas não está disponível: {e}")
                    logger.warning(f"   A API continuará funcionando apenas com os modelos disponíveis")
                
            except Exception as e:
                logger.error(f"❌ Erro ao criar instância {name}: {e}")
                logger.warning(f"   A API continuará funcionando apenas com os modelos disponíveis")
                # Criar um manager dummy para não quebrar a estrutura
                manager = None
                available = False
            
            # Criar instância
            instance = LLMInstance(
                name=name,
                host=host,
                port=port,
                manager=manager,
                available=available
            )
            
            self.instances[name] = instance
    
    async def generate_multiple(
        self, 
        model_names: List[str], 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1, 
        max_tokens: int = 2000
    ) -> List[LLMResponse]:
        """
        Gera respostas de múltiplas LLMs em paralelo usando threads separadas
        
        Cada LLM é processado em sua própria thread, permitindo verdadeiro paralelismo.
        Isso significa que uma requisição não bloqueia outra, mesmo que ambas usem
        modelos diferentes.
        
        Args:
            model_names: Lista de nomes de modelos
            prompt: Prompt para enviar
            system_prompt: Prompt do sistema (opcional, para separação system/user)
            temperature: Temperatura para geração
            max_tokens: Máximo de tokens
            
        Returns:
            Lista de respostas das LLMs
        """
        # Preparar lista de instâncias válidas
        valid_instances = []
        for model_name in model_names:
            if model_name in self.instances:
                instance = self.instances[model_name]
                if instance.available:
                    valid_instances.append((model_name, instance))
                else:
                    logger.warning(f"Instância {model_name} não está disponível")
            else:
                logger.warning(f"Instância {model_name} não encontrada")
        
        if not valid_instances:
            raise Exception("Nenhuma instância de LLM disponível")
        
        logger.info(f"🔄 Processando {len(valid_instances)} modelos em threads separadas: {[name for name, _ in valid_instances]}")
        
        # Executar cada LLM em uma thread separada usando ThreadPoolExecutor
        # Isso permite verdadeiro paralelismo - cada requisição não bloqueia outras
        loop = asyncio.get_event_loop()
        futures = []
        
        for model_name, instance in valid_instances:
            # Submeter tarefa para thread separada
            future = self.executor.submit(
                self._generate_single_sync,
                instance, model_name, prompt, system_prompt, temperature, max_tokens
            )
            futures.append((model_name, future))
        
        # Aguardar todas as threads completarem de forma assíncrona
        # Criar tasks assíncronas que aguardam cada future
        async def wait_for_future(model_name: str, future):
            """Aguarda um future em uma task assíncrona"""
            try:
                # Executar future.result() em thread separada para não bloquear
                result = await loop.run_in_executor(None, future.result)
                logger.info(f"✅ Thread do modelo {model_name} completou em {result.processing_time:.2f}s")
                return result
            except Exception as e:
                logger.error(f"❌ Erro na thread do modelo {model_name}: {e}")
                return LLMResponse(
                    model_name=model_name,
                    response_text="",
                    processing_time=0.0,
                    confidence=0.0,
                    error=str(e)
                )
        
        # Criar tasks para aguardar todos os futures
        tasks = [wait_for_future(model_name, future) for model_name, future in futures]
        
        # Aguardar todas as tasks com timeout
        results = []
        try:
            task_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300  # 5 minutos timeout total
            )
            # Filtrar exceções
            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    model_name = valid_instances[i][0]
                    logger.error(f"❌ Exceção na thread do modelo {model_name}: {result}")
                    results.append(LLMResponse(
                        model_name=model_name,
                        response_text="",
                        processing_time=0.0,
                        confidence=0.0,
                        error=str(result)
                    ))
                else:
                    results.append(result)
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Timeout ao aguardar threads (5 minutos)")
            # Criar respostas de erro para todos os modelos que não completaram
            completed_models = {r.model_name for r in results if hasattr(r, 'model_name')}
            for model_name, _ in valid_instances:
                if model_name not in completed_models:
                    results.append(LLMResponse(
                        model_name=model_name,
                        response_text="",
                        processing_time=0.0,
                        confidence=0.0,
                        error="Timeout ao processar requisição"
                    ))
        
        # Filtrar apenas resultados válidos (sem erro)
        valid_results = [r for r in results if r.error is None]
        
        if valid_results:
            logger.info(f"✅ {len(valid_results)} de {len(results)} modelos retornaram respostas válidas")
        else:
            logger.warning(f"⚠️ Nenhum modelo retornou resposta válida")
        
        return valid_results if valid_results else results
    
    def _generate_single_sync(
        self, 
        instance: LLMInstance, 
        model_name: str, 
        prompt: str, 
        system_prompt: Optional[str],
        temperature: float, 
        max_tokens: int
    ) -> LLMResponse:
        """
        Gera resposta de uma única LLM de forma síncrona (executada em thread separada)
        
        Esta função é executada em uma thread separada, permitindo que múltiplas
        requisições sejam processadas em paralelo sem bloqueio.
        """
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        logger.info(f"🧵 Thread {thread_name} (ID: {thread_id}) processando modelo {model_name}")
        
        if not instance.available or instance.manager is None:
            logger.warning(f"Instância {instance.name} não está disponível, pulando...")
            return LLMResponse(
                model_name=model_name,
                response_text="",
                processing_time=0.0,
                confidence=0.0,
                error=f"Instância {instance.name} não está disponível"
            )
        
        try:
            # Executar generate_single de forma síncrona (dentro da thread)
            # Como estamos em uma thread separada, podemos usar asyncio.run para criar
            # um novo event loop para esta thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    instance.manager.generate_single(
                        model_name=model_name,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                )
                logger.info(f"✅ Thread {thread_name} completou modelo {model_name} em {response.processing_time:.2f}s")
                return response
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"❌ Erro na thread {thread_name} para instância {instance.name}: {e}")
            # Marcar como não disponível para futuras tentativas (com lock para thread-safety)
            with self._lock:
                instance.available = False
            return LLMResponse(
                model_name=model_name,
                response_text="",
                processing_time=0.0,
                confidence=0.0,
                error=str(e)
            )
    
    async def _generate_single_async(
        self, 
        instance: LLMInstance, 
        model_name: str, 
        prompt: str, 
        system_prompt: Optional[str],
        temperature: float, 
        max_tokens: int
    ) -> LLMResponse:
        """
        Gera resposta de uma única LLM de forma assíncrona (método legado)
        
        Mantido para compatibilidade, mas agora generate_multiple usa threads.
        """
        if not instance.available or instance.manager is None:
            logger.warning(f"Instância {instance.name} não está disponível, pulando...")
            return LLMResponse(
                model_name=model_name,
                response_text="",
                processing_time=0.0,
                confidence=0.0,
                error=f"Instância {instance.name} não está disponível"
            )
        
        try:
            response = await instance.manager.generate_single(
                model_name=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            logger.error(f"Erro na instância {instance.name}: {e}")
            # Marcar como não disponível para futuras tentativas
            with self._lock:
                instance.available = False
            return LLMResponse(
                model_name=model_name,
                response_text="",
                processing_time=0.0,
                confidence=0.0,
                error=str(e)
            )
    
    def get_available_instances(self) -> List[str]:
        """Retorna lista de instâncias disponíveis"""
        return [name for name, instance in self.instances.items() if instance.available]
    
    def get_instance_info(self) -> Dict[str, Dict[str, Any]]:
        """Retorna informações das instâncias"""
        info = {}
        for name, instance in self.instances.items():
            if instance.manager is not None:
                try:
                    models = instance.manager.get_available_models()
                except:
                    models = []
            else:
                models = []
            
            info[name] = {
                "host": instance.host,
                "port": instance.port,
                "available": instance.available,
                "models": models
            }
        return info
    
    def shutdown(self):
        """Encerra o ThreadPoolExecutor"""
        logger.info("🛑 Encerrando ThreadPoolExecutor...")
        self.executor.shutdown(wait=True)
        logger.info("✅ ThreadPoolExecutor encerrado")
