"""
Gerenciador de LLMs para extração de entidades biomédicas
"""

import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed

# Tentar importar OpenAI client para suporte a vLLM
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Resposta de um modelo LLM"""
    model_name: str
    response_text: str
    processing_time: float
    tokens_used: Optional[int] = None
    confidence: float = 0.0
    error: Optional[str] = None


class LLMManager:
    """Gerenciador para múltiplos modelos LLM via Singularity/HPC"""
    
    def __init__(
        self, 
        host: str = "http://localhost:11434", 
        timeout: int = 60, 
        max_retries: int = 3,
        api_type: str = "ollama",
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Inicializa o gerenciador de LLMs
        
        Args:
            host: Host do Ollama ou vLLM (via túnel SSH do HPC)
            timeout: Timeout em segundos (aumentado para HPC)
            max_retries: Número máximo de tentativas
            api_type: Tipo de API ("ollama" ou "openai" para vLLM)
            base_url: URL base para API OpenAI-compatible (ex: "http://localhost:11315/v1")
            model_name: Nome do modelo para API OpenAI-compatible (ex: "meta-llama/Llama-3.2-3B")
        """
        self.host = host
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_type = api_type.lower()
        self.base_url = base_url or f"{host}/v1" if api_type.lower() == "openai" else None
        # Garantir que model_name seja string limpa se fornecido
        self.model_name = str(model_name).strip() if model_name else None
        
        # DEBUG: Log de inicialização
        logger.info(f"🔧 LLMManager.__init__:")
        logger.info(f"   api_type recebido: '{api_type}' (type: {type(api_type)})")
        logger.info(f"   self.api_type: '{self.api_type}' (type: {type(self.api_type)})")
        logger.info(f"   model_name recebido: '{model_name}' (type: {type(model_name)})")
        logger.info(f"   self.model_name: '{self.model_name}' (type: {type(self.model_name)})")
        logger.info(f"   bool(self.model_name): {bool(self.model_name)}")
        logger.info(f"   base_url: '{base_url}'")
        logger.info(f"   self.base_url: '{self.base_url}'")
        
        # Inicializar cliente apropriado
        if self.api_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("Biblioteca 'openai' não está instalada. Execute: pip install openai")
            # Para vLLM local, não precisamos de api_key - usar uma chave dummy
            # A biblioteca openai exige uma chave mesmo para APIs locais
            self.client = OpenAI(
                base_url=self.base_url, 
                timeout=timeout,
                api_key="dummy-key-for-local-vllm"  # Chave dummy para APIs locais
            )
            self.ollama_client = None
            logger.info(f"🔧 Configurado para usar API OpenAI-compatible (vLLM) em {self.base_url}")
        else:
            self.client = ollama.Client(host=host)
            self.ollama_client = self.client
            self.openai_client = None
            logger.info(f"🔧 Configurado para usar Ollama em {host}")
        
        self.available_models: List[str] = []
        
        # Verificar conexão e modelos disponíveis (não falhar na inicialização)
        try:
            self._check_connection()
        except Exception as e:
            logger.warning(f"Aviso: Não foi possível conectar na inicialização: {e}")
            self.available_models = []
    
    def _check_connection(self):
        """Verifica conexão com Ollama ou vLLM via túnel SSH do HPC"""
        try:
            if self.api_type == "openai":
                # Para vLLM/OpenAI-compatible API
                models = self.client.models.list()
                # Extrair apenas o ID do modelo (garantir que seja string limpa)
                self.available_models = [str(model.id).strip() for model in models.data if hasattr(model, 'id')]
                logger.info(f"✅ Conectado ao vLLM em {self.base_url}")
            else:
                # Para Ollama
                models = self.client.list()
                self.available_models = [model.model for model in models.models]
                logger.info(f"✅ Conectado ao Ollama em {self.host}")
            
            logger.info(f"📋 Modelos disponíveis: {self.available_models}")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar: {e}")
            self.available_models = []
    
    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis"""
        return self.available_models.copy()
    
    def is_model_available(self, model_name: str) -> bool:
        """Verifica se um modelo está disponível"""
        # Para API OpenAI com model_name configurado, SEMPRE aceitar qualquer nome
        # (será substituído pelo model_name configurado em generate_single)
        if self.api_type == "openai" and self.model_name:
            logger.info(f"✅ API OpenAI: aceitando modelo '{model_name}' (será mapeado para '{self.model_name}')")
            return True
        
        # Se não há modelos carregados, tentar conectar novamente
        if not self.available_models:
            try:
                self._check_connection()
            except Exception as e:
                logger.warning(f"Não foi possível conectar: {e}")
                return False
        
        # Para OpenAI-compatible API sem model_name configurado, verificar se está na lista
        if self.api_type == "openai":
            # Se temos modelos disponíveis, aceitar qualquer nome
            if len(self.available_models) > 0:
                return True
            return False
        
        # Para Ollama, verificar se o modelo está na lista
        return model_name in self.available_models
    
    async def generate_single(
        self, 
        model_name: str, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1500
    ) -> LLMResponse:
        """
        Gera resposta de um único modelo
        
        Args:
            model_name: Nome do modelo
            prompt: Prompt do usuário
            system_prompt: Prompt do sistema (opcional)
            temperature: Temperatura para geração
            max_tokens: Máximo de tokens
            
        Returns:
            Resposta do modelo
        """
        # Para OpenAI-compatible API, SEMPRE usar model_name configurado se disponível
        actual_model = model_name
        
        # Log de debug para diagnóstico
        logger.info(f"🔍 generate_single chamado:")
        logger.info(f"   api_type: {self.api_type}")
        logger.info(f"   model_name (solicitado): {model_name}")
        logger.info(f"   self.model_name (configurado): {self.model_name}")
        logger.info(f"   self.base_url: {self.base_url}")
        
        if self.api_type == "openai":
            if self.model_name:
                # Tentar usar o modelo configurado, mas verificar se está disponível no vLLM
                # Se não estiver, tentar encontrar um modelo que corresponda
                configured_model = str(self.model_name).strip()
                
                # Atualizar lista de modelos disponíveis se necessário
                if not self.available_models:
                    try:
                        self._check_connection()
                    except Exception as e:
                        logger.warning(f"Não foi possível atualizar lista de modelos: {e}")
                
                # Verificar se o modelo configurado está na lista de modelos disponíveis
                if configured_model in self.available_models:
                    actual_model = configured_model
                    logger.info(f"✅ API OpenAI: usando modelo configurado '{actual_model}' (solicitado: '{model_name}')")
                elif len(self.available_models) == 1:
                    # Se há apenas um modelo disponível, usar esse
                    actual_model = self.available_models[0]
                    logger.warning(f"⚠️ Modelo configurado '{configured_model}' não encontrado, usando único modelo disponível: '{actual_model}'")
                elif self.available_models:
                    # Tentar encontrar correspondência parcial
                    matching_models = [m for m in self.available_models if configured_model.lower() in m.lower() or m.lower() in configured_model.lower()]
                    if matching_models:
                        actual_model = matching_models[0]
                        logger.warning(f"⚠️ Modelo configurado '{configured_model}' não encontrado, usando correspondência: '{actual_model}'")
                    else:
                        # Usar o primeiro modelo disponível como fallback
                        actual_model = self.available_models[0]
                        logger.warning(f"⚠️ Modelo configurado '{configured_model}' não encontrado, usando primeiro modelo disponível: '{actual_model}'")
                else:
                    # Se não há modelos disponíveis, tentar usar o configurado mesmo assim
                    actual_model = configured_model
                    logger.warning(f"⚠️ Não foi possível obter lista de modelos, tentando usar '{actual_model}'")
            else:
                # Se não há model_name configurado, usar o solicitado
                actual_model = str(model_name).strip()
                logger.warning(f"⚠️ API OpenAI sem model_name configurado, usando '{actual_model}'")
        
        # Verificar disponibilidade - para API OpenAI com model_name configurado, sempre aceitar
        # IMPORTANTE: Esta verificação deve ser feita ANTES de qualquer outra verificação
        logger.info(f"🔍 Verificação de disponibilidade:")
        logger.info(f"   api_type='{self.api_type}' (type: {type(self.api_type)})")
        logger.info(f"   self.model_name='{self.model_name}' (type: {type(self.model_name)})")
        logger.info(f"   bool(self.model_name)={bool(self.model_name)}")
        logger.info(f"   self.api_type == 'openai' = {self.api_type == 'openai'}")
        logger.info(f"   self.api_type == 'openai' and self.model_name = {self.api_type == 'openai' and bool(self.model_name)}")
        
        if self.api_type == "openai" and self.model_name:
            # Para API OpenAI com model_name configurado, sempre aceitar qualquer nome solicitado
            logger.info(f"✅ Aceitando requisição para '{model_name}', usando modelo '{actual_model}'")
            # Continuar para fazer a requisição (não precisa verificar disponibilidade)
        elif self.api_type == "openai":
            # API OpenAI mas sem model_name configurado - tentar usar o solicitado
            logger.warning(f"⚠️ API OpenAI sem model_name configurado, tentando usar '{actual_model}'")
            if not self.is_model_available(actual_model):
                error_msg = f"Modelo '{actual_model}' não está disponível (API OpenAI sem model_name configurado)"
                logger.error(f"❌ {error_msg}")
                return LLMResponse(
                    model_name=model_name,
                    response_text="",
                    processing_time=0.0,
                    error=error_msg
                )
        else:
            # Para Ollama ou outros, verificar disponibilidade normalmente
            if not self.is_model_available(model_name):
                error_msg = f"Modelo '{model_name}' não está disponível"
                logger.error(f"❌ {error_msg}")
                logger.error(f"   Debug: api_type={self.api_type}, model_name={self.model_name}, available_models={self.available_models}")
                return LLMResponse(
                    model_name=model_name,
                    response_text="",
                    processing_time=0.0,
                    error=error_msg
                )
        
        start_time = time.time()
        
        try:
            # Preparar mensagens
            # Modelos com chat template nativo (Llama-Instruct, Mistral, Phi-3, Qwen, Gemma)
            # podem usar system/user/assistant normalmente
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            # DEBUG: Log da requisição sendo enviada
            logger.debug(f"Enviando requisição para {actual_model} (API: {self.api_type})")
            logger.debug(f"Parâmetros: temperature={temperature}, max_tokens={max_tokens}")
            logger.debug(f"Mensagens: {messages}")
            
            # Fazer requisição com retry
            for attempt in range(self.max_retries):
                logger.debug(f"Tentativa {attempt + 1}/{self.max_retries}")
                try:
                    if self.api_type == "openai":
                        # Usar API OpenAI-compatible (vLLM)
                        response = self.client.chat.completions.create(
                            model=actual_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        response_text = response.choices[0].message.content.strip()
                    else:
                        # Usar Ollama
                        response = self.client.chat(
                            model=actual_model,
                            messages=messages,
                            options={
                                'temperature': temperature,
                                'top_p': 0.9,
                                'num_predict': max_tokens
                            }
                        )
                        response_text = response['message']['content'].strip()
                    
                    processing_time = time.time() - start_time
                    
                    # DEBUG: Log da resposta bruta da API
                    logger.debug("=" * 80)
                    logger.debug("📡 RESPOSTA BRUTA DA API:")
                    logger.debug("=" * 80)
                    logger.debug(f"Resposta completa: {response}")
                    logger.debug("=" * 80)
                    
                    # Tentar extrair confidence do JSON se possível
                    confidence = self._extract_confidence(response_text)
                    
                    return LLMResponse(
                        model_name=model_name,
                        response_text=response_text,
                        processing_time=processing_time,
                        confidence=confidence
                    )
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    logger.warning(f"Tentativa {attempt + 1} falhou para {actual_model}: {e}")
                    await asyncio.sleep(1)  # Aguardar antes da próxima tentativa
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Erro ao gerar com {model_name}: {e}")
            return LLMResponse(
                model_name=model_name,
                response_text="",
                processing_time=processing_time,
                error=str(e)
            )
    
    async def generate_multiple(
        self,
        model_names: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
        max_concurrent: int = 3
    ) -> List[LLMResponse]:
        """
        Gera respostas de múltiplos modelos em paralelo
        
        Args:
            model_names: Lista de nomes dos modelos
            prompt: Prompt do usuário
            system_prompt: Prompt do sistema (opcional)
            temperature: Temperatura para geração
            max_tokens: Máximo de tokens
            max_concurrent: Máximo de modelos simultâneos
            
        Returns:
            Lista de respostas dos modelos
        """
        # Filtrar modelos disponíveis
        available_models = [name for name in model_names if self.is_model_available(name)]
        
        if not available_models:
            logger.warning("Nenhum modelo disponível para geração")
            return []
        
        logger.info(f"Gerando respostas com {len(available_models)} modelos: {available_models}")
        
        # Executar em paralelo com limitação de concorrência
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(model_name):
            async with semaphore:
                return await self.generate_single(
                    model_name=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        
        # Executar todas as gerações
        tasks = [generate_with_semaphore(model) for model in available_models]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Processar resultados
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Erro na geração com {available_models[i]}: {response}")
                valid_responses.append(LLMResponse(
                    model_name=available_models[i],
                    response_text="",
                    processing_time=0.0,
                    error=str(response)
                ))
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    def _extract_confidence(self, response_text: str) -> float:
        """
        Extrai confidence score da resposta JSON
        
        Args:
            response_text: Texto da resposta
            
        Returns:
            Confidence score (0.0 a 1.0)
        """
        try:
            # Tentar parsear JSON
            if response_text.strip().startswith('{'):
                data = json.loads(response_text)
                
                # Procurar por confidence em diferentes estruturas
                if 'confidence' in data:
                    return float(data['confidence'])
                elif 'entities' in data and isinstance(data['entities'], list):
                    # Calcular confidence médio das entidades
                    confidences = [e.get('confidence', 0.0) for e in data['entities'] 
                                 if isinstance(e, dict) and 'confidence' in e]
                    if confidences:
                        return sum(confidences) / len(confidences)
            
            # Se não conseguir extrair, retornar confidence padrão
            return 0.8
            
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0.8
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Obtém informações sobre um modelo
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Informações do modelo
        """
        try:
            if self.api_type == "openai":
                # Para vLLM/OpenAI-compatible API, informações básicas
                return {
                    'name': self.model_name or model_name,
                    'size': 'Unknown',
                    'modified_at': 'Unknown',
                    'available': True,
                    'api_type': 'openai'
                }
            else:
                # Tentar obter informações do modelo Ollama
                info = self.client.show(model_name)
                return {
                    'name': model_name,
                    'size': info.get('size', 'Unknown'),
                    'modified_at': info.get('modified_at', 'Unknown'),
                    'available': True
                }
        except Exception as e:
            logger.warning(f"Erro ao obter info do modelo {model_name}: {e}")
            return {
                'name': model_name,
                'size': 'Unknown',
                'modified_at': 'Unknown',
                'available': False,
                'error': str(e)
            }
    
    def get_all_models_info(self) -> List[Dict[str, Any]]:
        """
        Obtém informações de todos os modelos disponíveis
        
        Returns:
            Lista de informações dos modelos
        """
        models_info = []
        for model_name in self.available_models:
            models_info.append(self.get_model_info(model_name))
        return models_info
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verifica saúde do sistema LLM
        
        Returns:
            Status do sistema
        """
        try:
            # Testar conexão
            start_time = time.time()
            if self.api_type == "openai":
                models = self.client.models.list()
            else:
                models = self.client.list()
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'host': self.host,
                'api_type': self.api_type,
                'base_url': self.base_url if self.api_type == "openai" else None,
                'response_time': round(response_time, 3),
                'available_models': len(self.available_models),
                'models': self.available_models[:5],  # Primeiros 5 modelos
                'timeout': self.timeout,
                'max_retries': self.max_retries
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'host': self.host,
                'api_type': self.api_type,
                'error': str(e),
                'available_models': 0,
                'models': []
            }


async def test_llm_manager():
    """Função de teste para o LLM Manager"""
    manager = LLMManager()
    
    # Verificar saúde
    health = await manager.health_check()
    print("🏥 Health Check:")
    for key, value in health.items():
        print(f"  {key}: {value}")
    
    # Testar geração simples
    if manager.available_models:
        model_name = manager.available_models[0]
        print(f"\n🧪 Testando geração com {model_name}:")
        
        prompt = "Extract medical entities from: 'Lithium carbonate toxicity in newborn infant.'"
        response = await manager.generate_single(model_name, prompt)
        
        print(f"  Resposta: {response.response_text[:100]}...")
        print(f"  Tempo: {response.processing_time:.2f}s")
        print(f"  Confiança: {response.confidence:.2f}")
        if response.error:
            print(f"  Erro: {response.error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_llm_manager())
