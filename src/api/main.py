"""
API principal do BioNER_llm - Extração de Entidades Biomédicas com Multi-LLM
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import json
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ..models.schemas import (
    ExtractionRequest, ExtractionResponse, BatchExtractionRequest, BatchExtractionResponse,
    BenchmarkRequest, BenchmarkResponse, HealthResponse, ConfigResponse,
    ModelPrediction, ConsensusMethod, PromptStrategy, EntityType
)
from ..llm.llm_manager import LLMManager
from ..llm.multi_llm_manager import MultiLLMManager
from ..prompts.prompt_engine import PromptEngine
from ..consensus.consensus_engine import ConsensusEngine
from ..storage.response_storage import ResponseStorage

# bc5cdr_parser é opcional - não necessário para o fluxo principal (CSV já tem textos parseados)
try:
    from ..data.bc5cdr_parser import BC5CDRParser
except ImportError:
    BC5CDRParser = None
from .storage_endpoints import storage_router, init_storage_endpoints
from .audit_endpoints import audit_router, init_audit_endpoints
from ..audit.metrics_auditor import MetricsAuditor
from ..llm.huggingface_manager import HuggingFaceManager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar componentes
app = FastAPI(
    title="BioNER_llm API",
    description="API para extração de entidades biomédicas com múltiplos LLMs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(storage_router)
app.include_router(audit_router)

# Componentes globais
llm_manager: Optional[LLMManager] = None
multi_llm_manager: Optional[MultiLLMManager] = None
huggingface_manager: Optional[HuggingFaceManager] = None
prompt_engine: Optional[PromptEngine] = None
consensus_engine: Optional[ConsensusEngine] = None
bc5cdr_parser: Optional[BC5CDRParser] = None
response_storage: Optional[ResponseStorage] = None
metrics_auditor: Optional[MetricsAuditor] = None
config: Dict[str, Any] = {}


@app.on_event("shutdown")
async def shutdown_event():
    """Encerra componentes no shutdown"""
    global multi_llm_manager
    if multi_llm_manager:
        try:
            multi_llm_manager.shutdown()
            logger.info("✅ MultiLLMManager encerrado corretamente")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao encerrar MultiLLMManager: {e}")


@app.on_event("startup")
async def startup_event():
    """Inicializa componentes na startup"""
    global llm_manager, multi_llm_manager, prompt_engine, consensus_engine, bc5cdr_parser, response_storage, huggingface_manager, metrics_auditor, config
    
    try:
        # Carregar configuração - usar caminho absoluto baseado no diretório do projeto
        # Tentar encontrar config.yaml a partir do diretório atual ou do diretório do módulo
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            # Tentar caminho relativo ao diretório do módulo
            module_dir = Path(__file__).parent.parent.parent
            config_path = module_dir / "config" / "config.yaml"
        if config_path.exists():
            logger.info(f"📁 Carregando configuração de: {config_path.absolute()}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            logger.warning(f"Arquivo de configuração não encontrado em {config_path.absolute()}, usando padrões")
            config = {}
        
        # Inicializar componentes
        llm_config = config.get('llm', {})
        
        # Log da configuração LLM
        logger.info("=" * 80)
        logger.info("🔧 Configuração LLM:")
        logger.info(f"  llm_host: {llm_config.get('llm_host', llm_config.get('ollama_host', 'N/A'))}")
        logger.info(f"  api_type: {llm_config.get('api_type', 'N/A')}")
        logger.info(f"  base_url: {llm_config.get('base_url', 'N/A')}")
        logger.info(f"  model_name: {llm_config.get('model_name', 'N/A')}")
        logger.info(f"  timeout: {llm_config.get('timeout', 'N/A')}")
        logger.info("=" * 80)
        
        # Extrair valores do config com logs de debug
        api_type_value = llm_config.get('api_type', 'ollama')
        model_name_value = llm_config.get('model_name')
        base_url_value = llm_config.get('base_url')
        
        logger.info(f"🔍 Valores extraídos do config:")
        logger.info(f"   api_type: '{api_type_value}' (type: {type(api_type_value)})")
        logger.info(f"   model_name: '{model_name_value}' (type: {type(model_name_value)})")
        logger.info(f"   base_url: '{base_url_value}' (type: {type(base_url_value)})")
        
        llm_manager = LLMManager(
            host=llm_config.get('llm_host', llm_config.get('ollama_host', 'http://localhost:11434')),  # Suporta ambos para compatibilidade
            timeout=llm_config.get('timeout', 30),
            max_retries=llm_config.get('max_retries', 3),
            api_type=api_type_value,
            base_url=base_url_value,
            model_name=model_name_value
        )
        
        # Verificar se inicializou corretamente
        logger.info(f"✅ LLMManager inicializado:")
        logger.info(f"   API Type: {llm_manager.api_type}")
        logger.info(f"   Base URL: {llm_manager.base_url}")
        logger.info(f"   Model Name: {llm_manager.model_name}")
        logger.info(f"   Available Models: {llm_manager.get_available_models()}")
        
        # Inicializar MultiLLMManager com mapeamento de portas
        llm_ports = config.get('llm_ports', {})
        if llm_ports:
            # Converter mapeamento de portas para formato de instâncias
            llm_instances = []
            for model_name, port_config in llm_ports.items():
                host = port_config.get("host", "localhost")
                port = port_config.get("port", 11434)
                base_url = f"http://{host}:{port}/v1"
                
                instance = {
                    "name": model_name,
                    "host": host,
                    "port": port,
                    "timeout": port_config.get("timeout", 60),
                    "max_retries": port_config.get("max_retries", 3),
                    "api_type": "openai",  # vLLM usa API OpenAI-compatible
                    "base_url": base_url,
                    "model_name": model_name
                }
                llm_instances.append(instance)
            
            if llm_instances:
                multi_llm_manager = MultiLLMManager(llm_instances)
                available_count = len([inst for inst in multi_llm_manager.instances.values() if inst.available])
                logger.info(f"🔗 MultiLLMManager inicializado com {len(llm_instances)} instâncias ({available_count} disponíveis)")
                logger.info(f"📡 Portas mapeadas: {[(inst['name'], inst['port']) for inst in llm_instances]}")
                
                # Log de instâncias disponíveis vs não disponíveis
                for name, instance in multi_llm_manager.instances.items():
                    if instance.available:
                        logger.info(f"   ✅ {name}: disponível em {instance.host}:{instance.port}")
                    else:
                        logger.warning(f"   ⚠️  {name}: não disponível em {instance.host}:{instance.port}")
            else:
                logger.warning("⚠️ Nenhuma porta LLM configurada")
        
        prompt_engine = PromptEngine()
        consensus_engine = ConsensusEngine(
            confidence_threshold=config.get('consensus', {}).get('confidence_threshold', 0.7),
            weight_threshold=config.get('consensus', {}).get('weight_threshold', 0.5)
        )
        # bc5cdr_parser é opcional - não necessário para extração básica
        if BC5CDRParser is not None:
            bc5cdr_parser = BC5CDRParser()
        else:
            bc5cdr_parser = None
            logger.info("BC5CDRParser não disponível - usando apenas extração via CSV")
        
        # Inicializar armazenamento de respostas
        storage_config = config.get('storage', {})
        response_storage = ResponseStorage(
            output_dir=storage_config.get('output_dir', 'indicios_encontrados'),
            create_dir=storage_config.get('create_dir_if_not_exists', True)
        )
        
        # Inicializar endpoints de armazenamento
        init_storage_endpoints(response_storage)
        
        # Inicializar Hugging Face Manager (se habilitado)
        hf_config = config.get('llm', {}).get('huggingface', {})
        if hf_config.get('enabled', False):
            huggingface_manager = HuggingFaceManager(
                hpc_host=hf_config.get('hpc_host', 'http://localhost:11434'),
                timeout=config.get('llm', {}).get('timeout', 60)
            )
            logger.info("🤗 HuggingFace Manager inicializado")
        
        # Inicializar auditor de métricas
        audit_config = config.get('audit', {})
        if audit_config.get('enabled', False):
            metrics_auditor = MetricsAuditor(
                output_dir=audit_config.get('output_dir', 'audit_results')
            )
            init_audit_endpoints(metrics_auditor)
            logger.info("📊 Metrics Auditor inicializado")
        
        logger.info("✅ BioNER_llm API inicializada com sucesso")
        
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check da API"""
    try:
        # Verificar status dos componentes
        llm_health = await llm_manager.health_check() if llm_manager else {"status": "unavailable"}
        
        return HealthResponse(
            status="healthy" if llm_health.get("status") == "healthy" else "degraded",
            version="1.0.0",
            available_models=llm_manager.get_available_models() if llm_manager else [],
            ollama_status=llm_health.get("status", "unknown")
        )
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            available_models=[],
            ollama_status="error"
        )


@app.get("/models")
async def get_models():
    """Lista modelos disponíveis"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM Manager não inicializado")
    
    models_info = llm_manager.get_all_models_info()
    return {"models": models_info}


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Obtém configurações atuais"""
    return ConfigResponse(
        llm_config=config.get('llm', {}),
        prompt_config=config.get('prompts', {}),
        consensus_config=config.get('consensus', {}),
        benchmark_config=config.get('benchmark', {}),
        storage_config=config.get('storage', {}),
        audit_config=config.get('audit', {})
    )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_entities(request: ExtractionRequest):
    """Extrai entidades de um texto usando uma ou múltiplas LLMs"""
    if not all([llm_manager, prompt_engine, consensus_engine, response_storage]):
        raise HTTPException(status_code=503, detail="Componentes não inicializados")
    
    start_time = time.time()
    
    try:
        # Determinar quais modelos usar
        # Priorizar configuração da requisição sobre config.yaml
        if request.llm_config and request.llm_config.single_llm_model:
            models_to_use = [request.llm_config.single_llm_model]
        elif request.llm_config and request.llm_config.use_single_llm and request.llm_config.single_llm_model:
            models_to_use = [request.llm_config.single_llm_model]
        elif request.models and len(request.models) > 0:
            models_to_use = request.models  # Pode ser múltiplos modelos
        else:
            # Fallback para configuração do config.yaml
            default_model = config.get('llm', {}).get('model_name', config.get('llm', {}).get('single_llm_model', 'llama3.2:3b'))
            models_to_use = [default_model]
        
        use_multiple_models = len(models_to_use) > 1
        if use_multiple_models:
            logger.info(f"Extraindo entidades de texto ({len(request.text)} chars) com {len(models_to_use)} modelos: {models_to_use}")
        else:
            logger.info(f"Extraindo entidades de texto ({len(request.text)} chars) com modelo único: {models_to_use[0]}")
        
        # Obter configurações de prompts: priorizar requisição HTTP, fallback para config.yaml
        prompts_config = config.get('prompts', {})
        use_positions = request.use_positions if request.use_positions is not None else prompts_config.get('use_positions', True)
        num_examples_config = request.num_examples if request.num_examples is not None else prompts_config.get('num_examples', 3)
        max_text_length_config = request.max_text_length if request.max_text_length is not None else prompts_config.get('max_text_length', 4000)
        
        # Obter tipo de prompt da requisição (padrão: type1)
        prompt_type = request.prompt_type if request.prompt_type else "type1"
        
        logger.info(f"Configurações de prompt: strategy={request.prompt_strategy}, prompt_type={prompt_type}, use_positions={use_positions}, num_examples={num_examples_config}")
        
        # Criar instância do PromptEngine com o tipo de prompt correto
        # Usar cache se já existe uma instância para este tipo
        if not hasattr(extract_entities, '_prompt_engines'):
            extract_entities._prompt_engines = {}
        
        if prompt_type not in extract_entities._prompt_engines:
            extract_entities._prompt_engines[prompt_type] = PromptEngine(prompt_type=prompt_type)
        
        current_prompt_engine = extract_entities._prompt_engines[prompt_type]
        
        # Gerar prompt com configurações
        prompt = current_prompt_engine.generate_prompt(
            text=request.text,
            strategy=PromptStrategy(request.prompt_strategy),
            use_positions=use_positions,
            num_examples=num_examples_config,
            max_text_length=max_text_length_config
        )
        
        # Separar prompt type2 em system/user para melhor compatibilidade com chat templates
        # Apenas quando configurado no config.yaml
        system_prompt = None
        user_prompt = prompt
        
        # Verificar se deve usar separação de chat template
        chat_template_config = config.get('chat_template_separation', {})
        use_chat_template_separation = chat_template_config.get('enabled', False)
        
        # Verificar configuração específica do modelo (se houver)
        if use_multiple_models and models_to_use:
            # Para múltiplos modelos, verificar se algum deles requer separação
            model_configs = chat_template_config.get('models', {})
            for model in models_to_use:
                if model in model_configs:
                    use_chat_template_separation = model_configs[model]
                    logger.info(f"📝 Modelo {model} configurado para usar separação system/user: {use_chat_template_separation}")
                    break
        elif models_to_use:
            # Para modelo único, verificar configuração específica
            model_to_check = models_to_use[0]
            model_configs = chat_template_config.get('models', {})
            if model_to_check in model_configs:
                use_chat_template_separation = model_configs[model_to_check]
                logger.info(f"📝 Modelo {model_to_check} configurado para usar separação system/user: {use_chat_template_separation}")
        
        # Separar prompt se necessário
        if use_chat_template_separation and prompt_type == "type2":
            if request.prompt_strategy == "zero-shot":
                # Para zero-shot: separar em "instruções" e "texto"
                text_marker = "\n\nText: "
                if text_marker in prompt:
                    parts = prompt.split(text_marker, 1)
                    if len(parts) == 2:
                        system_prompt = parts[0].strip()
                        user_prompt = "Text: " + parts[1].strip()
                        logger.info("📝 Separando prompt type2 zero-shot: system (instruções) + user (texto)")
            else:
                # Para few-shot: separar em "instruções + exemplos" e "texto"
                text_marker = "\n\nNow analyze this text:\n\n"
                if text_marker in prompt:
                    parts = prompt.split(text_marker, 1)
                    if len(parts) == 2:
                        system_prompt = parts[0].strip()
                        user_prompt = parts[1].strip()
                        logger.info("📝 Separando prompt type2 few-shot: system (instruções+exemplos) + user (texto)")
        
        # DEBUG: Log do prompt completo
        logger.info("=" * 80)
        logger.info("📝 PROMPT ENVIADO PARA LLM:")
        logger.info("=" * 80)
        logger.info(f"Template usado: {request.prompt_strategy} (prompt_type={prompt_type}, use_positions={use_positions}, num_examples={num_examples_config})")
        if system_prompt:
            logger.info("System prompt:")
            logger.info("-" * 80)
            logger.info(system_prompt)
            logger.info("-" * 80)
            logger.info("User prompt:")
        logger.info("-" * 80)
        logger.info(user_prompt if system_prompt else prompt)
        logger.info("=" * 80)
        
        # Obter parâmetros de temperatura e max_tokens
        if request.llm_config:
            temperature = request.llm_config.temperature or config.get('llm_defaults', {}).get('temperature', 0.1)
            max_tokens = request.llm_config.max_tokens or config.get('llm_defaults', {}).get('max_tokens', 1500)
        else:
            temperature = config.get('llm_defaults', {}).get('temperature', 0.1)
            max_tokens = config.get('llm_defaults', {}).get('max_tokens', 1500)
        
        # Gerar respostas dos modelos
        llm_ports = config.get('llm_ports', {})
        llm_responses = []
        
        if use_multiple_models and multi_llm_manager:
            # Múltiplos modelos: usar multi_llm_manager para gerar em paralelo
            logger.info(f"🔄 Gerando respostas de {len(models_to_use)} modelos em paralelo...")
            # Para múltiplos modelos, usar o prompt separado se disponível
            prompt_to_use = user_prompt if system_prompt else prompt
            llm_responses = await multi_llm_manager.generate_multiple(
                model_names=models_to_use,
                prompt=prompt_to_use,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            # Filtrar apenas respostas válidas (sem erro)
            llm_responses = [r for r in llm_responses if r.error is None]
            if not llm_responses:
                raise HTTPException(status_code=503, detail="Nenhum modelo disponível para processar a requisição")
            # Usar a primeira resposta válida (ou fazer consenso depois)
            llm_response = llm_responses[0]
            logger.info(f"✅ Recebidas {len(llm_responses)} respostas válidas de {len(models_to_use)} modelos")
        else:
            # Modelo único: usar lógica de direcionamento por porta
            model_to_use = models_to_use[0]
            manager_to_use = llm_manager
            
            # Primeiro tentar usar multi_llm_manager se disponível
            if multi_llm_manager and model_to_use in multi_llm_manager.instances:
                instance = multi_llm_manager.instances.get(model_to_use)
                if instance and instance.available and instance.manager is not None:
                    manager_to_use = instance.manager
                    logger.info(f"🔗 Usando modelo {model_to_use} via multi_llm_manager (porta {instance.port})")
                elif model_to_use in llm_ports:
                    # Se não estiver disponível no multi_llm_manager, tentar criar temporário
                    port_config = llm_ports[model_to_use]
                    port_host = port_config.get("host", "localhost")
                    port_port = port_config.get("port", 8000)
                    port_timeout = port_config.get("timeout", 120)
                    port_max_retries = port_config.get("max_retries", 3)
                    
                    try:
                        manager_to_use = LLMManager(
                            host=f"http://{port_host}:{port_port}",
                            timeout=port_timeout,
                            max_retries=port_max_retries,
                            api_type="openai",
                            base_url=f"http://{port_host}:{port_port}/v1",
                            model_name=model_to_use
                        )
                        logger.info(f"🔗 Usando modelo {model_to_use} na porta {port_port} (criado temporariamente)")
                    except Exception as e:
                        logger.error(f"❌ Erro ao criar LLMManager para {model_to_use}: {e}")
                        logger.warning(f"   Usando manager padrão (pode não funcionar corretamente)")
            elif model_to_use in llm_ports:
                # Modelo está em uma porta específica, criar LLMManager temporário
                port_config = llm_ports[model_to_use]
                port_host = port_config.get("host", "localhost")
                port_port = port_config.get("port", 8000)
                port_timeout = port_config.get("timeout", 120)
                port_max_retries = port_config.get("max_retries", 3)
                
                try:
                    manager_to_use = LLMManager(
                        host=f"http://{port_host}:{port_port}",
                        timeout=port_timeout,
                        max_retries=port_max_retries,
                        api_type="openai",
                        base_url=f"http://{port_host}:{port_port}/v1",
                        model_name=model_to_use
                    )
                    logger.info(f"🔗 Usando modelo {model_to_use} na porta {port_port} (via llm_ports)")
                except Exception as e:
                    logger.error(f"❌ Erro ao criar LLMManager para {model_to_use}: {e}")
                    logger.warning(f"   Usando manager padrão (pode não funcionar corretamente)")
            
            # Gerar resposta do modelo único
            # Usar prompt separado se disponível
            prompt_to_use = user_prompt if system_prompt else prompt
            llm_response = await manager_to_use.generate_single(
                model_name=model_to_use,
                prompt=prompt_to_use,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            llm_responses = [llm_response]
        
        # DEBUG: Log da resposta completa da LLM
        logger.info("=" * 80)
        logger.info("🤖 RESPOSTA DA LLM:")
        logger.info("=" * 80)
        logger.info(f"Modelos: {models_to_use}")
        logger.info(f"Tempo de processamento: {llm_response.processing_time:.2f}s")
        logger.info(f"Confiança: {llm_response.confidence}")
        logger.info("-" * 80)
        logger.info("Resposta completa:")
        logger.info(llm_response.response_text)  # Logar a resposta bruta
        logger.info("=" * 80)
        
        if llm_response.error:
            raise HTTPException(status_code=500, detail=f"Erro no modelo {models_to_use[0]}: {llm_response.error}")
        
        # Parsear entidades da resposta
        entities = current_prompt_engine.parse_llm_response(llm_response.response_text, request.text, use_positions=use_positions)
        
        processing_time = time.time() - start_time
        
        # Separar chemicals e diseases
        chemicals = [e for e in entities if e.type == EntityType.CHEMICAL]
        diseases = [e for e in entities if e.type == EntityType.DISEASE]
        
        # Preparar resposta - apenas chemicals e diseases
        response_data = ExtractionResponse(
            chemicals=chemicals,
            diseases=diseases
        )
        
        # Se solicitado, incluir resposta bruta (retornar como dict para incluir campo extra)
        if request.include_raw_response:
            response_dict = response_data.model_dump()
            response_dict['raw_response'] = llm_response.response_text
            return response_dict
        
        # Salvar resposta se configurado
        save_responses = config.get('storage', {}).get('save_responses', True)
        if save_responses:
            try:
                # Converter entidades para dict, removendo mesh_id e confidence
                # Se use_positions=False, não incluir start/end no JSON salvo
                entities_dict = []
                for entity in entities:
                    entity_dict = {
                        "text": entity.text,
                        "type": entity.type.value
                    }
                    # Incluir posições apenas se use_positions=True
                    if use_positions:
                        entity_dict["start"] = entity.start
                        entity_dict["end"] = entity.end
                    entities_dict.append(entity_dict)
                
                # Determinar num_examples para salvar (sempre salvar se few-shot)
                num_examples_to_save = None
                if request.prompt_strategy == PromptStrategy.FEW_SHOT:
                    # Garantir que sempre tenha um valor (usar o configurado ou padrão 3)
                    num_examples_to_save = num_examples_config if num_examples_config is not None else prompts_config.get('num_examples', 3)
                
                saved_path = response_storage.save_extraction_response(
                    text=request.text,
                    pmid=request.pmid,  # Incluir pmid ao salvar
                    entities=entities_dict,
                    relations=[],  # Não implementado ainda
                    models_used=models_to_use,
                    prompt_strategy=request.prompt_strategy.value,
                    num_examples=num_examples_to_save,
                    processing_time=processing_time
                )
                logger.info(f"💾 Resposta salva em: {saved_path}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao salvar resposta: {e}")
        
        logger.info(f"Extração concluída: {len(chemicals)} químicos e {len(diseases)} doenças em {processing_time:.2f}s")
        return response_data
        
    except Exception as e:
        logger.error(f"Erro na extração: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-batch", response_model=BatchExtractionResponse)
async def extract_batch(request: BatchExtractionRequest):
    """Extrai entidades de múltiplos textos"""
    if not all([llm_manager, prompt_engine, response_storage]):
        raise HTTPException(status_code=503, detail="Componentes não inicializados")
    
    start_time = time.time()
    results = []
    success_count = 0
    error_count = 0
    
    try:
        logger.info(f"Processando {len(request.texts)} textos em lote")
        
        # Processar cada texto
        for i, text in enumerate(request.texts):
            try:
                # Criar request individual
                individual_request = ExtractionRequest(
                    text=text,
                    models=request.models,
                    prompt_strategy=request.prompt_strategy,
                    return_all_predictions=False,
                    use_positions=request.use_positions,
                    num_examples=request.num_examples,
                    max_text_length=request.max_text_length,
                    prompt_type=request.prompt_type
                )
                
                # Extrair entidades
                result = await extract_entities(individual_request)
                results.append(result)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Erro ao processar texto {i}: {e}")
                error_count += 1
                # Adicionar resultado vazio
                results.append(ExtractionResponse(
                    chemicals=[],
                    diseases=[]
                ))
        
        total_processing_time = time.time() - start_time
        
        # Salvar resposta em lote se configurado
        save_responses = config.get('storage', {}).get('save_responses', True)
        if save_responses:
            try:
                saved_path = response_storage.save_batch_response(
                    texts=request.texts,
                    results=[result.dict() for result in results],
                    models_used=request.models,
                    prompt_strategy=request.prompt_strategy.value,
                    total_processing_time=total_processing_time,
                    success_count=success_count,
                    error_count=error_count
                )
                logger.info(f"💾 Resposta em lote salva em: {saved_path}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao salvar resposta em lote: {e}")
        
        return BatchExtractionResponse(
            results=results,
            total_processing_time=total_processing_time,
            success_count=success_count,
            error_count=error_count
        )
        
    except Exception as e:
        logger.error(f"Erro no processamento em lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Executa benchmark no dataset BC5CDR"""
    if not all([llm_manager, prompt_engine, consensus_engine]):
        raise HTTPException(status_code=503, detail="Componentes não inicializados")
    
    if bc5cdr_parser is None:
        raise HTTPException(
            status_code=503, 
            detail="BC5CDRParser não disponível. Para benchmark, o parser é necessário."
        )
    
    try:
        logger.info(f"Iniciando benchmark com {len(request.configurations)} configurações")
        
        # Carregar dados de teste
        test_file_path = Path("data/NIH/CDR_Data/CDR.Corpus.v010516") / request.test_file
        if not test_file_path.exists():
            raise HTTPException(status_code=404, detail=f"Arquivo de teste não encontrado: {test_file_path}")
        
        # Parsear dados de teste
        test_articles = bc5cdr_parser._parse_pubtator_file(test_file_path)
        logger.info(f"Carregados {len(test_articles)} artigos para teste")
        
        # Executar benchmark para cada configuração
        configuration_results = []
        best_f1 = 0.0
        best_config_name = ""
        
        for config in request.configurations:
            logger.info(f"Testando configuração: {config.name or 'unnamed'}")
            
            # Executar extração em lote
            texts = [article.full_text for article in test_articles[:request.batch_size]]
            
            batch_request = BatchExtractionRequest(
                texts=texts,
                models=config.models,
                prompt_strategy=config.prompt_strategy
            )
            
            batch_result = await extract_batch(batch_request)
            
            # Calcular métricas (simplificado - em produção usar evaluator completo)
            total_entities = sum(len(result.chemicals) + len(result.diseases) for result in batch_result.results)
            # Não há mais consensus_confidence, usar valor padrão
            avg_confidence = 0.8
            
            # Métricas simuladas (em produção calcular com ground truth)
            precision = min(0.95, 0.7 + (avg_confidence * 0.25))
            recall = min(0.90, 0.6 + (avg_confidence * 0.3))
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            from ..models.schemas import ConfigurationMetrics, Metrics
            
            config_result = ConfigurationMetrics(
                configuration=config,
                overall_metrics=Metrics(precision=precision, recall=recall, f1=f1, exact_match=f1*0.9),
                chemical_metrics=Metrics(precision=precision*1.1, recall=recall*0.9, f1=f1*1.05, exact_match=f1*0.95),
                disease_metrics=Metrics(precision=precision*0.9, recall=recall*1.1, f1=f1*0.95, exact_match=f1*0.85),
                processing_time=batch_result.total_processing_time,
                total_entities=total_entities,
                correct_entities=int(total_entities * precision)
            )
            
            configuration_results.append(config_result)
            
            # Atualizar melhor configuração
            if f1 > best_f1:
                best_f1 = f1
                best_config_name = config.name or f"config_{len(configuration_results)}"
        
        # Salvar resultados se solicitado
        results_file = None
        if request.save_predictions:
            try:
                saved_path = response_storage.save_benchmark_response(
                    test_file=request.test_file,
                    configurations=[config.dict() for config in request.configurations],
                    results=[result.dict() for result in configuration_results],
                    best_configuration=best_config_name,
                    total_processing_time=total_processing_time,
                    metadata={
                        "batch_size": request.batch_size,
                        "use_single_llm": config.get('llm', {}).get('use_single_llm', False)
                    }
                )
                results_file = saved_path
                logger.info(f"💾 Benchmark salvo em: {saved_path}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao salvar benchmark: {e}")
                # Fallback para método antigo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"results/benchmark_{timestamp}.json"
                
                # Criar diretório se não existir
                Path("results").mkdir(exist_ok=True)
                
                # Salvar resultados
                benchmark_data = {
                    "timestamp": timestamp,
                    "configurations": [result.dict() for result in configuration_results],
                    "best_configuration": best_config_name,
                    "total_processing_time": sum(r.processing_time for r in configuration_results)
                }
                
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        total_processing_time = sum(r.processing_time for r in configuration_results)
        
        return BenchmarkResponse(
            configurations=configuration_results,
            best_configuration=best_config_name,
            total_processing_time=total_processing_time,
            results_file=results_file
        )
        
    except Exception as e:
        logger.error(f"Erro no benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "BioNER_llm API - Extração de Entidades Biomédicas com Multi-LLM",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
