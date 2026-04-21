"""
Schemas Pydantic para a API BioNER_llm
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class EntityType(str, Enum):
    """Tipos de entidades"""
    CHEMICAL = "Chemical"
    DISEASE = "Disease"


class PromptStrategy(str, Enum):
    """Estratégias de prompt"""
    ZERO_SHOT = "zero-shot"
    FEW_SHOT = "few-shot"
    CHAIN_OF_THOUGHT = "chain-of-thought"


class ConsensusMethod(str, Enum):
    """Métodos de consenso"""
    SIMPLE = "simple"
    WEIGHTED = "weighted"
    CASCADE = "cascade"
    ALL = "all"


class Entity(BaseModel):
    """Entidade extraída do texto"""
    text: str = Field(..., description="Texto da entidade")
    start: int = Field(..., description="Posição inicial no texto")
    end: int = Field(..., description="Posição final no texto")
    type: EntityType = Field(..., description="Tipo da entidade")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiança da extração")
    mesh_id: Optional[str] = Field(None, description="MeSH ID da entidade")


class Relation(BaseModel):
    """Relação entre entidades"""
    chemical: str = Field(..., description="Texto do químico")
    disease: str = Field(..., description="Texto da doença")
    chemical_mesh_id: Optional[str] = Field(None, description="MeSH ID do químico")
    disease_mesh_id: Optional[str] = Field(None, description="MeSH ID da doença")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiança da relação")
    relation_type: str = Field(default="CID", description="Tipo de relação")


class ModelPrediction(BaseModel):
    """Predição de um modelo individual"""
    model_name: str = Field(..., description="Nome do modelo")
    entities: List[Entity] = Field(default_factory=list, description="Entidades extraídas")
    relations: List[Relation] = Field(default_factory=list, description="Relações extraídas")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiança geral")
    processing_time: float = Field(..., description="Tempo de processamento em segundos")
    error: Optional[str] = Field(None, description="Erro se houver")


class LLMConfig(BaseModel):
    """Configuração de LLM"""
    use_single_llm: bool = Field(default=False, description="Usar apenas uma LLM")
    single_llm_model: Optional[str] = Field(None, description="Modelo quando use_single_llm=True")
    multi_llm_models: Optional[List[str]] = Field(None, description="Modelos para múltiplas LLMs")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperatura")
    max_tokens: Optional[int] = Field(None, ge=1, description="Máximo de tokens")


class ExtractionRequest(BaseModel):
    """Request para extração de entidades"""
    text: str = Field(..., description="Texto para extrair entidades")
    pmid: Optional[str] = Field(None, description="PMID do artigo (opcional, não enviado ao LLM)")
    models: List[str] = Field(default=["llama3.2:3b"], description="Modelos a usar")
    prompt_strategy: PromptStrategy = Field(default=PromptStrategy.FEW_SHOT, description="Estratégia de prompt")
    consensus_method: ConsensusMethod = Field(default=ConsensusMethod.WEIGHTED, description="Método de consenso")
    return_all_predictions: bool = Field(default=False, description="Retornar todas as predições individuais")
    llm_config: Optional[LLMConfig] = Field(None, description="Configuração de LLM")
    use_positions: Optional[bool] = Field(None, description="Se True, extrai posições (start/end). Se None, usa config.yaml")
    num_examples: Optional[int] = Field(None, ge=1, le=32, description="Número de exemplos para few-shot (1-32). Se None, usa config.yaml")
    max_text_length: Optional[int] = Field(None, ge=1, description="Comprimento máximo do texto. Se None, usa config.yaml")
    prompt_type: Optional[str] = Field(default="type1", description="Tipo de prompt (type1: mais restritivo, type2: mais inclusivo). Padrão: type1")
    include_raw_response: Optional[bool] = Field(default=False, description="Incluir resposta bruta do LLM na resposta")


class ExtractionResponse(BaseModel):
    """Response de extração de entidades"""
    chemicals: List[Entity] = Field(default_factory=list, description="Químicos extraídos")
    diseases: List[Entity] = Field(default_factory=list, description="Doenças extraídas")


class BatchExtractionRequest(BaseModel):
    """Request para extração em lote"""
    texts: List[str] = Field(..., description="Lista de textos para processar")
    models: List[str] = Field(default=["llama3.2:3b"], description="Modelos a usar")
    prompt_strategy: PromptStrategy = Field(default=PromptStrategy.FEW_SHOT, description="Estratégia de prompt")
    consensus_method: ConsensusMethod = Field(default=ConsensusMethod.WEIGHTED, description="Método de consenso")
    use_positions: Optional[bool] = Field(None, description="Se True, extrai posições (start/end). Se None, usa config.yaml")
    num_examples: Optional[int] = Field(None, ge=1, le=32, description="Número de exemplos para few-shot (1-32). Se None, usa config.yaml")
    max_text_length: Optional[int] = Field(None, ge=1, description="Comprimento máximo do texto. Se None, usa config.yaml")
    prompt_type: Optional[str] = Field(default="type1", description="Tipo de prompt (type1: mais restritivo, type2: mais inclusivo). Padrão: type1")


class BatchExtractionResponse(BaseModel):
    """Response de extração em lote"""
    results: List[ExtractionResponse] = Field(default_factory=list, description="Resultados por texto")
    success_count: int = Field(..., description="Número de sucessos")
    error_count: int = Field(..., description="Número de erros")
    total_processing_time: float = Field(..., description="Tempo total de processamento")


class BenchmarkConfiguration(BaseModel):
    """Configuração para benchmark"""
    name: str = Field(..., description="Nome da configuração")
    models: List[str] = Field(..., description="Modelos a usar")
    prompt_strategy: PromptStrategy = Field(..., description="Estratégia de prompt")
    consensus_method: ConsensusMethod = Field(default=ConsensusMethod.WEIGHTED, description="Método de consenso")


class Metrics(BaseModel):
    """Métricas de avaliação"""
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall")
    f1: float = Field(..., ge=0.0, le=1.0, description="F1-Score")
    exact_match_f1: Optional[float] = Field(None, ge=0.0, le=1.0, description="Exact Match F1")
    partial_match_f1: Optional[float] = Field(None, ge=0.0, le=1.0, description="Partial Match F1")


class ConfigurationMetrics(BaseModel):
    """Métricas de uma configuração"""
    configuration: BenchmarkConfiguration = Field(..., description="Configuração testada")
    overall_metrics: Metrics = Field(..., description="Métricas gerais")
    chemical_metrics: Metrics = Field(..., description="Métricas para químicos")
    disease_metrics: Metrics = Field(..., description="Métricas para doenças")
    processing_time: float = Field(..., description="Tempo de processamento")


class BenchmarkRequest(BaseModel):
    """Request para benchmark"""
    test_file: str = Field(..., description="Arquivo de teste")
    configurations: List[BenchmarkConfiguration] = Field(..., description="Configurações para testar")
    batch_size: int = Field(default=10, ge=1, description="Tamanho do batch")
    save_predictions: bool = Field(default=True, description="Salvar predições")


class BenchmarkResponse(BaseModel):
    """Response de benchmark"""
    configurations: List[ConfigurationMetrics] = Field(default_factory=list, description="Resultados por configuração")
    best_configuration: Optional[str] = Field(None, description="Melhor configuração")
    total_processing_time: float = Field(..., description="Tempo total de processamento")


class HealthResponse(BaseModel):
    """Response de health check"""
    status: str = Field(..., description="Status da API")
    version: str = Field(..., description="Versão da API")
    available_models: List[str] = Field(default_factory=list, description="Modelos disponíveis")
    ollama_status: str = Field(..., description="Status do Ollama")


class ConfigResponse(BaseModel):
    """Response de configuração"""
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="Configuração LLM")
    prompts_config: Dict[str, Any] = Field(default_factory=dict, description="Configuração de prompts")
    consensus_config: Dict[str, Any] = Field(default_factory=dict, description="Configuração de consenso")
    benchmark_config: Dict[str, Any] = Field(default_factory=dict, description="Configuração de benchmark")
    storage_config: Dict[str, Any] = Field(default_factory=dict, description="Configuração de armazenamento")
    audit_config: Dict[str, Any] = Field(default_factory=dict, description="Configuração de auditoria")

