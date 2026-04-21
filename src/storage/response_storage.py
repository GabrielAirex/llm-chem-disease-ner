"""
Sistema de armazenamento de respostas da API BioNER_llm
"""

import json
import os
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResponseStorage:
    """Gerenciador de armazenamento de respostas"""
    
    def __init__(self, output_dir: str = "indicios_encontrados", create_dir: bool = True):
        """
        Inicializa o gerenciador de armazenamento
        
        Args:
            output_dir: Diretório para salvar respostas
            create_dir: Se True, cria o diretório se não existir
        """
        self.output_dir = Path(output_dir)
        self.create_dir = create_dir
        
        # Criar diretório se necessário
        if self.create_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Diretório criado: {self.output_dir}")
    
    def save_extraction_response(
        self,
        text: str,
        pmid: Optional[str] = None,
        entities: list = None,
        relations: list = None,
        models_used: list = None,
        prompt_strategy: str = None,
        num_examples: Optional[int] = None,
        processing_time: float = None
    ) -> str:
        """
        Salva resposta de extração de entidades
        
        Args:
            text: Texto analisado
            pmid: PMID do artigo (opcional)
            entities: Entidades extraídas
            relations: Relações extraídas
            models_used: Modelos utilizados
            prompt_strategy: Estratégia de prompt
            processing_time: Tempo de processamento
            
        Returns:
            Caminho do arquivo salvo
        """
        # Gerar hash do texto para identificação única (apenas para nome do arquivo)
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Limpar entidades: remover mesh_id e confidence
        cleaned_entities = []
        for entity in (entities or []):
            cleaned_entity = {
                "text": entity.get("text", ""),
                "start": entity.get("start", 0),
                "end": entity.get("end", 0),
                "type": entity.get("type", "")
            }
            cleaned_entities.append(cleaned_entity)
        
        # Preparar dados para salvar
        response_data = {
            "pmid": pmid,  # Incluir pmid
            "text": text,
            "entities": cleaned_entities,
            "relations": relations or [],
            "models_used": models_used or [],
            "prompt_strategy": prompt_strategy,
            "num_examples": num_examples,  # Incluir num_examples se few-shot
            "processing_time": processing_time
        }
        
        # Gerar nome do arquivo
        filename = f"extraction_{timestamp}_{text_hash}.json"
        file_path = self.output_dir / filename
        
        # Salvar arquivo
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Resposta salva em: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar resposta: {e}")
            raise
    
    def save_batch_response(
        self,
        texts: list,
        results: list,
        models_used: list,
        prompt_strategy: str,
        total_processing_time: float,
        success_count: int,
        error_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Salva resposta de extração em lote
        
        Args:
            texts: Lista de textos analisados
            results: Lista de resultados
            models_used: Modelos utilizados
            prompt_strategy: Estratégia de prompt
            total_processing_time: Tempo total de processamento
            success_count: Número de sucessos
            error_count: Número de erros
            metadata: Metadados adicionais (opcional)
            
        Returns:
            Caminho do arquivo salvo
        """
        # Gerar hash dos textos (apenas para nome do arquivo)
        texts_hash = hashlib.md5(str(texts).encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Preparar dados para salvar
        batch_data = {
            "texts": texts,
            "results": results,
            "models_used": models_used,
            "prompt_strategy": prompt_strategy,
            "total_processing_time": total_processing_time,
            "success_count": success_count,
            "error_count": error_count
        }
        
        # Gerar nome do arquivo
        filename = f"batch_extraction_{timestamp}_{texts_hash}.json"
        file_path = self.output_dir / filename
        
        # Salvar arquivo
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Resposta em lote salva em: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar resposta em lote: {e}")
            raise
    
    def save_benchmark_response(
        self,
        test_file: str,
        configurations: list,
        results: list,
        best_configuration: str,
        total_processing_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Salva resposta de benchmark
        
        Args:
            test_file: Arquivo de teste usado
            configurations: Configurações testadas
            results: Resultados do benchmark
            best_configuration: Melhor configuração
            total_processing_time: Tempo total de processamento
            metadata: Metadados adicionais (opcional)
            
        Returns:
            Caminho do arquivo salvo
        """
        # Gerar timestamp apenas para nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Preparar dados para salvar
        benchmark_data = {
            "test_file": test_file,
            "configurations": configurations,
            "results": results,
            "best_configuration": best_configuration,
            "total_processing_time": total_processing_time
        }
        
        # Gerar nome do arquivo
        filename = f"benchmark_{timestamp}.json"
        file_path = self.output_dir / filename
        
        # Salvar arquivo
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Benchmark salvo em: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar benchmark: {e}")
            raise
    
    def list_saved_responses(self, response_type: Optional[str] = None) -> list:
        """
        Lista respostas salvas
        
        Args:
            response_type: Tipo de resposta ('extraction', 'batch', 'benchmark')
            
        Returns:
            Lista de arquivos salvos
        """
        if not self.output_dir.exists():
            return []
        
        files = []
        for file_path in self.output_dir.glob("*.json"):
            if response_type is None or file_path.name.startswith(response_type):
                files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return sorted(files, key=lambda x: x["modified"], reverse=True)
    
    def get_response_info(self, file_path: str) -> Dict[str, Any]:
        """
        Obtém informações de uma resposta salva
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Informações da resposta
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                "models_used": data.get("models_used"),
                "prompt_strategy": data.get("prompt_strategy"),
                "processing_time": data.get("processing_time"),
                "entities_count": len(data.get("entities", [])),
                "relations_count": len(data.get("relations", []))
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao ler arquivo {file_path}: {e}")
            return {}
    
    def cleanup_old_responses(self, days: int = 30) -> int:
        """
        Remove respostas antigas
        
        Args:
            days: Número de dias para manter arquivos
            
        Returns:
            Número de arquivos removidos
        """
        if not self.output_dir.exists():
            return 0
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        removed_count = 0
        
        for file_path in self.output_dir.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    removed_count += 1
                    logger.info(f"🗑️ Arquivo antigo removido: {file_path.name}")
                except Exception as e:
                    logger.error(f"❌ Erro ao remover arquivo {file_path.name}: {e}")
        
        return removed_count


def test_response_storage():
    """Função de teste para o Response Storage"""
    storage = ResponseStorage()
    
    # Testar salvamento de extração
    print("🧪 Testando Response Storage...")
    
    # Dados de teste
    test_data = {
        "text": "Lithium carbonate toxicity in newborn infant.",
        "entities": [
            {
                "text": "lithium carbonate",
                "start": 0,
                "end": 15,
                "type": "Chemical",
                "confidence": 0.95,
                "mesh_id": "D016651"
            }
        ],
        "relations": [],
        "models_used": ["llama3.2:3b"],
        "prompt_strategy": "few-shot",
        "consensus_method": "weighted",
        "processing_time": 2.5,
        "consensus_confidence": 0.92
    }
    
    # Salvar resposta
    file_path = storage.save_extraction_response(**test_data)
    print(f"✅ Resposta salva em: {file_path}")
    
    # Listar respostas
    responses = storage.list_saved_responses("extraction")
    print(f"📁 Respostas encontradas: {len(responses)}")
    
    # Obter informações
    if responses:
        info = storage.get_response_info(responses[0]["path"])
        print(f"📊 Informações: {info}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_response_storage()
