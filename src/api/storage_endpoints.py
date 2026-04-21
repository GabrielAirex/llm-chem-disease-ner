"""
Endpoints para gerenciamento de armazenamento de respostas
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

from ..storage.response_storage import ResponseStorage
from ..models.schemas import ConfigResponse

logger = logging.getLogger(__name__)

# Router para endpoints de armazenamento
storage_router = APIRouter(prefix="/storage", tags=["storage"])

# Instância global do storage (será inicializada na main)
response_storage: Optional[ResponseStorage] = None


@storage_router.get("/responses")
async def list_responses(
    response_type: Optional[str] = Query(None, description="Tipo de resposta (extraction, batch, benchmark)"),
    limit: int = Query(50, description="Número máximo de respostas a retornar")
):
    """Lista respostas salvas"""
    if not response_storage:
        raise HTTPException(status_code=503, detail="Response storage não inicializado")
    
    try:
        responses = response_storage.list_saved_responses(response_type)
        return {
            "total": len(responses),
            "responses": responses[:limit]
        }
    except Exception as e:
        logger.error(f"Erro ao listar respostas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@storage_router.get("/responses/{filename}")
async def get_response_info(filename: str):
    """Obtém informações de uma resposta específica"""
    if not response_storage:
        raise HTTPException(status_code=503, detail="Response storage não inicializado")
    
    try:
        file_path = response_storage.output_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Arquivo não encontrado")
        
        info = response_storage.get_response_info(str(file_path))
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter informações da resposta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@storage_router.delete("/responses/{filename}")
async def delete_response(filename: str):
    """Remove uma resposta específica"""
    if not response_storage:
        raise HTTPException(status_code=503, detail="Response storage não inicializado")
    
    try:
        file_path = response_storage.output_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Arquivo não encontrado")
        
        file_path.unlink()
        return {"message": f"Arquivo {filename} removido com sucesso"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao remover resposta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@storage_router.post("/cleanup")
async def cleanup_old_responses(days: int = Query(30, description="Número de dias para manter arquivos")):
    """Remove respostas antigas"""
    if not response_storage:
        raise HTTPException(status_code=503, detail="Response storage não inicializado")
    
    try:
        removed_count = response_storage.cleanup_old_responses(days)
        return {
            "message": f"Limpeza concluída",
            "removed_files": removed_count,
            "days": days
        }
    except Exception as e:
        logger.error(f"Erro na limpeza: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@storage_router.get("/stats")
async def get_storage_stats():
    """Obtém estatísticas do armazenamento"""
    if not response_storage:
        raise HTTPException(status_code=503, detail="Response storage não inicializado")
    
    try:
        # Listar todos os arquivos
        all_responses = response_storage.list_saved_responses()
        
        # Contar por tipo
        extraction_count = len([r for r in all_responses if r["filename"].startswith("extraction_")])
        batch_count = len([r for r in all_responses if r["filename"].startswith("batch_")])
        benchmark_count = len([r for r in all_responses if r["filename"].startswith("benchmark_")])
        
        # Calcular tamanho total
        total_size = sum(r["size"] for r in all_responses)
        
        return {
            "total_files": len(all_responses),
            "extraction_files": extraction_count,
            "batch_files": batch_count,
            "benchmark_files": benchmark_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "output_directory": str(response_storage.output_dir)
        }
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def init_storage_endpoints(storage: ResponseStorage):
    """Inicializa os endpoints de armazenamento"""
    global response_storage
    response_storage = storage
