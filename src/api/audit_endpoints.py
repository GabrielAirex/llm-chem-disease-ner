"""
Endpoints para auditoria de métricas
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pathlib import Path

from ..audit.metrics_auditor import MetricsAuditor

logger = logging.getLogger(__name__)

# Router para endpoints de auditoria
audit_router = APIRouter(prefix="/audit", tags=["audit"])

# Instância global do auditor (será inicializada na main)
metrics_auditor: Optional[MetricsAuditor] = None


@audit_router.post("/run")
async def run_audit(
    results_file: str,
    ground_truth_file: str,
    output_file: Optional[str] = None,
    generate_report: bool = True
):
    """Executa auditoria de métricas"""
    if not metrics_auditor:
        raise HTTPException(status_code=503, detail="Metrics auditor não inicializado")
    
    try:
        # Verificar arquivos
        if not Path(results_file).exists():
            raise HTTPException(status_code=404, detail=f"Arquivo de resultados não encontrado: {results_file}")
        
        if not Path(ground_truth_file).exists():
            raise HTTPException(status_code=404, detail=f"Arquivo de ground truth não encontrado: {ground_truth_file}")
        
        # Executar auditoria
        audit_results = metrics_auditor.audit_extraction_results(
            results_file=results_file,
            ground_truth_file=ground_truth_file,
            output_file=output_file
        )
        
        # Gerar relatório se solicitado
        report_path = None
        if generate_report:
            report_filename = f"audit_report_{Path(results_file).stem}.md"
            report_path = metrics_auditor.generate_audit_report(audit_results, report_filename)
        
        return {
            "status": "success",
            "audit_results": audit_results,
            "report_path": report_path,
            "message": "Auditoria executada com sucesso"
        }
        
    except Exception as e:
        logger.error(f"Erro na auditoria: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@audit_router.get("/results")
async def list_audit_results():
    """Lista resultados de auditoria"""
    if not metrics_auditor:
        raise HTTPException(status_code=503, detail="Metrics auditor não inicializado")
    
    try:
        audit_dir = metrics_auditor.output_dir
        if not audit_dir.exists():
            return {"results": [], "total": 0}
        
        results = []
        for file_path in audit_dir.glob("*.json"):
            results.append({
                "filename": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime
            })
        
        return {
            "results": sorted(results, key=lambda x: x["modified"], reverse=True),
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"Erro ao listar resultados: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@audit_router.get("/results/{filename}")
async def get_audit_result(filename: str):
    """Obtém resultado de auditoria específico"""
    if not metrics_auditor:
        raise HTTPException(status_code=503, detail="Metrics auditor não inicializado")
    
    try:
        file_path = metrics_auditor.output_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Arquivo não encontrado")
        
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
        
    except Exception as e:
        logger.error(f"Erro ao obter resultado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@audit_router.get("/compare")
async def compare_configurations(
    results_file: str,
    metric: str = Query("f1", description="Métrica para comparação")
):
    """Compara diferentes configurações"""
    if not metrics_auditor:
        raise HTTPException(status_code=503, detail="Metrics auditor não inicializado")
    
    try:
        if not Path(results_file).exists():
            raise HTTPException(status_code=404, detail="Arquivo não encontrado")
        
        import json
        with open(results_file, 'r', encoding='utf-8') as f:
            audit_data = json.load(f)
        
        comparison = metrics_auditor.compare_configurations(audit_data)
        
        # Ordenar por métrica especificada
        sorted_configs = sorted(
            comparison.items(),
            key=lambda x: x[1].get(f'{metric}_mean', 0),
            reverse=True
        )
        
        return {
            "metric": metric,
            "configurations": sorted_configs,
            "best_configuration": sorted_configs[0] if sorted_configs else None
        }
        
    except Exception as e:
        logger.error(f"Erro na comparação: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@audit_router.get("/stats")
async def get_audit_stats():
    """Obtém estatísticas de auditoria"""
    if not metrics_auditor:
        raise HTTPException(status_code=503, detail="Metrics auditor não inicializado")
    
    try:
        audit_dir = metrics_auditor.output_dir
        if not audit_dir.exists():
            return {
                "total_audits": 0,
                "total_size_mb": 0,
                "output_directory": str(audit_dir)
            }
        
        total_audits = len(list(audit_dir.glob("*.json")))
        total_size = sum(f.stat().st_size for f in audit_dir.glob("*"))
        
        return {
            "total_audits": total_audits,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "output_directory": str(audit_dir)
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def init_audit_endpoints(auditor: MetricsAuditor):
    """Inicializa os endpoints de auditoria"""
    global metrics_auditor
    metrics_auditor = auditor
