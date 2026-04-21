"""
Sistema de auditoria para métricas F1, Recall e Precision
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricsAuditor:
    """Auditor de métricas para análise de performance"""
    
    def __init__(self, output_dir: str = "audit_results"):
        """
        Inicializa o auditor de métricas
        
        Args:
            output_dir: Diretório para salvar resultados de auditoria
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📊 MetricsAuditor inicializado. Saída em: {self.output_dir}")
    
    def calculate_ner_metrics(
        self,
        predicted_entities: List[Dict[str, Any]],
        ground_truth_entities: List[Dict[str, Any]],
        text: str
    ) -> Dict[str, float]:
        """
        Calcula métricas NER (Precision, Recall, F1)
        
        Args:
            predicted_entities: Entidades preditas
            ground_truth_entities: Entidades ground truth
            text: Texto original
            
        Returns:
            Dicionário com métricas
        """
        # Converter para formato padronizado
        pred_entities = self._normalize_entities(predicted_entities, text)
        gt_entities = self._normalize_entities(ground_truth_entities, text)
        
        # Calcular métricas por tipo
        metrics_by_type = {}
        for entity_type in ['Chemical', 'Disease']:
            pred_type = [e for e in pred_entities if e['type'] == entity_type]
            gt_type = [e for e in gt_entities if e['type'] == entity_type]
            
            metrics_by_type[entity_type] = self._calculate_type_metrics(pred_type, gt_type)
        
        # Calcular métricas gerais
        overall_metrics = self._calculate_type_metrics(pred_entities, gt_entities)
        
        return {
            'overall': overall_metrics,
            'by_type': metrics_by_type,
            'total_predicted': len(pred_entities),
            'total_ground_truth': len(gt_entities)
        }
    
    def _normalize_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Normaliza entidades para cálculo de métricas"""
        normalized = []
        
        for entity in entities:
            # Extrair informações básicas
            entity_text = entity.get('text', '').lower().strip()
            entity_type = entity.get('type', '')
            start = entity.get('start', 0)
            end = entity.get('end', len(entity_text))
            
            # Validar posições
            if start < 0 or end > len(text) or start >= end:
                continue
            
            # Verificar se o texto corresponde
            actual_text = text[start:end].lower().strip()
            if entity_text != actual_text:
                # Tentar encontrar correspondência
                if entity_text in text.lower():
                    start = text.lower().find(entity_text)
                    end = start + len(entity_text)
                else:
                    continue
            
            normalized.append({
                'text': entity_text,
                'type': entity_type,
                'start': start,
                'end': end,
                'confidence': entity.get('confidence', 1.0)
            })
        
        return normalized
    
    def _calculate_type_metrics(
        self,
        predicted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calcula métricas para um tipo específico de entidade"""
        
        if not predicted and not ground_truth:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'exact_match_f1': 1.0}
        
        if not predicted:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'exact_match_f1': 0.0}
        
        if not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'exact_match_f1': 0.0}
        
        # Calcular métricas exact match
        tp_exact, fp_exact, fn_exact = self._calculate_exact_match_metrics(predicted, ground_truth)
        
        # Calcular métricas partial match
        tp_partial, fp_partial, fn_partial = self._calculate_partial_match_metrics(predicted, ground_truth)
        
        # Métricas exact match
        precision_exact = tp_exact / (tp_exact + fp_exact) if (tp_exact + fp_exact) > 0 else 0.0
        recall_exact = tp_exact / (tp_exact + fn_exact) if (tp_exact + fn_exact) > 0 else 0.0
        f1_exact = 2 * precision_exact * recall_exact / (precision_exact + recall_exact) if (precision_exact + recall_exact) > 0 else 0.0
        
        # Métricas partial match
        precision_partial = tp_partial / (tp_partial + fp_partial) if (tp_partial + fp_partial) > 0 else 0.0
        recall_partial = tp_partial / (tp_partial + fn_partial) if (tp_partial + fn_partial) > 0 else 0.0
        f1_partial = 2 * precision_partial * recall_partial / (precision_partial + recall_partial) if (precision_partial + recall_partial) > 0 else 0.0
        
        return {
            'precision': precision_exact,
            'recall': recall_exact,
            'f1': f1_exact,
            'exact_match_f1': f1_exact,
            'partial_match_f1': f1_partial,
            'tp_exact': tp_exact,
            'fp_exact': fp_exact,
            'fn_exact': fn_exact,
            'tp_partial': tp_partial,
            'fp_partial': fp_partial,
            'fn_partial': fn_partial
        }
    
    def _calculate_exact_match_metrics(
        self,
        predicted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Tuple[int, int, int]:
        """Calcula métricas exact match"""
        
        # Criar sets para comparação
        pred_set = set()
        gt_set = set()
        
        for entity in predicted:
            key = (entity['text'], entity['type'], entity['start'], entity['end'])
            pred_set.add(key)
        
        for entity in ground_truth:
            key = (entity['text'], entity['type'], entity['start'], entity['end'])
            gt_set.add(key)
        
        # Calcular métricas
        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        return tp, fp, fn
    
    def _calculate_partial_match_metrics(
        self,
        predicted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Tuple[int, int, int]:
        """Calcula métricas partial match (IoU > 0.5)"""
        
        tp = 0
        fp = 0
        fn = 0
        
        matched_gt = set()
        matched_pred = set()
        
        # Para cada predição, encontrar melhor match no ground truth
        for i, pred_entity in enumerate(predicted):
            best_match = None
            best_iou = 0.0
            
            for j, gt_entity in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                
                # Verificar se tipos coincidem
                if pred_entity['type'] != gt_entity['type']:
                    continue
                
                # Calcular IoU
                iou = self._calculate_iou(pred_entity, gt_entity)
                
                if iou > best_iou and iou > 0.5:  # Threshold para partial match
                    best_match = j
                    best_iou = iou
            
            if best_match is not None:
                tp += 1
                matched_gt.add(best_match)
                matched_pred.add(i)
            else:
                fp += 1
        
        # Calcular false negatives
        fn = len(ground_truth) - len(matched_gt)
        
        return tp, fp, fn
    
    def _calculate_iou(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        """Calcula Intersection over Union para entidades"""
        
        start1, end1 = entity1['start'], entity1['end']
        start2, end2 = entity2['start'], entity2['end']
        
        # Calcular interseção
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)
        
        # Calcular união
        union = (end1 - start1) + (end2 - start2) - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def audit_extraction_results(
        self,
        results_file: str,
        ground_truth_file: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Audita resultados de extração contra ground truth
        
        Args:
            results_file: Arquivo com resultados preditos
            ground_truth_file: Arquivo com ground truth
            output_file: Arquivo de saída (opcional)
            
        Returns:
            Dicionário com métricas de auditoria
        """
        logger.info(f"🔍 Iniciando auditoria: {results_file} vs {ground_truth_file}")
        
        # Carregar resultados
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Carregar ground truth
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # Processar cada resultado
        audit_results = []
        overall_metrics = defaultdict(list)
        
        for result in results_data:
            if 'text' not in result or 'predicted_entities' not in result:
                continue
            
            text = result['text']
            predicted_entities = result['predicted_entities']
            ground_truth_entities = result.get('ground_truth_entities', [])
            
            # Calcular métricas
            metrics = self.calculate_ner_metrics(
                predicted_entities, ground_truth_entities, text
            )
            
            # Adicionar informações do resultado
            audit_result = {
                'text_hash': result.get('text_hash', ''),
                'text': text[:100] + '...' if len(text) > 100 else text,
                'metrics': metrics,
                'timestamp': result.get('timestamp', ''),
                'models_used': result.get('models_used', []),
                'prompt_strategy': result.get('prompt_strategy', ''),
                'consensus_method': result.get('consensus_method', '')
            }
            
            audit_results.append(audit_result)
            
            # Acumular métricas gerais
            overall_metrics['precision'].append(metrics['overall']['precision'])
            overall_metrics['recall'].append(metrics['overall']['recall'])
            overall_metrics['f1'].append(metrics['overall']['f1'])
            overall_metrics['exact_match_f1'].append(metrics['overall']['exact_match_f1'])
            overall_metrics['partial_match_f1'].append(metrics['overall']['partial_match_f1'])
        
        # Calcular métricas agregadas
        aggregated_metrics = {
            'overall': {
                'precision': np.mean(overall_metrics['precision']),
                'recall': np.mean(overall_metrics['recall']),
                'f1': np.mean(overall_metrics['f1']),
                'exact_match_f1': np.mean(overall_metrics['exact_match_f1']),
                'partial_match_f1': np.mean(overall_metrics['partial_match_f1'])
            },
            'std': {
                'precision': np.std(overall_metrics['precision']),
                'recall': np.std(overall_metrics['recall']),
                'f1': np.std(overall_metrics['f1']),
                'exact_match_f1': np.std(overall_metrics['exact_match_f1']),
                'partial_match_f1': np.std(overall_metrics['partial_match_f1'])
            }
        }
        
        # Preparar resultado final
        audit_summary = {
            'timestamp': datetime.now().isoformat(),
            'results_file': results_file,
            'ground_truth_file': ground_truth_file,
            'total_results': len(audit_results),
            'aggregated_metrics': aggregated_metrics,
            'individual_results': audit_results
        }
        
        # Salvar resultado se especificado
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(audit_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Auditoria salva em: {output_path}")
        
        logger.info(f"✅ Auditoria concluída: F1={aggregated_metrics['overall']['f1']:.3f}")
        return audit_summary
    
    def generate_audit_report(
        self,
        audit_results: Dict[str, Any],
        output_file: str
    ) -> str:
        """
        Gera relatório de auditoria em formato legível
        
        Args:
            audit_results: Resultados da auditoria
            output_file: Nome do arquivo de saída
            
        Returns:
            Caminho do arquivo gerado
        """
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Relatório de Auditoria - BioNER_llm\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"**Timestamp**: {audit_results['timestamp']}\n")
            f.write(f"**Arquivo de Resultados**: {audit_results['results_file']}\n")
            f.write(f"**Arquivo Ground Truth**: {audit_results['ground_truth_file']}\n")
            f.write(f"**Total de Resultados**: {audit_results['total_results']}\n\n")
            
            # Métricas agregadas
            f.write("## Métricas Agregadas\n")
            f.write("-" * 30 + "\n\n")
            
            metrics = audit_results['aggregated_metrics']['overall']
            std_metrics = audit_results['aggregated_metrics']['std']
            
            f.write(f"| Métrica | Valor | Desvio Padrão |\n")
            f.write(f"|---------|-------|---------------|\n")
            f.write(f"| Precision | {metrics['precision']:.3f} | {std_metrics['precision']:.3f} |\n")
            f.write(f"| Recall | {metrics['recall']:.3f} | {std_metrics['recall']:.3f} |\n")
            f.write(f"| F1-Score | {metrics['f1']:.3f} | {std_metrics['f1']:.3f} |\n")
            f.write(f"| Exact Match F1 | {metrics['exact_match_f1']:.3f} | {std_metrics['exact_match_f1']:.3f} |\n")
            f.write(f"| Partial Match F1 | {metrics['partial_match_f1']:.3f} | {std_metrics['partial_match_f1']:.3f} |\n\n")
            
            # Análise por configuração
            f.write("## Análise por Configuração\n")
            f.write("-" * 30 + "\n\n")
            
            config_metrics = defaultdict(list)
            for result in audit_results['individual_results']:
                key = f"{result['prompt_strategy']}_{result['consensus_method']}"
                config_metrics[key].append(result['metrics']['overall']['f1'])
            
            for config, f1_scores in config_metrics.items():
                f.write(f"**{config}**: F1={np.mean(f1_scores):.3f} (n={len(f1_scores)})\n")
            
            f.write("\n")
            
            # Top 10 melhores resultados
            f.write("## Top 10 Melhores Resultados\n")
            f.write("-" * 30 + "\n\n")
            
            sorted_results = sorted(
                audit_results['individual_results'],
                key=lambda x: x['metrics']['overall']['f1'],
                reverse=True
            )[:10]
            
            for i, result in enumerate(sorted_results, 1):
                f.write(f"{i}. **F1={result['metrics']['overall']['f1']:.3f}** - {result['text']}\n")
                f.write(f"   - Models: {result['models_used']}\n")
                f.write(f"   - Strategy: {result['prompt_strategy']}\n")
                f.write(f"   - Consensus: {result['consensus_method']}\n\n")
            
            # Top 10 piores resultados
            f.write("## Top 10 Piores Resultados\n")
            f.write("-" * 30 + "\n\n")
            
            sorted_results = sorted(
                audit_results['individual_results'],
                key=lambda x: x['metrics']['overall']['f1']
            )[:10]
            
            for i, result in enumerate(sorted_results, 1):
                f.write(f"{i}. **F1={result['metrics']['overall']['f1']:.3f}** - {result['text']}\n")
                f.write(f"   - Models: {result['models_used']}\n")
                f.write(f"   - Strategy: {result['prompt_strategy']}\n")
                f.write(f"   - Consensus: {result['consensus_method']}\n\n")
        
        logger.info(f"📊 Relatório de auditoria gerado: {output_path}")
        return str(output_path)
    
    def compare_configurations(
        self,
        audit_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compara diferentes configurações
        
        Args:
            audit_results: Resultados da auditoria
            
        Returns:
            Análise comparativa
        """
        config_analysis = defaultdict(lambda: {
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'count': 0
        })
        
        for result in audit_results['individual_results']:
            key = f"{result['prompt_strategy']}_{result['consensus_method']}"
            metrics = result['metrics']['overall']
            
            config_analysis[key]['f1_scores'].append(metrics['f1'])
            config_analysis[key]['precision_scores'].append(metrics['precision'])
            config_analysis[key]['recall_scores'].append(metrics['recall'])
            config_analysis[key]['count'] += 1
        
        # Calcular estatísticas por configuração
        comparison = {}
        for config, data in config_analysis.items():
            comparison[config] = {
                'count': data['count'],
                'f1_mean': np.mean(data['f1_scores']),
                'f1_std': np.std(data['f1_scores']),
                'precision_mean': np.mean(data['precision_scores']),
                'precision_std': np.std(data['precision_scores']),
                'recall_mean': np.mean(data['recall_scores']),
                'recall_std': np.std(data['recall_scores'])
            }
        
        return comparison


def test_metrics_auditor():
    """Função de teste para o Metrics Auditor"""
    auditor = MetricsAuditor()
    
    # Dados de teste
    predicted_entities = [
        {'text': 'lithium carbonate', 'type': 'Chemical', 'start': 0, 'end': 15, 'confidence': 0.9},
        {'text': 'toxicity', 'type': 'Disease', 'start': 16, 'end': 24, 'confidence': 0.8}
    ]
    
    ground_truth_entities = [
        {'text': 'lithium carbonate', 'type': 'Chemical', 'start': 0, 'end': 15},
        {'text': 'toxicity', 'type': 'Disease', 'start': 16, 'end': 24}
    ]
    
    text = "lithium carbonate toxicity in newborn infant"
    
    # Calcular métricas
    metrics = auditor.calculate_ner_metrics(predicted_entities, ground_truth_entities, text)
    
    print("🧪 Testando Metrics Auditor...")
    print(f"✅ Métricas calculadas: F1={metrics['overall']['f1']:.3f}")
    print(f"   Precision: {metrics['overall']['precision']:.3f}")
    print(f"   Recall: {metrics['overall']['recall']:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_metrics_auditor()
