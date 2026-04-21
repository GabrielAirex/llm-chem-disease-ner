"""
Sistema de benchmark e avaliação para extração de entidades biomédicas
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

from ..models.schemas import (
    Entity, EntityType, BenchmarkConfiguration, ConfigurationMetrics, Metrics,
    ModelPrediction, ConsensusMethod
)

# bc5cdr_parser é opcional - não necessário para o fluxo principal
try:
    from ..data.bc5cdr_parser import BC5CDRArticle, BC5CDRParser
except ImportError:
    BC5CDRArticle = None
    BC5CDRParser = None

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Resultado de avaliação de uma configuração"""
    configuration: BenchmarkConfiguration
    overall_metrics: Metrics
    chemical_metrics: Metrics
    disease_metrics: Metrics
    processing_time: float
    total_entities: int
    correct_entities: int
    detailed_results: Dict[str, Any]


@dataclass
class EntityMatch:
    """Match entre entidade predita e ground truth"""
    predicted: Entity
    ground_truth: Entity
    match_type: str  # 'exact', 'partial', 'none'
    confidence: float


class BenchmarkEvaluator:
    """Avaliador para benchmark de extração de entidades"""
    
    def __init__(self, tolerance: int = 5):
        """
        Inicializa o avaliador
        
        Args:
            tolerance: Tolerância em caracteres para match de entidades
        """
        self.tolerance = tolerance
        # Parser é opcional - pode ser None se não disponível
        self.parser = BC5CDRParser() if BC5CDRParser is not None else None
    
    def evaluate_configuration(
        self,
        configuration: BenchmarkConfiguration,
        test_articles: List[BC5CDRArticle],
        predictions: List[List[Entity]],
        processing_time: float
    ) -> EvaluationResult:
        """
        Avalia uma configuração específica
        
        Args:
            configuration: Configuração testada
            test_articles: Artigos de teste
            predictions: Predições para cada artigo
            processing_time: Tempo de processamento
            
        Returns:
            Resultado da avaliação
        """
        logger.info(f"Avaliando configuração: {configuration.name or 'unnamed'}")
        
        # Coletar todas as entidades ground truth e preditas
        all_ground_truth = []
        all_predictions = []
        
        for article, article_predictions in zip(test_articles, predictions):
            all_ground_truth.extend(article.entities)
            all_predictions.extend(article_predictions)
        
        # Calcular métricas gerais
        overall_metrics = self._calculate_metrics(all_ground_truth, all_predictions)
        
        # Calcular métricas por tipo
        chemical_gt = [e for e in all_ground_truth if e.type == EntityType.CHEMICAL]
        chemical_pred = [e for e in all_predictions if e.type == EntityType.CHEMICAL]
        chemical_metrics = self._calculate_metrics(chemical_gt, chemical_pred)
        
        disease_gt = [e for e in all_ground_truth if e.type == EntityType.DISEASE]
        disease_pred = [e for e in all_predictions if e.type == EntityType.DISEASE]
        disease_metrics = self._calculate_metrics(disease_gt, disease_pred)
        
        # Calcular entidades corretas
        correct_entities = self._count_correct_entities(all_ground_truth, all_predictions)
        
        # Resultados detalhados
        detailed_results = self._generate_detailed_results(
            test_articles, predictions, all_ground_truth, all_predictions
        )
        
        return EvaluationResult(
            configuration=configuration,
            overall_metrics=overall_metrics,
            chemical_metrics=chemical_metrics,
            disease_metrics=disease_metrics,
            processing_time=processing_time,
            total_entities=len(all_predictions),
            correct_entities=correct_entities,
            detailed_results=detailed_results
        )
    
    def _calculate_metrics(self, ground_truth: List[Entity], predictions: List[Entity]) -> Metrics:
        """
        Calcula métricas de precisão, recall e F1
        
        Args:
            ground_truth: Entidades ground truth
            predictions: Entidades preditas
            
        Returns:
            Métricas calculadas
        """
        if not ground_truth and not predictions:
            return Metrics(precision=1.0, recall=1.0, f1=1.0, exact_match=1.0)
        
        if not predictions:
            return Metrics(precision=0.0, recall=0.0, f1=0.0, exact_match=0.0)
        
        if not ground_truth:
            return Metrics(precision=0.0, recall=1.0, f1=0.0, exact_match=0.0)
        
        # Encontrar matches
        matches = self._find_entity_matches(ground_truth, predictions)
        
        # Calcular métricas
        true_positives = len([m for m in matches if m.match_type in ['exact', 'partial']])
        false_positives = len(predictions) - true_positives
        false_negatives = len(ground_truth) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Exact match (apenas matches exatos)
        exact_matches = len([m for m in matches if m.match_type == 'exact'])
        exact_match = exact_matches / len(ground_truth) if ground_truth else 0.0
        
        return Metrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            exact_match=round(exact_match, 4)
        )
    
    def _find_entity_matches(self, ground_truth: List[Entity], predictions: List[Entity]) -> List[EntityMatch]:
        """
        Encontra matches entre entidades preditas e ground truth
        
        Args:
            ground_truth: Entidades ground truth
            predictions: Entidades preditas
            
        Returns:
            Lista de matches encontrados
        """
        matches = []
        used_predictions = set()
        
        for gt_entity in ground_truth:
            best_match = None
            best_score = 0
            
            for i, pred_entity in enumerate(predictions):
                if i in used_predictions:
                    continue
                
                # Verificar se tipos são iguais
                if gt_entity.type != pred_entity.type:
                    continue
                
                # Calcular score de match
                score = self._calculate_match_score(gt_entity, pred_entity)
                
                if score > best_score and score > 0.5:  # Threshold mínimo
                    best_score = score
                    best_match = (i, pred_entity)
            
            if best_match:
                i, pred_entity = best_match
                used_predictions.add(i)
                
                # Determinar tipo de match
                match_type = self._determine_match_type(gt_entity, pred_entity)
                
                match = EntityMatch(
                    predicted=pred_entity,
                    ground_truth=gt_entity,
                    match_type=match_type,
                    confidence=best_score
                )
                matches.append(match)
        
        return matches
    
    def _calculate_match_score(self, gt_entity: Entity, pred_entity: Entity) -> float:
        """
        Calcula score de match entre duas entidades
        
        Args:
            gt_entity: Entidade ground truth
            pred_entity: Entidade predita
            
        Returns:
            Score de match (0.0 a 1.0)
        """
        # Verificar sobreposição de posições
        overlap = self._calculate_overlap(gt_entity, pred_entity)
        if overlap == 0:
            return 0.0
        
        # Verificar similaridade de texto
        text_similarity = self._calculate_text_similarity(gt_entity.text, pred_entity.text)
        
        # Score combinado
        score = (overlap * 0.7) + (text_similarity * 0.3)
        return min(1.0, score)
    
    def _calculate_overlap(self, gt_entity: Entity, pred_entity: Entity) -> float:
        """
        Calcula sobreposição entre duas entidades
        
        Args:
            gt_entity: Entidade ground truth
            pred_entity: Entidade predita
            
        Returns:
            Score de sobreposição (0.0 a 1.0)
        """
        # Calcular interseção
        start_max = max(gt_entity.start, pred_entity.start)
        end_min = min(gt_entity.end, pred_entity.end)
        
        if start_max >= end_min:
            return 0.0
        
        intersection = end_min - start_max
        
        # Calcular união
        union = max(gt_entity.end, pred_entity.end) - min(gt_entity.start, pred_entity.start)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similaridade entre dois textos
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            Score de similaridade (0.0 a 1.0)
        """
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()
        
        # Match exato
        if text1_lower == text2_lower:
            return 1.0
        
        # Verificar se um contém o outro
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return 0.8
        
        # Similaridade por palavras
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_match_type(self, gt_entity: Entity, pred_entity: Entity) -> str:
        """
        Determina tipo de match entre entidades
        
        Args:
            gt_entity: Entidade ground truth
            pred_entity: Entidade predita
            
        Returns:
            Tipo de match ('exact', 'partial', 'none')
        """
        # Verificar match exato
        if (gt_entity.text.lower() == pred_entity.text.lower() and
            abs(gt_entity.start - pred_entity.start) <= self.tolerance and
            abs(gt_entity.end - pred_entity.end) <= self.tolerance):
            return 'exact'
        
        # Verificar match parcial
        overlap = self._calculate_overlap(gt_entity, pred_entity)
        if overlap > 0.5:
            return 'partial'
        
        return 'none'
    
    def _count_correct_entities(self, ground_truth: List[Entity], predictions: List[Entity]) -> int:
        """
        Conta entidades corretas (exact match)
        
        Args:
            ground_truth: Entidades ground truth
            predictions: Entidades preditas
            
        Returns:
            Número de entidades corretas
        """
        matches = self._find_entity_matches(ground_truth, predictions)
        return len([m for m in matches if m.match_type == 'exact'])
    
    def _generate_detailed_results(
        self,
        test_articles: List[BC5CDRArticle],
        predictions: List[List[Entity]],
        all_ground_truth: List[Entity],
        all_predictions: List[Entity]
    ) -> Dict[str, Any]:
        """
        Gera resultados detalhados da avaliação
        
        Args:
            test_articles: Artigos de teste
            predictions: Predições por artigo
            all_ground_truth: Todas as entidades ground truth
            all_predictions: Todas as entidades preditas
            
        Returns:
            Resultados detalhados
        """
        # Estatísticas por artigo
        article_stats = []
        for i, (article, article_predictions) in enumerate(zip(test_articles, predictions)):
            article_metrics = self._calculate_metrics(article.entities, article_predictions)
            article_stats.append({
                'article_id': article.article_id,
                'text_length': len(article.full_text),
                'gt_entities': len(article.entities),
                'pred_entities': len(article_predictions),
                'precision': article_metrics.precision,
                'recall': article_metrics.recall,
                'f1': article_metrics.f1
            })
        
        # Análise por tipo de entidade
        chemical_analysis = self._analyze_entity_type(
            [e for e in all_ground_truth if e.type == EntityType.CHEMICAL],
            [e for e in all_predictions if e.type == EntityType.CHEMICAL]
        )
        
        disease_analysis = self._analyze_entity_type(
            [e for e in all_ground_truth if e.type == EntityType.DISEASE],
            [e for e in all_predictions if e.type == EntityType.DISEASE]
        )
        
        # Análise de erros
        error_analysis = self._analyze_errors(all_ground_truth, all_predictions)
        
        return {
            'article_statistics': article_stats,
            'chemical_analysis': chemical_analysis,
            'disease_analysis': disease_analysis,
            'error_analysis': error_analysis,
            'summary': {
                'total_articles': len(test_articles),
                'total_gt_entities': len(all_ground_truth),
                'total_pred_entities': len(all_predictions),
                'avg_entities_per_article': len(all_ground_truth) / len(test_articles) if test_articles else 0
            }
        }
    
    def _analyze_entity_type(self, gt_entities: List[Entity], pred_entities: List[Entity]) -> Dict[str, Any]:
        """
        Analisa performance por tipo de entidade
        
        Args:
            gt_entities: Entidades ground truth
            pred_entities: Entidades preditas
            
        Returns:
            Análise do tipo de entidade
        """
        matches = self._find_entity_matches(gt_entities, pred_entities)
        
        # Análise de matches
        exact_matches = [m for m in matches if m.match_type == 'exact']
        partial_matches = [m for m in matches if m.match_type == 'partial']
        
        # Entidades mais comuns
        gt_texts = [e.text.lower() for e in gt_entities]
        pred_texts = [e.text.lower() for e in pred_entities]
        
        gt_common = Counter(gt_texts).most_common(10)
        pred_common = Counter(pred_texts).most_common(10)
        
        return {
            'total_gt': len(gt_entities),
            'total_pred': len(pred_entities),
            'exact_matches': len(exact_matches),
            'partial_matches': len(partial_matches),
            'gt_most_common': gt_common,
            'pred_most_common': pred_common,
            'match_rate': len(matches) / len(gt_entities) if gt_entities else 0.0
        }
    
    def _analyze_errors(self, ground_truth: List[Entity], predictions: List[Entity]) -> Dict[str, Any]:
        """
        Analisa erros na predição
        
        Args:
            ground_truth: Entidades ground truth
            predictions: Entidades preditas
            
        Returns:
            Análise de erros
        """
        matches = self._find_entity_matches(ground_truth, predictions)
        matched_gt = {m.ground_truth for m in matches}
        matched_pred = {m.predicted for m in matches}
        
        # Entidades não encontradas (false negatives)
        missed_entities = [e for e in ground_truth if e not in matched_gt]
        
        # Entidades incorretas (false positives)
        incorrect_entities = [e for e in predictions if e not in matched_pred]
        
        # Análise de tipos de erro
        missed_chemicals = [e for e in missed_entities if e.type == EntityType.CHEMICAL]
        missed_diseases = [e for e in missed_entities if e.type == EntityType.DISEASE]
        
        incorrect_chemicals = [e for e in incorrect_entities if e.type == EntityType.CHEMICAL]
        incorrect_diseases = [e for e in incorrect_entities if e.type == EntityType.DISEASE]
        
        return {
            'missed_entities': len(missed_entities),
            'incorrect_entities': len(incorrect_entities),
            'missed_chemicals': len(missed_chemicals),
            'missed_diseases': len(missed_diseases),
            'incorrect_chemicals': len(incorrect_chemicals),
            'incorrect_diseases': len(incorrect_diseases),
            'missed_entity_examples': [e.text for e in missed_entities[:5]],
            'incorrect_entity_examples': [e.text for e in incorrect_entities[:5]]
        }
    
    def generate_report(self, evaluation_results: List[EvaluationResult], output_path: str):
        """
        Gera relatório detalhado dos resultados
        
        Args:
            evaluation_results: Resultados das avaliações
            output_path: Caminho para salvar o relatório
        """
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_configurations': len(evaluation_results),
            'configurations': []
        }
        
        for result in evaluation_results:
            config_data = {
                'name': result.configuration.name,
                'models': result.configuration.models,
                'prompt_strategy': result.configuration.prompt_strategy.value,
                'consensus_method': result.configuration.consensus_method.value if result.configuration.consensus_method else None,
                'overall_metrics': result.overall_metrics.dict(),
                'chemical_metrics': result.chemical_metrics.dict(),
                'disease_metrics': result.disease_metrics.dict(),
                'processing_time': result.processing_time,
                'total_entities': result.total_entities,
                'correct_entities': result.correct_entities,
                'detailed_results': result.detailed_results
            }
            report['configurations'].append(config_data)
        
        # Encontrar melhor configuração
        best_config = max(evaluation_results, key=lambda r: r.overall_metrics.f1)
        report['best_configuration'] = {
            'name': best_config.configuration.name,
            'f1_score': best_config.overall_metrics.f1,
            'precision': best_config.overall_metrics.precision,
            'recall': best_config.overall_metrics.recall
        }
        
        # Salvar relatório
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Relatório salvo em {output_path}")
    
    def export_to_csv(self, evaluation_results: List[EvaluationResult], output_path: str):
        """
        Exporta resultados para CSV
        
        Args:
            evaluation_results: Resultados das avaliações
            output_path: Caminho para salvar o CSV
        """
        data = []
        
        for result in evaluation_results:
            row = {
                'configuration_name': result.configuration.name,
                'models': '|'.join(result.configuration.models),
                'prompt_strategy': result.configuration.prompt_strategy.value,
                'consensus_method': result.configuration.consensus_method.value if result.configuration.consensus_method else None,
                'overall_precision': result.overall_metrics.precision,
                'overall_recall': result.overall_metrics.recall,
                'overall_f1': result.overall_metrics.f1,
                'overall_exact_match': result.overall_metrics.exact_match,
                'chemical_precision': result.chemical_metrics.precision,
                'chemical_recall': result.chemical_metrics.recall,
                'chemical_f1': result.chemical_metrics.f1,
                'disease_precision': result.disease_metrics.precision,
                'disease_recall': result.disease_metrics.recall,
                'disease_f1': result.disease_metrics.f1,
                'processing_time': result.processing_time,
                'total_entities': result.total_entities,
                'correct_entities': result.correct_entities
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Resultados exportados para CSV: {output_path}")


def test_evaluator():
    """Função de teste para o Benchmark Evaluator"""
    from ..models.schemas import Entity, EntityType, BenchmarkConfiguration, PromptStrategy
    
    evaluator = BenchmarkEvaluator()
    
    # Criar dados de teste
    gt_entities = [
        Entity(text="lithium carbonate", start=0, end=15, type=EntityType.CHEMICAL, confidence=1.0, mesh_id="D016651"),
        Entity(text="toxicity", start=16, end=24, type=EntityType.DISEASE, confidence=1.0, mesh_id="D064420")
    ]
    
    pred_entities = [
        Entity(text="lithium carbonate", start=0, end=15, type=EntityType.CHEMICAL, confidence=0.95, mesh_id="D016651"),
        Entity(text="toxicity", start=16, end=24, type=EntityType.DISEASE, confidence=0.90, mesh_id="D064420"),
        Entity(text="newborn", start=25, end=32, type=EntityType.DISEASE, confidence=0.7)  # False positive
    ]
    
    # Testar métricas
    metrics = evaluator._calculate_metrics(gt_entities, pred_entities)
    print("🧪 Testando Benchmark Evaluator:")
    print(f"  Precision: {metrics.precision}")
    print(f"  Recall: {metrics.recall}")
    print(f"  F1: {metrics.f1}")
    print(f"  Exact Match: {metrics.exact_match}")
    
    # Testar matches
    matches = evaluator._find_entity_matches(gt_entities, pred_entities)
    print(f"\n  Matches encontrados: {len(matches)}")
    for match in matches:
        print(f"    - {match.ground_truth.text} -> {match.predicted.text} ({match.match_type})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_evaluator()
