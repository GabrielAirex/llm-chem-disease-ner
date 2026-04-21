"""
Sistema de consenso para combinar predições de múltiplos LLMs
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
import statistics

from ..models.schemas import Entity, EntityType, Relation, ConsensusMethod, ModelPrediction
from ..llm.llm_manager import LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Resultado do consenso entre modelos"""
    entities: List[Entity]
    relations: List[Relation]
    consensus_confidence: float
    agreement_score: float
    model_votes: Dict[str, int]
    processing_time: float


class ConsensusEngine:
    """Engine para combinar predições de múltiplos modelos"""
    
    def __init__(self, confidence_threshold: float = 0.7, weight_threshold: float = 0.5):
        """
        Inicializa o engine de consenso
        
        Args:
            confidence_threshold: Limite mínimo de confiança
            weight_threshold: Limite mínimo de peso para votação ponderada
        """
        self.confidence_threshold = confidence_threshold
        self.weight_threshold = weight_threshold
    
    def combine_predictions(
        self,
        predictions: List[ModelPrediction],
        method: ConsensusMethod,
        min_models: int = 2
    ) -> ConsensusResult:
        """
        Combina predições usando método de consenso
        
        Args:
            predictions: Lista de predições dos modelos
            method: Método de consenso
            min_models: Número mínimo de modelos para consenso
            
        Returns:
            Resultado do consenso
        """
        if len(predictions) < min_models:
            logger.warning(f"Apenas {len(predictions)} predições disponíveis, mínimo: {min_models}")
            return self._create_empty_result()
        
        # Filtrar predições válidas
        valid_predictions = [p for p in predictions if p.entities]
        if not valid_predictions:
            logger.warning("Nenhuma predição válida encontrada")
            return self._create_empty_result()
        
        logger.info(f"Combinando {len(valid_predictions)} predições usando método {method.value}")
        
        # Aplicar método de consenso
        if method == ConsensusMethod.SIMPLE:
            return self._simple_voting(valid_predictions)
        elif method == ConsensusMethod.WEIGHTED:
            return self._weighted_voting(valid_predictions)
        elif method == ConsensusMethod.CASCADE:
            return self._cascade_consensus(valid_predictions)
        elif method == ConsensusMethod.ALL:
            return self._all_predictions(valid_predictions)
        elif method == ConsensusMethod.CROSS_REVIEW:
            return self._cross_review_consensus(valid_predictions)
        elif method == ConsensusMethod.ITERATIVE:
            return self._iterative_consensus(valid_predictions)
        elif method == ConsensusMethod.ENSEMBLE:
            return self._ensemble_consensus(valid_predictions)
        elif method == ConsensusMethod.MAJORITY_PLUS:
            return self._majority_plus_consensus(valid_predictions)
        else:
            raise ValueError(f"Método de consenso não suportado: {method}")
    
    def _simple_voting(self, predictions: List[ModelPrediction]) -> ConsensusResult:
        """
        Votação simples por maioria
        
        Args:
            predictions: Lista de predições
            
        Returns:
            Resultado do consenso
        """
        # Coletar todas as entidades
        all_entities = []
        for pred in predictions:
            all_entities.extend(pred.entities)
        
        # Agrupar entidades similares
        entity_groups = self._group_similar_entities(all_entities)
        
        # Votar por entidade
        consensus_entities = []
        model_votes = defaultdict(int)
        
        for group in entity_groups:
            # Contar votos para cada entidade no grupo
            entity_votes = Counter()
            for entity in group:
                entity_key = f"{entity.text}_{entity.type.value}"
                entity_votes[entity_key] += 1
                model_votes[entity_key] += 1
            
            # Entidade com mais votos
            if entity_votes:
                best_entity_key = entity_votes.most_common(1)[0][0]
                vote_count = entity_votes[best_entity_key]
                
                # Só incluir se tiver maioria
                if vote_count > len(predictions) / 2:
                    # Encontrar a entidade original
                    for entity in group:
                        if f"{entity.text}_{entity.type.value}" == best_entity_key:
                            # Ajustar confiança baseada no número de votos
                            consensus_entity = Entity(
                                text=entity.text,
                                start=entity.start,
                                end=entity.end,
                                type=entity.type,
                                confidence=min(0.95, 0.7 + (vote_count / len(predictions)) * 0.25),
                                mesh_id=entity.mesh_id
                            )
                            consensus_entities.append(consensus_entity)
                            break
        
        # Calcular métricas de consenso
        agreement_score = self._calculate_agreement_score(predictions)
        consensus_confidence = statistics.mean([p.confidence for p in predictions])
        
        return ConsensusResult(
            entities=consensus_entities,
            relations=[],  # TODO: Implementar consenso para relações
            consensus_confidence=consensus_confidence,
            agreement_score=agreement_score,
            model_votes=dict(model_votes),
            processing_time=sum(p.processing_time for p in predictions)
        )
    
    def _weighted_voting(self, predictions: List[ModelPrediction]) -> ConsensusResult:
        """
        Votação ponderada por confiança
        
        Args:
            predictions: Lista de predições
            
        Returns:
            Resultado do consenso
        """
        # Coletar todas as entidades com pesos
        weighted_entities = []
        for pred in predictions:
            weight = pred.confidence
            for entity in pred.entities:
                weighted_entities.append((entity, weight))
        
        # Agrupar entidades similares
        entity_groups = self._group_similar_entities([e for e, _ in weighted_entities])
        
        # Votar por entidade com pesos
        consensus_entities = []
        model_votes = defaultdict(float)
        
        for group in entity_groups:
            # Calcular peso total para cada entidade no grupo
            entity_weights = defaultdict(float)
            for entity in group:
                entity_key = f"{entity.text}_{entity.type.value}"
                # Encontrar peso da entidade
                for orig_entity, weight in weighted_entities:
                    if (orig_entity.text == entity.text and 
                        orig_entity.type == entity.type and
                        orig_entity.start == entity.start):
                        entity_weights[entity_key] += weight
                        model_votes[entity_key] += weight
                        break
            
            # Entidade com maior peso
            if entity_weights:
                best_entity_key = max(entity_weights, key=entity_weights.get)
                total_weight = entity_weights[best_entity_key]
                
                # Só incluir se peso for suficiente
                if total_weight >= self.weight_threshold:
                    # Encontrar a entidade original
                    for entity in group:
                        if f"{entity.text}_{entity.type.value}" == best_entity_key:
                            # Confiança baseada no peso total
                            consensus_entity = Entity(
                                text=entity.text,
                                start=entity.start,
                                end=entity.end,
                                type=entity.type,
                                confidence=min(0.95, total_weight),
                                mesh_id=entity.mesh_id
                            )
                            consensus_entities.append(consensus_entity)
                            break
        
        # Calcular métricas de consenso
        agreement_score = self._calculate_agreement_score(predictions)
        consensus_confidence = statistics.mean([p.confidence for p in predictions])
        
        return ConsensusResult(
            entities=consensus_entities,
            relations=[],  # TODO: Implementar consenso para relações
            consensus_confidence=consensus_confidence,
            agreement_score=agreement_score,
            model_votes=dict(model_votes),
            processing_time=sum(p.processing_time for p in predictions)
        )
    
    def _cascade_consensus(self, predictions: List[ModelPrediction]) -> ConsensusResult:
        """
        Consenso em cascata: usar modelo com maior confiança
        
        Args:
            predictions: Lista de predições
            
        Returns:
            Resultado do consenso
        """
        # Ordenar por confiança
        sorted_predictions = sorted(predictions, key=lambda p: p.confidence, reverse=True)
        
        # Usar predição com maior confiança
        best_prediction = sorted_predictions[0]
        
        # Se confiança for baixa, tentar combinar com segunda melhor
        if best_prediction.confidence < self.confidence_threshold and len(sorted_predictions) > 1:
            second_best = sorted_predictions[1]
            # Combinar entidades das duas melhores predições
            combined_entities = self._merge_entities(
                best_prediction.entities, 
                second_best.entities
            )
            
            return ConsensusResult(
                entities=combined_entities,
                relations=best_prediction.relations,
                consensus_confidence=(best_prediction.confidence + second_best.confidence) / 2,
                agreement_score=0.8,  # Assumir boa concordância
                model_votes={best_prediction.model_name: 1, second_best.model_name: 1},
                processing_time=best_prediction.processing_time + second_best.processing_time
            )
        else:
            return ConsensusResult(
                entities=best_prediction.entities,
                relations=best_prediction.relations,
                consensus_confidence=best_prediction.confidence,
                agreement_score=1.0,  # Uma única predição
                model_votes={best_prediction.model_name: 1},
                processing_time=best_prediction.processing_time
            )
    
    def _all_predictions(self, predictions: List[ModelPrediction]) -> ConsensusResult:
        """
        Retorna todas as predições sem consenso
        
        Args:
            predictions: Lista de predições
            
        Returns:
            Resultado com todas as predições
        """
        # Combinar todas as entidades
        all_entities = []
        for pred in predictions:
            all_entities.extend(pred.entities)
        
        # Remover duplicatas
        unique_entities = self._remove_duplicate_entities(all_entities)
        
        return ConsensusResult(
            entities=unique_entities,
            relations=[],  # TODO: Implementar para relações
            consensus_confidence=statistics.mean([p.confidence for p in predictions]),
            agreement_score=0.5,  # Concordância média
            model_votes={pred.model_name: 1 for pred in predictions},
            processing_time=sum(p.processing_time for p in predictions)
        )
    
    def _group_similar_entities(self, entities: List[Entity]) -> List[List[Entity]]:
        """
        Agrupa entidades similares (mesmo texto e tipo)
        
        Args:
            entities: Lista de entidades
            
        Returns:
            Lista de grupos de entidades similares
        """
        groups = defaultdict(list)
        
        for entity in entities:
            # Chave baseada em texto e tipo
            key = f"{entity.text.lower()}_{entity.type.value}"
            groups[key].append(entity)
        
        return list(groups.values())
    
    def _merge_entities(self, entities1: List[Entity], entities2: List[Entity]) -> List[Entity]:
        """
        Combina duas listas de entidades removendo duplicatas
        
        Args:
            entities1: Primeira lista de entidades
            entities2: Segunda lista de entidades
            
        Returns:
            Lista combinada sem duplicatas
        """
        all_entities = entities1 + entities2
        return self._remove_duplicate_entities(all_entities)
    
    def _remove_duplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove entidades duplicadas
        
        Args:
            entities: Lista de entidades
            
        Returns:
            Lista sem duplicatas
        """
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.text.lower(), entity.type.value, entity.start, entity.end)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _calculate_agreement_score(self, predictions: List[ModelPrediction]) -> float:
        """
        Calcula score de concordância entre predições
        
        Args:
            predictions: Lista de predições
            
        Returns:
            Score de concordância (0.0 a 1.0)
        """
        if len(predictions) < 2:
            return 1.0
        
        # Coletar todas as entidades
        all_entities = []
        for pred in predictions:
            all_entities.extend(pred.entities)
        
        # Agrupar entidades similares
        entity_groups = self._group_similar_entities(all_entities)
        
        # Calcular concordância
        total_entities = len(all_entities)
        if total_entities == 0:
            return 1.0
        
        agreed_entities = 0
        for group in entity_groups:
            # Se entidade aparece em múltiplas predições, há concordância
            if len(group) > 1:
                agreed_entities += len(group)
        
        return agreed_entities / total_entities if total_entities > 0 else 1.0
    
    def _cross_review_consensus(self, predictions: List[ModelPrediction]) -> ConsensusResult:
        """
        Consenso por revisão cruzada: LLMs analisam respostas umas das outras
        
        Args:
            predictions: Lista de predições
            
        Returns:
            Resultado do consenso após revisão cruzada
        """
        logger.info("Aplicando consenso por revisão cruzada")
        
        # Primeiro, obter consenso inicial usando weighted voting
        initial_consensus = self._weighted_voting(predictions)
        
        # Se não há entidades no consenso inicial, retornar
        if not initial_consensus.entities:
            return initial_consensus
        
        # Criar prompt para revisão cruzada
        review_prompt = self._create_cross_review_prompt(predictions, initial_consensus)
        
        # TODO: Implementar chamada para LLMs revisarem as respostas
        # Por enquanto, retornar consenso inicial
        logger.info("Revisão cruzada implementada (placeholder)")
        
        return initial_consensus
    
    def _iterative_consensus(self, predictions: List[ModelPrediction]) -> ConsensusResult:
        """
        Consenso iterativo: refinamento em múltiplas rodadas
        
        Args:
            predictions: Lista de predições
            
        Returns:
            Resultado do consenso iterativo
        """
        logger.info("Aplicando consenso iterativo")
        
        # Rodada 1: Consenso básico
        round1 = self._weighted_voting(predictions)
        
        # Rodada 2: Refinamento baseado na rodada 1
        if round1.entities:
            # Filtrar predições que concordam com o consenso da rodada 1
            agreeing_predictions = []
            for pred in predictions:
                if self._prediction_agrees_with_consensus(pred, round1.entities):
                    agreeing_predictions.append(pred)
            
            # Se há predições concordantes, refinar consenso
            if agreeing_predictions:
                round2 = self._weighted_voting(agreeing_predictions)
                # Usar o melhor resultado entre as duas rodadas
                if len(round2.entities) > len(round1.entities):
                    return round2
        
        return round1
    
    def _ensemble_consensus(self, predictions: List[ModelPrediction]) -> ConsensusResult:
        """
        Consenso por ensemble: combina múltiplos métodos
        
        Args:
            predictions: Lista de predições
            
        Returns:
            Resultado do consenso por ensemble
        """
        logger.info("Aplicando consenso por ensemble")
        
        # Aplicar múltiplos métodos
        simple_result = self._simple_voting(predictions)
        weighted_result = self._weighted_voting(predictions)
        cascade_result = self._cascade_consensus(predictions)
        
        # Combinar resultados
        all_entities = []
        all_entities.extend(simple_result.entities)
        all_entities.extend(weighted_result.entities)
        all_entities.extend(cascade_result.entities)
        
        # Remover duplicatas e calcular consenso final
        unique_entities = self._remove_duplicate_entities(all_entities)
        
        # Calcular confiança média
        avg_confidence = (simple_result.consensus_confidence + 
                         weighted_result.consensus_confidence + 
                         cascade_result.consensus_confidence) / 3
        
        return ConsensusResult(
            entities=unique_entities,
            relations=[],
            consensus_confidence=avg_confidence,
            agreement_score=0.8,  # Assumir boa concordância no ensemble
            model_votes={pred.model_name: 1 for pred in predictions},
            processing_time=sum(p.processing_time for p in predictions)
        )
    
    def _majority_plus_consensus(self, predictions: List[ModelPrediction]) -> ConsensusResult:
        """
        Consenso por maioria com validação adicional
        
        Args:
            predictions: Lista de predições
            
        Returns:
            Resultado do consenso com validação
        """
        logger.info("Aplicando consenso por maioria com validação")
        
        # Primeiro, aplicar votação simples
        simple_result = self._simple_voting(predictions)
        
        # Validar entidades com critérios adicionais
        validated_entities = []
        for entity in simple_result.entities:
            if self._validate_entity_quality(entity, predictions):
                validated_entities.append(entity)
        
        # Ajustar confiança baseada na validação
        if validated_entities:
            validation_score = len(validated_entities) / len(simple_result.entities)
            adjusted_confidence = simple_result.consensus_confidence * validation_score
        else:
            adjusted_confidence = simple_result.consensus_confidence * 0.5
        
        return ConsensusResult(
            entities=validated_entities,
            relations=simple_result.relations,
            consensus_confidence=adjusted_confidence,
            agreement_score=simple_result.agreement_score,
            model_votes=simple_result.model_votes,
            processing_time=simple_result.processing_time
        )
    
    def _create_cross_review_prompt(self, predictions: List[ModelPrediction], consensus: ConsensusResult) -> str:
        """
        Cria prompt para revisão cruzada
        
        Args:
            predictions: Lista de predições
            consensus: Consenso inicial
            
        Returns:
            Prompt para revisão cruzada
        """
        prompt = "Analise as seguintes extrações de entidades biomédicas de diferentes modelos:\n\n"
        
        for i, pred in enumerate(predictions, 1):
            prompt += f"Modelo {i} ({pred.model_name}):\n"
            if pred.entities:
                for entity in pred.entities:
                    prompt += f"  - {entity.text} ({entity.type.value}) - confiança: {entity.confidence:.2f}\n"
            else:
                prompt += "  - Nenhuma entidade encontrada\n"
            prompt += "\n"
        
        prompt += f"Consenso inicial: {len(consensus.entities)} entidades\n"
        if consensus.entities:
            for entity in consensus.entities:
                prompt += f"  - {entity.text} ({entity.type.value}) - confiança: {entity.confidence:.2f}\n"
        
        prompt += "\nAvalie a qualidade das extrações e sugira melhorias."
        
        return prompt
    
    def _prediction_agrees_with_consensus(self, prediction: ModelPrediction, consensus_entities: List[Entity]) -> bool:
        """
        Verifica se uma predição concorda com o consenso
        
        Args:
            prediction: Predição a verificar
            consensus_entities: Entidades do consenso
            
        Returns:
            True se concorda, False caso contrário
        """
        if not prediction.entities or not consensus_entities:
            return False
        
        # Verificar se há sobreposição significativa
        consensus_texts = {e.text.lower() for e in consensus_entities}
        prediction_texts = {e.text.lower() for e in prediction.entities}
        
        overlap = len(consensus_texts.intersection(prediction_texts))
        total = len(consensus_texts.union(prediction_texts))
        
        return overlap / total > 0.5 if total > 0 else False
    
    def _validate_entity_quality(self, entity: Entity, predictions: List[ModelPrediction]) -> bool:
        """
        Valida qualidade de uma entidade
        
        Args:
            entity: Entidade a validar
            predictions: Predições originais
            
        Returns:
            True se entidade é de boa qualidade
        """
        # Critérios de validação
        if entity.confidence < 0.5:
            return False
        
        if len(entity.text) < 3:
            return False
        
        # Verificar se entidade aparece em múltiplas predições
        appearances = 0
        for pred in predictions:
            for pred_entity in pred.entities:
                if (pred_entity.text.lower() == entity.text.lower() and 
                    pred_entity.type == entity.type):
                    appearances += 1
                    break
        
        return appearances >= 1  # Pelo menos uma aparição

    def _create_empty_result(self) -> ConsensusResult:
        """Cria resultado vazio"""
        return ConsensusResult(
            entities=[],
            relations=[],
            consensus_confidence=0.0,
            agreement_score=0.0,
            model_votes={},
            processing_time=0.0
        )
    
    def analyze_consensus_quality(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        """
        Analisa qualidade do consenso
        
        Args:
            predictions: Lista de predições
            
        Returns:
            Análise da qualidade do consenso
        """
        if not predictions:
            return {"error": "Nenhuma predição disponível"}
        
        # Estatísticas básicas
        total_entities = sum(len(p.entities) for p in predictions)
        avg_entities = total_entities / len(predictions)
        
        # Variação no número de entidades
        entity_counts = [len(p.entities) for p in predictions]
        entity_variance = statistics.variance(entity_counts) if len(entity_counts) > 1 else 0
        
        # Concordância
        agreement_score = self._calculate_agreement_score(predictions)
        
        # Confiança média
        avg_confidence = statistics.mean([p.confidence for p in predictions])
        
        # Análise por tipo de entidade
        chemical_count = sum(len([e for e in p.entities if e.type == EntityType.CHEMICAL]) for p in predictions)
        disease_count = sum(len([e for e in p.entities if e.type == EntityType.DISEASE]) for p in predictions)
        
        return {
            "total_predictions": len(predictions),
            "total_entities": total_entities,
            "avg_entities_per_prediction": round(avg_entities, 2),
            "entity_variance": round(entity_variance, 2),
            "agreement_score": round(agreement_score, 3),
            "avg_confidence": round(avg_confidence, 3),
            "chemical_entities": chemical_count,
            "disease_entities": disease_count,
            "consensus_quality": "high" if agreement_score > 0.8 else "medium" if agreement_score > 0.5 else "low"
        }


def test_consensus_engine():
    """Função de teste para o Consensus Engine"""
    from ..models.schemas import ModelPrediction, Entity, EntityType
    
    engine = ConsensusEngine()
    
    # Criar predições de teste
    pred1 = ModelPrediction(
        model_name="llama3.2",
        entities=[
            Entity(text="lithium carbonate", start=0, end=15, type=EntityType.CHEMICAL, confidence=0.9, mesh_id="D016651"),
            Entity(text="toxicity", start=16, end=24, type=EntityType.DISEASE, confidence=0.85, mesh_id="D064420")
        ],
        relations=[],
        processing_time=2.1,
        confidence=0.9
    )
    
    pred2 = ModelPrediction(
        model_name="mixtral",
        entities=[
            Entity(text="lithium carbonate", start=0, end=15, type=EntityType.CHEMICAL, confidence=0.95, mesh_id="D016651"),
            Entity(text="toxicity", start=16, end=24, type=EntityType.DISEASE, confidence=0.8, mesh_id="D064420")
        ],
        relations=[],
        processing_time=1.8,
        confidence=0.88
    )
    
    pred3 = ModelPrediction(
        model_name="phi3",
        entities=[
            Entity(text="lithium", start=0, end=7, type=EntityType.CHEMICAL, confidence=0.7, mesh_id="D008094"),
            Entity(text="toxicity", start=16, end=24, type=EntityType.DISEASE, confidence=0.75, mesh_id="D064420")
        ],
        relations=[],
        processing_time=1.5,
        confidence=0.75
    )
    
    predictions = [pred1, pred2, pred3]
    
    print("🧪 Testando Consensus Engine:")
    
    # Testar diferentes métodos
    for method in ConsensusMethod:
        print(f"\n📊 Método: {method.value}")
        result = engine.combine_predictions(predictions, method)
        
        print(f"  Entidades: {len(result.entities)}")
        print(f"  Confiança: {result.consensus_confidence:.3f}")
        print(f"  Concordância: {result.agreement_score:.3f}")
        print(f"  Tempo: {result.processing_time:.2f}s")
        
        for entity in result.entities:
            print(f"    - {entity.text} ({entity.type.value}) - {entity.confidence:.2f}")
    
    # Análise de qualidade
    print("\n📈 Análise de Qualidade:")
    quality = engine.analyze_consensus_quality(predictions)
    for key, value in quality.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_consensus_engine()
