"""
Gerenciador para modelos Hugging Face no HPC
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

logger = logging.getLogger(__name__)


class HuggingFaceManager:
    """Gerenciador para modelos Hugging Face no HPC"""
    
    def __init__(self, hpc_host: str = "http://localhost:11434", timeout: int = 60):
        """
        Inicializa o gerenciador Hugging Face
        
        Args:
            hpc_host: Host do HPC (via túnel SSH)
            timeout: Timeout em segundos
        """
        self.hpc_host = hpc_host
        self.timeout = timeout
        self.available_models: List[str] = []
        self.loaded_models: Dict[str, Any] = {}
        
        # Verificar conexão com HPC
        self._check_hpc_connection()
        
        logger.info("🤗 HuggingFaceManager inicializado para HPC")
    
    def _check_hpc_connection(self):
        """Verifica conexão com HPC"""
        try:
            # Tentar conectar com o HPC
            response = requests.get(f"{self.hpc_host}/health", timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ Conectado ao HPC em {self.hpc_host}")
            else:
                logger.warning(f"⚠️ HPC respondeu com status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar com HPC: {e}")
            logger.error("💡 Verifique se o túnel SSH está ativo: ssh -L 11434:localhost:11434 user@hpc")
    
    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis"""
        return self.available_models.copy()
    
    def load_model(self, model_name: str, task: str = "ner") -> bool:
        """
        Carrega um modelo Hugging Face
        
        Args:
            model_name: Nome do modelo
            task: Tipo de tarefa (ner, text-classification, etc.)
            
        Returns:
            True se carregado com sucesso
        """
        try:
            logger.info(f"🔄 Carregando modelo {model_name}...")
            
            # Carregar pipeline
            pipe = pipeline(
                task,
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            self.loaded_models[model_name] = pipe
            self.available_models.append(model_name)
            
            logger.info(f"✅ Modelo {model_name} carregado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo {model_name}: {e}")
            return False
    
    def generate_with_model(
        self,
        model_name: str,
        text: str,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Gera resposta usando modelo Hugging Face
        
        Args:
            model_name: Nome do modelo
            text: Texto para processar
            prompt: Prompt para o modelo
            task: Tipo de tarefa
            
        Returns:
            Resposta do modelo
        """
        start_time = time.time()
        
        try:
            if model_name not in self.loaded_models:
                if not self.load_model(model_name):
                    return {
                        "error": f"Modelo {model_name} não pôde ser carregado",
                        "processing_time": 0.0
                    }
            
            model = self.loaded_models[model_name]
            
            # Processar texto
            if hasattr(model, 'predict'):
                # Para modelos de classificação
                result = model.predict(text)
            else:
                # Para modelos de NER
                result = model(text)
            
            processing_time = time.time() - start_time
            
            # Formatar resposta
            response_text = self._format_hf_response(result, text)
            
            return {
                "response_text": response_text,
                "processing_time": processing_time,
                "confidence": self._calculate_confidence(result),
                "tokens_used": len(text.split()),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na geração com {model_name}: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _format_hf_response(self, result: Any, text: str) -> str:
        """Formata resposta do Hugging Face"""
        try:
            if isinstance(result, list):
                # Para NER
                entities = []
                for item in result:
                    if isinstance(item, dict) and 'entity' in item:
                        entities.append({
                            "text": item.get('word', ''),
                            "type": item.get('entity', ''),
                            "start": item.get('start', 0),
                            "end": item.get('end', 0),
                            "confidence": item.get('score', 0.0)
                        })
                
                return json.dumps({
                    "entities": entities,
                    "relations": []
                })
            else:
                # Para outros tipos de modelo
                return json.dumps({
                    "entities": [],
                    "relations": [],
                    "raw_response": str(result)
                })
                
        except Exception as e:
            logger.error(f"❌ Erro ao formatar resposta: {e}")
            return json.dumps({
                "entities": [],
                "relations": [],
                "error": str(e)
            })
    
    def _calculate_confidence(self, result: Any) -> float:
        """Calcula confiança da resposta"""
        try:
            if isinstance(result, list):
                scores = [item.get('score', 0.0) for item in result if isinstance(item, dict)]
                return np.mean(scores) if scores else 0.0
            else:
                return 0.5  # Confiança padrão
        except:
            return 0.0
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do gerenciador"""
        return {
            "status": "healthy",
            "hpc_host": self.hpc_host,
            "available_models": self.available_models,
            "loaded_models": list(self.loaded_models.keys()),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def unload_model(self, model_name: str) -> bool:
        """
        Descarrega um modelo
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            True se descarregado com sucesso
        """
        try:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                self.available_models.remove(model_name)
                logger.info(f"✅ Modelo {model_name} descarregado")
                return True
            else:
                logger.warning(f"⚠️ Modelo {model_name} não está carregado")
                return False
        except Exception as e:
            logger.error(f"❌ Erro ao descarregar modelo {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Obtém informações de um modelo
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Informações do modelo
        """
        if model_name not in self.loaded_models:
            return {"error": "Modelo não carregado"}
        
        try:
            model = self.loaded_models[model_name]
            return {
                "model_name": model_name,
                "type": type(model).__name__,
                "device": str(model.device) if hasattr(model, 'device') else "unknown",
                "loaded": True
            }
        except Exception as e:
            return {"error": str(e)}


def test_huggingface_manager():
    """Função de teste para o Hugging Face Manager"""
    manager = HuggingFaceManager()
    
    print("🧪 Testando HuggingFace Manager...")
    
    # Verificar saúde
    health = manager.health_check()
    print(f"✅ Status: {health['status']}")
    print(f"   HPC Host: {health['hpc_host']}")
    print(f"   CUDA Disponível: {health['cuda_available']}")
    
    # Testar carregamento de modelo
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    if manager.load_model(model_name):
        print(f"✅ Modelo {model_name} carregado com sucesso")
        
        # Testar geração
        result = manager.generate_with_model(
            model_name=model_name,
            text="Lithium carbonate toxicity in newborn infant",
            prompt="Extract medical entities"
        )
        
        if result.get('error'):
            print(f"❌ Erro na geração: {result['error']}")
        else:
            print(f"✅ Geração bem-sucedida: {result['processing_time']:.2f}s")
            print(f"   Confiança: {result['confidence']:.3f}")
    else:
        print(f"❌ Erro ao carregar modelo {model_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_huggingface_manager()
