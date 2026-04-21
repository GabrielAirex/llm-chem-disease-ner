"""
Sistema de prompts para extração de entidades biomédicas
"""

import yaml
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from ..models.schemas import PromptStrategy, Entity, EntityType, Relation

# bc5cdr_parser é opcional - não necessário para o fluxo principal
try:
    from ..data.bc5cdr_parser import BC5CDRParser, BC5CDRArticle
except ImportError:
    BC5CDRParser = None
    BC5CDRArticle = None

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Template de prompt"""
    name: str
    template: str
    description: str
    variables: List[str]


class PromptEngine:
    """Engine para geração de prompts com diferentes estratégias"""
    
    def __init__(self, prompts_config_path: Optional[str] = None, prompt_type: str = "type1"):
        """
        Inicializa o engine de prompts
        
        Args:
            prompts_config_path: Caminho para arquivo de configuração de prompts (opcional)
            prompt_type: Tipo de prompt (type1 ou type2). Padrão: type1
        """
        self.prompt_type = prompt_type
        
        # Determinar caminho do arquivo de configuração
        if prompts_config_path is None:
            # Tentar encontrar o arquivo relativo ao diretório do projeto
            # O arquivo está em: onconavegation-BioNER_llm/config/prompts_type1.yaml ou prompts_type2.yaml
            # Este módulo está em: onconavegation-BioNER_llm/src/prompts/prompt_engine.py
            current_file = Path(__file__)  # src/prompts/prompt_engine.py
            project_root = current_file.parent.parent.parent  # Volta para onconavegation-BioNER_llm
            prompts_config_path = project_root / "config" / f"prompts_{prompt_type}.yaml"
        
        self.config_path = Path(prompts_config_path)
        self.templates: Dict[str, PromptTemplate] = {}
        self.bc5cdr_parser: Optional[BC5CDRParser] = None
        self.examples_config_path: Optional[Path] = None
        self.examples_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Determinar caminho do arquivo de exemplos
        if prompts_config_path:
            config_dir = Path(prompts_config_path).parent
            self.examples_config_path = config_dir / f"examples_{prompt_type}.yaml"
        else:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            self.examples_config_path = project_root / "config" / f"examples_{prompt_type}.yaml"
        
        # Carregar templates
        self._load_templates()
        
        # Carregar exemplos
        self._load_examples()
    
    def _load_templates(self):
        """Carrega templates de prompts do arquivo YAML"""
        try:
            # Verificar se o arquivo existe
            if not self.config_path.exists():
                logger.warning(f"Arquivo de configuração não encontrado: {self.config_path}")
                logger.info("Usando templates padrão (hardcoded)")
                self._load_default_templates()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Carregar templates
            for name, template_data in config.items():
                if isinstance(template_data, str):
                    self.templates[name] = PromptTemplate(
                        name=name,
                        template=template_data,
                        description=f"Template {name}",
                        variables=self._extract_variables(template_data)
                    )
            
            logger.info(f"Carregados {len(self.templates)} templates de prompt")
            
        except Exception as e:
            logger.error(f"Erro ao carregar templates: {e}")
            # Templates padrão em caso de erro
            self._load_default_templates()
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extrai variáveis do template (formato {variavel})"""
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        return list(set(variables))
    
    def _load_default_templates(self):
        """Carrega templates padrão"""
        self.templates = {
            'zero_shot': PromptTemplate(
                name='zero_shot',
                template="""Você é um especialista em NLP biomédico. Analise o texto abaixo e extraia:

1. SUBSTÂNCIAS QUÍMICAS (Chemical): medicamentos, compostos, drogas
2. DOENÇAS (Disease): condições médicas, sintomas, diagnósticos

Para cada entidade encontrada, forneça:
- Texto da entidade
- Posição inicial e final no texto
- Tipo (Chemical ou Disease)
- Confiança (0.0 a 1.0)
- ID MeSH se conhecido

Texto: {text}

Responda em formato JSON:
{{
  "entities": [
    {{
      "text": "nome da entidade",
      "start": posição_inicial,
      "end": posição_final,
      "type": "Chemical" ou "Disease",
      "confidence": 0.95,
      "mesh_id": "D123456"
    }}
  ]
}}""",
                description="Prompt zero-shot para extração de entidades",
                variables=['text']
            ),
            
            'few_shot': PromptTemplate(
                name='few_shot',
                template="""Você é um especialista em NLP biomédico. Analise o texto abaixo e extraia entidades biomédicas.

EXEMPLOS:
{examples}

Agora analise este texto:
{text}

Responda em formato JSON:
{{
  "entities": [
    {{
      "text": "nome da entidade",
      "start": posição_inicial,
      "end": posição_final,
      "type": "Chemical" ou "Disease",
      "confidence": 0.95,
      "mesh_id": "D123456"
    }}
  ]
}}""",
                description="Prompt few-shot com exemplos",
                variables=['examples', 'text']
            ),
            
            'chain_of_thought': PromptTemplate(
                name='chain_of_thought',
                template="""Você é um especialista em NLP biomédico. Analise o texto abaixo seguindo estes passos:

1. PRIMEIRO: Identifique todas as menções de substâncias químicas (medicamentos, compostos)
2. SEGUNDO: Identifique todas as menções de doenças (condições médicas, sintomas)
3. TERCEIRO: Para cada entidade, determine a posição exata no texto
4. QUARTO: Avalie sua confiança na identificação (0.0 a 1.0)

Texto: {text}

RAZOCÍNIO:
[Descreva seu processo de identificação passo-a-passo]

RESULTADO FINAL em JSON:
{{
  "entities": [
    {{
      "text": "nome da entidade",
      "start": posição_inicial,
      "end": posição_final,
      "type": "Chemical" ou "Disease",
      "confidence": 0.95,
      "mesh_id": "D123456"
    }}
  ]
}}""",
                description="Prompt chain-of-thought com raciocínio",
                variables=['text']
            )
        }
    
    def generate_prompt(
        self, 
        text: str, 
        strategy: PromptStrategy,
        examples: Optional[List[Dict[str, Any]]] = None,
        max_text_length: int = 4000,
        use_positions: bool = True,
        num_examples: Optional[int] = None
    ) -> str:
        """
        Gera prompt baseado na estratégia
        
        Args:
            text: Texto para análise
            strategy: Estratégia de prompt
            examples: Exemplos para few-shot (opcional)
            max_text_length: Comprimento máximo do texto
            use_positions: Se True, usa templates com posições (start/end)
            num_examples: Número de exemplos para few-shot (usado para selecionar template)
            
        Returns:
            Prompt formatado
        """
        # Truncar texto se muito longo
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
            logger.warning(f"Texto truncado para {max_text_length} caracteres")
        
        # Construir nome do template baseado na estratégia e configurações
        # Sempre usa no_positions (with_positions foi removido)
        positions_suffix = "no_positions"
        
        if strategy == PromptStrategy.ZERO_SHOT:
            template_name = f"zero_shot_{positions_suffix}"
        elif strategy == PromptStrategy.FEW_SHOT:
            # Determinar número de exemplos
            if num_examples is None:
                if examples:
                    num_examples = len(examples)
                else:
                    num_examples = 3  # Padrão
            
            # Limitar entre 1 e 32 (suporta templates até 32 exemplos)
            num_examples = max(1, min(32, num_examples))
            
            # Se o número de exemplos não tem template específico, usar o mais próximo disponível
            # Templates disponíveis: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32
            if num_examples <= 10:
                template_name = f"few_shot_{num_examples}_{positions_suffix}"
            elif num_examples <= 16:
                template_name = f"few_shot_16_{positions_suffix}"
            else:  # num_examples > 16 e <= 32
                template_name = f"few_shot_32_{positions_suffix}"
        else:  # chain-of-thought ou outros
            # Para outras estratégias, usar template padrão sem sufixo de posições
            template_name = strategy.value.replace('-', '_')
        
        # Buscar template
        template = self.templates.get(template_name)
        
        # Fallback: tentar template sem sufixo de posições
        if not template:
            logger.warning(f"Template '{template_name}' não encontrado, tentando fallback")
            fallback_name = strategy.value.replace('-', '_')
            template = self.templates.get(fallback_name)
        
        # Fallback final: zero-shot
        if not template:
            logger.warning(f"Template não encontrado, usando zero_shot_no_positions")
            template = self.templates.get('zero_shot_no_positions') or self.templates.get('zero_shot')
        
        if not template:
            raise ValueError(f"Nenhum template de prompt disponível. Tentou: {template_name}")
        
        # Preparar variáveis
        variables = {'text': text}
        
        if strategy == PromptStrategy.FEW_SHOT:
            if examples is None:
                # Get examples from cache or BC5CDR
                examples = self._get_bc5cdr_examples(num_examples or 3, with_positions=False)  # Sempre False
            
            # Limit examples to specified number
            if num_examples and len(examples) > num_examples:
                examples = examples[:num_examples]
            
            examples_text = self._format_examples(examples, with_positions=False)  # Sempre False
            variables['examples'] = examples_text
        
        # Formatar template
        try:
            prompt = template.template.format(**variables)
            return prompt
        except KeyError as e:
            logger.error(f"Variável não encontrada no template: {e}")
            return template.template.format(text=text)
    
    def _get_bc5cdr_examples(self, num_examples: int = 3, with_positions: bool = False) -> List[Dict[str, Any]]:
        """
        Get examples for few-shot learning
        
        Args:
            num_examples: Number of examples
            with_positions: Sempre False (with_positions foi removido)
            
        Returns:
            List of examples
        """
        # First try to load from YAML file
        # Sempre usa no_positions (with_positions foi removido)
        cache_key = 'no_positions'
        if cache_key in self.examples_cache and self.examples_cache[cache_key]:
            examples = self.examples_cache[cache_key]
            if len(examples) >= num_examples:
                return examples[:num_examples]
            else:
                logger.warning(f"Only {len(examples)} examples available in cache, requested {num_examples}")
                return examples
        
        # Try BC5CDR parser if available
        try:
            if BC5CDRParser is None:
                # Parser not available, use hardcoded examples
                return self._get_hardcoded_examples()[:num_examples]
            
            if self.bc5cdr_parser is None:
                self.bc5cdr_parser = BC5CDRParser()
            
            # Load development split
            dev_articles = self.bc5cdr_parser.load_split('dev')
            examples = self.bc5cdr_parser.get_examples_for_few_shot(dev_articles, num_examples)
            return examples
        except Exception as e:
            logger.warning(f"Error loading examples from BC5CDR: {e}")
            # Hardcoded examples as fallback
            return self._get_hardcoded_examples()[:num_examples]
    
    def _load_examples(self):
        """Load examples from YAML file"""
        try:
            if self.examples_config_path and self.examples_config_path.exists():
                with open(self.examples_config_path, 'r', encoding='utf-8') as f:
                    examples_config = yaml.safe_load(f)
                
                # Load examples without positions
                if 'examples_no_positions' in examples_config:
                    self.examples_cache['no_positions'] = examples_config['examples_no_positions']
                
                # examples_with_positions foi removido - sempre usa no_positions
                
                logger.info(f"Loaded {len(self.examples_cache.get('no_positions', []))} examples without positions")
            else:
                logger.warning(f"Examples file not found: {self.examples_config_path}")
        except Exception as e:
            logger.warning(f"Error loading examples from YAML: {e}")
    
    def _get_hardcoded_examples(self) -> List[Dict[str, Any]]:
        """Hardcoded examples as fallback"""
        return [
            {
                'text': 'Lithium carbonate toxicity in newborn infant.',
                'entities': [
                    {'text': 'Lithium carbonate', 'start': 0, 'end': 17, 'type': 'Chemical'},
                    {'text': 'toxicity', 'start': 18, 'end': 26, 'type': 'Disease'},
                    {'text': 'newborn infant', 'start': 30, 'end': 44, 'type': 'Disease'}
                ],
                'article_id': 'example_1'
            },
            {
                'text': 'Phenobarbital-induced dyskinesia in neurologically-impaired child.',
                'entities': [
                    {'text': 'Phenobarbital', 'start': 0, 'end': 13, 'type': 'Chemical'},
                    {'text': 'dyskinesia', 'start': 22, 'end': 32, 'type': 'Disease'},
                    {'text': 'neurologically-impaired', 'start': 36, 'end': 58, 'type': 'Disease'}
                ],
                'article_id': 'example_2'
            },
            {
                'text': 'Valproic acid treatment for epilepsy.',
                'entities': [
                    {'text': 'Valproic acid', 'start': 0, 'end': 13, 'type': 'Chemical'},
                    {'text': 'epilepsy', 'start': 26, 'end': 34, 'type': 'Disease'}
                ],
                'article_id': 'example_3'
            }
        ]
    
    def _format_examples(self, examples: List[Dict[str, Any]], with_positions: bool = False) -> str:
        """
        Formata exemplos para o prompt
        
        Args:
            examples: Lista de exemplos
            with_positions: Sempre False (with_positions foi removido)
            
        Returns:
            Texto formatado dos exemplos
        """
        # Tentar usar templates do YAML se disponíveis
        # Sempre usa no_positions (with_positions foi removido)
        template_name = 'example_template_no_positions'
        entity_template_name = 'entity_template_no_positions'
        
        example_template = self.templates.get(template_name)
        entity_template = self.templates.get(entity_template_name)
        
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            # Formatar entidades
            entities_text = []
            entities_list = example.get('entities', [])
            
            # Se entities é uma string, tentar parsear ou usar diretamente
            if isinstance(entities_list, str):
                entities_text.append(entities_list)
            else:
                # Formatar cada entidade usando template ou formato simples
                for entity in entities_list:
                    if entity_template and isinstance(entity, dict):
                        try:
                            # Sempre sem posições (with_positions foi removido)
                            entity_line = entity_template.template.format(
                                text=entity.get('text', ''),
                                type=entity.get('type', '')
                            )
                            entities_text.append(entity_line)
                        except KeyError:
                            # Fallback para formato simples (sem posições)
                            entities_text.append(f"- \"{entity.get('text', '')}\" (tipo: {entity.get('type', '')})")
                    else:
                        # Formato simples (sem posições)
                        if isinstance(entity, dict):
                            entities_text.append(f"- \"{entity.get('text', '')}\" (tipo: {entity.get('type', '')})")
                        else:
                            entities_text.append(str(entity))
            
            # Formatar exemplo completo usando template ou formato simples
            if example_template:
                try:
                    formatted_example = example_template.template.format(
                        num=i,
                        text=example.get('text', ''),
                        entities='\n'.join(entities_text)
                    )
                    formatted_examples.append(formatted_example)
                except KeyError:
                    # Fallback para formato simples
                    formatted_example = f"""Exemplo {i}:
Texto: "{example.get('text', '')}"
Entidades encontradas:
{chr(10).join(entities_text)}"""
                    formatted_examples.append(formatted_example)
            else:
                # Formato simples sem template
                formatted_example = f"""Exemplo {i}:
Texto: "{example.get('text', '')}"
Entidades encontradas:
{chr(10).join(entities_text)}"""
                formatted_examples.append(formatted_example)
        
        return "\n\n".join(formatted_examples) + "\n\n"
    
    def generate_relation_prompt(self, text: str) -> str:
        """
        Gera prompt para extração de relações
        
        Args:
            text: Texto para análise
            
        Returns:
            Prompt para extração de relações
        """
        template = """Analise o texto abaixo e identifique relações entre substâncias químicas e doenças.

Texto: {text}

Para cada relação encontrada, forneça:
- Químico envolvido
- Doença envolvida
- Tipo de relação (CID = Chemical-Induced Disease)
- Confiança da relação

Responda em formato JSON:
{{
  "relations": [
    {{
      "chemical": "nome do químico",
      "disease": "nome da doença",
      "relation_type": "CID",
      "confidence": 0.85
    }}
  ]
}}"""
        
        return template.format(text=text)
    
    def parse_llm_response(self, response_text: str, original_text: str, use_positions: bool = True) -> List[Entity]:
        """
        Parseia resposta do LLM para extrair entidades
        
        Args:
            response_text: Resposta do LLM
            original_text: Texto original
            use_positions: Se True, extrai e valida posições. Se False, ignora posições.
            
        Returns:
            Lista de entidades extraídas
        """
        entities = []
        
        try:
            # Tentar parsear JSON - procurar por JSON em qualquer lugar da resposta
            json_start = response_text.find('{')
            if json_start != -1:
                # Extrair JSON da resposta
                json_text = response_text[json_start:]
                # Procurar pelo final do JSON
                json_end = json_text.rfind('}') + 1
                if json_end > 0:
                    json_text = json_text[:json_end]
                    
                try:
                    data = json.loads(json_text)
                    
                    if 'entities' in data and isinstance(data['entities'], list):
                        for entity_data in data['entities']:
                            if isinstance(entity_data, dict):
                                entity = self._parse_entity(entity_data, original_text, use_positions)
                                if entity:
                                    entities.append(entity)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: tentar extrair entidades com regex
            if not entities:
                entities = self._extract_entities_fallback(response_text, original_text, use_positions)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Erro ao parsear resposta JSON: {e}")
            entities = self._extract_entities_fallback(response_text, original_text, use_positions)
        
        return entities
    
    def _parse_entity(self, entity_data: Dict[str, Any], original_text: str, use_positions: bool = True) -> Optional[Entity]:
        """
        Parseia dados de uma entidade
        
        Args:
            entity_data: Dados da entidade
            original_text: Texto original
            use_positions: Se True, valida posições. Se False, usa valores padrão.
            
        Returns:
            Entidade parseada ou None
        """
        try:
            text = entity_data.get('text', '')
            entity_type = entity_data.get('type', '')
            confidence = entity_data.get('confidence', 0.8)
            mesh_id = entity_data.get('mesh_id')
            
            # Validar tipo
            if entity_type not in ['Chemical', 'Disease']:
                return None
            
            # Processar posições
            if use_positions:
                # Quando use_positions=True, validar e usar posições do JSON
                start = entity_data.get('start')
                end = entity_data.get('end')
                
                # Se não tiver posições no JSON, tentar encontrar no texto
                if start is None or end is None:
                    pos = original_text.find(text)
                    if pos != -1:
                        start = pos
                        end = pos + len(text)
                    else:
                        return None
                
                # Validar posições
                if start < 0 or end > len(original_text) or start >= end:
                    # Tentar encontrar posição no texto
                    pos = original_text.find(text)
                    if pos != -1:
                        start = pos
                        end = pos + len(text)
                    else:
                        return None
            else:
                # Quando use_positions=False, usar valores padrão (não serão usados, mas Entity requer)
                # Tentar encontrar posição no texto para valores padrão
                pos = original_text.find(text)
                if pos != -1:
                    start = pos
                    end = pos + len(text)
                else:
                    # Se não encontrar, usar valores padrão que não serão salvos
                    start = 0
                    end = len(text)
            
            return Entity(
                text=text,
                start=start,
                end=end,
                type=EntityType.CHEMICAL if entity_type == 'Chemical' else EntityType.DISEASE,
                confidence=float(confidence),
                mesh_id=mesh_id
            )
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Erro ao parsear entidade: {e}")
            return None
    
    def _extract_entities_fallback(self, response_text: str, original_text: str, use_positions: bool = True) -> List[Entity]:
        """
        Extrai entidades usando regex como fallback
        
        Args:
            response_text: Resposta do LLM
            original_text: Texto original
            use_positions: Se True, inclui posições. Se False, usa valores padrão.
            
        Returns:
            Lista de entidades extraídas
        """
        entities = []
        
        # Padrões para encontrar entidades na resposta
        import re
        
        # Procurar por padrões como "Chemical: nome" ou "Disease: nome"
        chemical_pattern = r'(?:Chemical|Químico)[:\s]+([A-Za-z\s]+)'
        disease_pattern = r'(?:Disease|Doença)[:\s]+([A-Za-z\s]+)'
        
        chemical_matches = re.findall(chemical_pattern, response_text, re.IGNORECASE)
        disease_matches = re.findall(disease_pattern, response_text, re.IGNORECASE)
        
        # Adicionar químicos
        for match in chemical_matches:
            text = match.strip()
            if text and len(text) > 2:
                pos = original_text.find(text)
                if pos != -1:
                    entity = Entity(
                        text=text,
                        start=pos,
                        end=pos + len(text),
                        type=EntityType.CHEMICAL,
                        confidence=0.7,  # Confiança menor para fallback
                        mesh_id=None
                    )
                    entities.append(entity)
        
        # Adicionar doenças
        for match in disease_matches:
            text = match.strip()
            if text and len(text) > 2:
                pos = original_text.find(text)
                if pos != -1:
                    entity = Entity(
                        text=text,
                        start=pos,
                        end=pos + len(text),
                        type=EntityType.DISEASE,
                        confidence=0.7,  # Confiança menor para fallback
                        mesh_id=None
                    )
                    entities.append(entity)
        
        return entities
    
    def get_available_strategies(self) -> List[str]:
        """Retorna estratégias de prompt disponíveis"""
        return [strategy.value for strategy in PromptStrategy]
    
    def get_template_info(self, strategy: PromptStrategy) -> Dict[str, Any]:
        """
        Obtém informações sobre um template
        
        Args:
            strategy: Estratégia de prompt
            
        Returns:
            Informações do template
        """
        template_name = strategy.value.replace('-', '_')
        template = self.templates.get(template_name)
        
        if template:
            return {
                'name': template.name,
                'description': template.description,
                'variables': template.variables,
                'template_preview': template.template[:200] + "..."
            }
        else:
            return {
                'name': template_name,
                'description': 'Template não encontrado',
                'variables': [],
                'template_preview': 'N/A'
            }


def test_prompt_engine():
    """Função de teste para o Prompt Engine"""
    engine = PromptEngine()
    
    # Testar diferentes estratégias
    text = "Lithium carbonate toxicity in newborn infant."
    
    print("🧪 Testando Prompt Engine:")
    
    for strategy in PromptStrategy:
        print(f"\n📝 Estratégia: {strategy.value}")
        prompt = engine.generate_prompt(text, strategy)
        print(f"Prompt gerado ({len(prompt)} chars):")
        print(prompt[:200] + "...")
    
    # Testar parsing de resposta
    print("\n🔍 Testando parsing de resposta:")
    mock_response = '''{
        "entities": [
            {
                "text": "Lithium carbonate",
                "start": 0,
                "end": 15,
                "type": "Chemical",
                "confidence": 0.95,
                "mesh_id": "D016651"
            },
            {
                "text": "toxicity",
                "start": 16,
                "end": 24,
                "type": "Disease",
                "confidence": 0.90,
                "mesh_id": "D064420"
            }
        ]
    }'''
    
    entities = engine.parse_llm_response(mock_response, text)
    print(f"Entidades extraídas: {len(entities)}")
    for entity in entities:
        print(f"  - {entity.text} ({entity.type.value}) - Confiança: {entity.confidence}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_prompt_engine()
