"""
Template-Based Response Generation Service
Gera respostas variadas usando templates + fatos da KB
"""
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from .astrology_entity_service import AstrologyEntityService, AstrologyEntities
from ..core.logger import get_logger

logger = get_logger(__name__)


class TemplateResponseService:
    """
    Gera respostas usando templates que preenchem fatos da KB.
    Garante precisão factual com variação linguística natural.
    """
    
    # Templates para diferentes tipos de resposta (organizados por tópico)
    RESPONSE_TEMPLATES = {
        'elemento': [
            "{signo} pertence ao elemento {elemento}, que representa {qualidades}.",
            "O elemento de {signo} é {elemento}, trazendo características como {qualidades}.",
            "{elemento} é o elemento que rege {signo}, manifestando {qualidades}.",
            "{signo} é do elemento {elemento}, que traz {qualidades}.",
            "Como signo de {elemento}, {signo} expressa {qualidades}.",
            "{elemento} governa {signo}, conferindo {qualidades}.",
            "A natureza de {signo} é moldada pelo elemento {elemento}, com {qualidades}.",
            "{signo} carrega a energia do {elemento}, manifestando {qualidades}.",
            "Sob a influência do elemento {elemento}, {signo} demonstra {qualidades}.",
            "O signo de {signo} vibra com o elemento {elemento} e suas {qualidades}.",
            "{elemento} é o princípio elemental de {signo}, trazendo {qualidades}."
        ],
        'planeta': [
            "{signo} é regido por {planeta}, que influencia com características de {influencia}.",
            "O planeta regente de {signo} é {planeta}, trazendo {influencia}.",
            "{planeta} rege {signo}, manifestando {influencia}.",
            "{signo} tem como regente {planeta}, que traz {influencia}.",
            "Sob a regência de {planeta}, {signo} expressa {influencia}.",
            "{planeta} governa {signo}, conferindo {influencia}.",
            "A influência de {planeta} sobre {signo} resulta em {influencia}.",
            "{signo} recebe a energia de {planeta}, que traz {influencia}.",
            "Como regente, {planeta} dá a {signo} {influencia}.",
            "{planeta} é o astro regente de {signo}, manifestando {influencia}.",
            "O poder de {planeta} guia {signo} através de {influencia}."
        ],
        'caracteristicas': [
            "{signo} é {caracteristicas}. Pessoas deste signo tendem a ser {principal}.",
            "Características marcantes de {signo}: {caracteristicas}.",
            "Quem é de {signo} costuma ser {caracteristicas}, com forte tendência para {principal}.",
            "{signo} se destaca por ser {caracteristicas}.",
            "Os nativos de {signo} são {caracteristicas}, demonstrando {principal}.",
            "{signo} manifesta {caracteristicas} como traços naturais.",
            "É típico de {signo} apresentar {caracteristicas}.",
            "A personalidade de {signo} revela {caracteristicas} e {principal}.",
            "Pessoas de {signo} carregam {caracteristicas} em sua essência.",
            "{signo} é conhecido por {caracteristicas}, sendo {principal}.",
            "Entre as qualidades de {signo} estão {caracteristicas}.",
            "O jeito de ser de {signo} inclui {caracteristicas} e tendência para {principal}."
        ],
        'corpo': [
            "{signo} rege {parte_corpo}. É importante cuidar especialmente desta região.",
            "A parte do corpo regida por {signo} é {parte_corpo}.",
            "{parte_corpo} está sob a regência de {signo}, merecendo atenção especial.",
            "{signo} governa {parte_corpo}, sendo importante manter esta área saudável.",
            "A energia de {signo} influencia {parte_corpo}.",
            "{signo} tem conexão com {parte_corpo}, área que merece cuidado.",
            "Sob a regência de {signo}, {parte_corpo} requer atenção.",
            "{parte_corpo} é a área corporal associada a {signo}.",
            "{signo} protege e rege {parte_corpo}.",
            "A região de {parte_corpo} é especialmente importante para {signo}.",
            "{signo} mantém ligação energética com {parte_corpo}."
        ],
        'pedras': [
            "As pedras de {signo} são {pedras}. Elas potencializam as qualidades do signo.",
            "{signo} tem afinidade com {pedras}, cristais que trazem proteção e energia.",
            "Para {signo}, as pedras ideais são {pedras}.",
            "{pedras} são as pedras de {signo}, fortalecendo sua energia natural.",
            "Cristais recomendados para {signo}: {pedras}.",
            "{signo} vibra bem com {pedras}, pedras de poder.",
            "Use {pedras} para amplificar a energia de {signo}.",
            "{pedras} harmonizam perfeitamente com {signo}.",
            "As pedras de proteção e sorte de {signo} são {pedras}.",
            "{signo} encontra apoio energético em {pedras}.",
            "Carregue {pedras} para potencializar a energia de {signo}."
        ],
        'cores': [
            "As cores de sorte de {signo} são {cores}. Usar essas cores traz harmonia.",
            "{signo} vibra com as cores {cores}.",
            "Para {signo}, as cores favoráveis são {cores}.",
            "{cores} são as cores que harmonizam com a energia de {signo}.",
            "Use {cores} para atrair a energia positiva de {signo}.",
            "{signo} se fortalece com as cores {cores}.",
            "As tonalidades de {cores} favorecem {signo}.",
            "{cores} são as cores da sorte para quem é {signo}.",
            "Vista {cores} para potencializar a vibração de {signo}.",
            "{signo} encontra harmonia nas cores {cores}.",
            "Decore com {cores} para atrair a energia de {signo}."
        ],
        'numeros': [
            "Os números da sorte de {signo} são {numeros}.",
            "{signo} tem afinidade com os números {numeros}.",
            "Para {signo}, os números favoráveis são {numeros}.",
            "{numeros} são números que vibram com {signo}.",
            "Use {numeros} como números da sorte se você é {signo}.",
            "{signo} encontra fortuna nos números {numeros}.",
            "Os números {numeros} trazem boa energia para {signo}.",
            "{numeros} são especialmente favoráveis para {signo}.",
            "Aposte nos números {numeros} se você é de {signo}.",
            "{signo} vibra positivamente com {numeros}.",
            "Números de poder para {signo}: {numeros}."
        ],
        'profissoes': [
            "{signo} se dá bem em profissões como {profissoes}. Sua natureza {caracteristica} favorece estas áreas.",
            "Profissões ideais para {signo}: {profissoes}.",
            "{signo} tem talento natural para {profissoes}, graças à sua {caracteristica}.",
            "Para {signo}, áreas como {profissoes} são especialmente favoráveis.",
            "Carreiras em {profissoes} combinam com o perfil de {signo}.",
            "{signo} brilha profissionalmente em {profissoes}.",
            "Sua {caracteristica} torna {signo} excelente em {profissoes}.",
            "{profissoes} são campos onde {signo} se realiza.",
            "O talento de {signo} se destaca em {profissoes}.",
            "{signo} encontra sucesso trabalhando com {profissoes}.",
            "Áreas como {profissoes} valorizam as qualidades de {signo}."
        ],
        'desafios': [
            "Os principais desafios de {signo} incluem {desafios}. Trabalhar estes aspectos traz evolução.",
            "{signo} precisa estar atento a {desafios}.",
            "Pontos de atenção para {signo}: {desafios}.",
            "Para {signo}, é importante cuidar de {desafios} para maior equilíbrio.",
            "{signo} evolui ao trabalhar {desafios}.",
            "Aspectos que {signo} deve superar: {desafios}.",
            "{desafios} são lições importantes para {signo}.",
            "{signo} cresce ao enfrentar {desafios}.",
            "Desafios evolutivos de {signo}: {desafios}.",
            "{signo} encontra crescimento ao lidar com {desafios}.",
            "Para amadurecer, {signo} precisa trabalhar {desafios}."
        ],
        'amor': [
            "{signo} no amor é {caracteristica}. Busca um parceiro que valorize sua {qualidade}.",
            "Em relacionamentos, {signo} tende a ser {caracteristica}.",
            "{signo} se relaciona de forma {caracteristica}, valorizando {qualidade}.",
            "No campo amoroso, {signo} é {caracteristica} e busca {qualidade}.",
            "Quando ama, {signo} demonstra {caracteristica}.",
            "{signo} expressa amor sendo {caracteristica}.",
            "Em relações, {signo} valoriza {qualidade} e age com {caracteristica}.",
            "O jeito de amar de {signo} é {caracteristica}.",
            "{signo} busca {qualidade} nos relacionamentos.",
            "No amor, a natureza {caracteristica} de {signo} se revela.",
            "{signo} se entrega de forma {caracteristica} aos relacionamentos."
        ],
        'periodo': [
            "{signo} vai de {periodo}.",
            "O período de {signo} é {periodo}.",
            "{signo} compreende o período entre {periodo}.",
            "Nascidos entre {periodo} são de {signo}.",
            "{signo} abrange {periodo}.",
            "Quem nasce entre {periodo} é do signo de {signo}.",
            "A temporada de {signo} vai de {periodo}.",
            "{periodo} marca o período de {signo}.",
            "{signo} acontece entre {periodo}.",
            "O signo de {signo} cobre {periodo}."
        ],
        'identificar_signo': [
            "Quem nasceu em {data} é de {signo}.",
            "Nascidos em {data} são do signo de {signo}.",
            "Seu signo é {signo} (nascimento em {data}).",
            "{signo}! Essa é a resposta para quem nasce em {data}.",
            "Se você nasceu em {data}, seu signo é {signo}.",
            "A data {data} corresponde ao signo de {signo}."
        ]
    }
    
    def __init__(self, kb_path: Optional[str] = None):
        """
        Inicializa o serviço de geração baseada em templates
        
        Args:
            kb_path: Caminho para arquivo CSV da base de conhecimento
        """
        self.entity_service = AstrologyEntityService()
        self.kb_data: Dict[str, Dict[str, Dict[str, str]]] = {}
        
        if kb_path:
            self.load_kb(kb_path)
    
    def load_kb(self, kb_path: str):
        """
        Carrega base de conhecimento e estrutura para lookup rápido
        
        Args:
            kb_path: Caminho para arquivo CSV
        """
        try:
            df = pd.read_csv(kb_path, encoding='utf-8-sig')
            logger.info(f"KB carregada: {len(df)} entradas de {kb_path}")
            
            # Estruturar: kb_data[signo][categoria] = dados
            for _, row in df.iterrows():
                categoria = row.get('categoria', '')
                resposta = row.get('resposta', '')
                pergunta = row.get('pergunta', '')
                
                # Extrair signo da pergunta ou resposta
                entities = self.entity_service.extract(pergunta)
                if not entities.has_zodiac():
                    entities = self.entity_service.extract(resposta)
                
                if entities.has_zodiac():
                    signo = entities.zodiac_sign
                    
                    if signo not in self.kb_data:
                        self.kb_data[signo] = {}
                    
                    # Mapear categoria para tópico
                    topic = self._map_category_to_topic(categoria)
                    if topic:
                        if topic not in self.kb_data[signo]:
                            self.kb_data[signo][topic] = {}
                        
                        # Parsear resposta para extrair fatos
                        facts = self._parse_response_facts(resposta, topic)
                        self.kb_data[signo][topic].update(facts)
            
            logger.info(f"KB estruturada: {len(self.kb_data)} signos com dados")
            
        except Exception as e:
            logger.error(f"Erro ao carregar KB: {e}")
            import traceback
            traceback.print_exc()
    
    def _map_category_to_topic(self, categoria: str) -> Optional[str]:
        """Mapeia categoria da KB para tópico normalizado"""
        mapping = {
            'signo_basico': 'elemento',  # Contém elemento e planeta
            'signo_caracteristicas': 'caracteristicas',
            'signo_corpo': 'corpo',
            'signo_pedras': 'pedras',
            'signo_cores': 'cores',
            'signo_numeros': 'numeros',
            'signo_profissoes': 'profissoes',
            'signo_desafios': 'desafios',
            'signo_amor': 'amor',
            'elementos': 'elemento',
            'data_para_signo': 'periodo'
        }
        return mapping.get(categoria)
    
    def _parse_response_facts(self, resposta: str, topic: str) -> Dict[str, str]:
        """
        Extrai fatos estruturados de uma resposta
        
        Args:
            resposta: Texto da resposta
            topic: Tópico da resposta
            
        Returns:
            Dicionário com fatos extraídos
        """
        facts = {}
        
        if topic == 'elemento':
            # "Áries é do elemento Fogo, que traz Energia, paixão..."
            if 'elemento' in resposta.lower():
                match = re.search(r'elemento\s+(\w+)', resposta, re.IGNORECASE)
                if match:
                    facts['elemento'] = match.group(1)
                
                # Extrair qualidades após vírgula
                if ',' in resposta:
                    parts = resposta.split(',', 1)
                    if len(parts) > 1:
                        qualidades = parts[1].replace('que traz', '').strip(' .')
                        facts['qualidades'] = qualidades
        
        elif topic == 'planeta':
            # "Áries é regido por Marte, que influencia..."
            if 'regido' in resposta.lower() or 'rege' in resposta.lower():
                match = re.search(r'(?:regido por|rege)\s+(\w+)', resposta, re.IGNORECASE)
                if match:
                    facts['planeta'] = match.group(1)
                
                if 'influencia' in resposta.lower():
                    match = re.search(r'influencia.*?com características de (.+)', resposta, re.IGNORECASE)
                    if match:
                        facts['influencia'] = match.group(1).strip(' .')
        
        elif topic == 'caracteristicas':
            # "Áries é corajoso, impulsivo, líder nato..."
            match = re.search(r'é\s+([^\.]+)\.', resposta)
            if match:
                facts['caracteristicas'] = match.group(1).strip()
        
        elif topic == 'corpo':
            # "Áries rege cabeça, face, cérebro"
            match = re.search(r'rege\s+([^\.]+)\.', resposta, re.IGNORECASE)
            if match:
                facts['parte_corpo'] = match.group(1).strip()
        
        elif topic in ['pedras', 'cores', 'numeros', 'profissoes', 'desafios']:
            # "As pedras de Áries são rubi, diamante..."
            pattern = rf'{topic}.*?(?:são|incluem)\s+([^\.]+)\.'
            match = re.search(pattern, resposta, re.IGNORECASE)
            if match:
                facts[topic] = match.group(1).strip()
        
        elif topic == 'periodo':
            # "Áries vai de 21/03 a 19/04"
            match = re.search(r'(\d{2}/\d{2})\s+a\s+(\d{2}/\d{2})', resposta)
            if match:
                facts['periodo'] = f"{match.group(1)} a {match.group(2)}"
        
        # Se não conseguiu extrair, guarda resposta original
        if not facts:
            facts['texto_original'] = resposta
        
        return facts
    
    def generate(
        self, 
        intent: str, 
        message: str, 
        entities: Optional[AstrologyEntities] = None
    ) -> Optional[str]:
        """
        Gera resposta baseada em template + fatos da KB
        
        Args:
            intent: Intent classificado
            message: Mensagem original do usuário
            entities: Entidades já extraídas (opcional)
            
        Returns:
            Resposta gerada ou None se não conseguir
        """
        # Extrair entidades se não fornecidas
        if entities is None:
            entities = self.entity_service.extract(message)
        
        # CASO ESPECIAL: identificar_signo com data
        if intent == 'identificar_signo' and entities.zodiac_sign and entities.date:
            logger.debug(f"Identificar signo - Entidades: signo={entities.zodiac_sign}, data={entities.date}")
            # Tem signo determinado pela data - usar template específico
            if 'identificar_signo' in self.RESPONSE_TEMPLATES:
                template = random.choice(self.RESPONSE_TEMPLATES['identificar_signo'])
                try:
                    # Formatar data para exibição mais legível
                    date_parts = entities.date.split('/')
                    logger.debug(f"Date parts: {date_parts}")
                    if len(date_parts) == 2:
                        day, month = date_parts
                        date_display = f"{int(day)} de {self._get_month_name(int(month))}"
                    else:
                        date_display = entities.date
                    
                    logger.debug(f"Template: '{template}', data_display: '{date_display}'")
                    response = template.format(signo=entities.zodiac_sign, data=date_display)
                    return response
                except KeyError:
                    pass
        
        # CASO NORMAL: Precisa de signo e tópico
        if not entities.has_zodiac():
            logger.debug(f"Sem signo para gerar resposta")
            return None
        
        # Se não tem tópico explícito mas tem data, inferir que é sobre período
        topic = entities.topic
        if not topic and entities.date:
            topic = 'periodo'
        
        if not topic:
            logger.debug(f"Sem tópico para gerar resposta")
            return None
        
        signo = entities.zodiac_sign
        
        # Buscar fatos na KB
        if signo not in self.kb_data or topic not in self.kb_data[signo]:
            logger.debug(f"Sem dados para signo={signo}, topic={topic}")
            return None
        
        facts = self.kb_data[signo][topic]
        
        # Selecionar template aleatório
        if topic not in self.RESPONSE_TEMPLATES:
            # Fallback: retorna fato original
            return facts.get('texto_original')
        
        template = random.choice(self.RESPONSE_TEMPLATES[topic])
        
        # Preencher template com fatos
        try:
            response = template.format(signo=signo, **facts)
            return response
        except KeyError as e:
            logger.warning(f"Template requer campo ausente: {e}")
            # Fallback: retorna fato original
            return facts.get('texto_original')
    
    def _get_month_name(self, month_num: int) -> str:
        """Converte número do mês para nome"""
        months = {
            1: 'janeiro', 2: 'fevereiro', 3: 'março', 4: 'abril',
            5: 'maio', 6: 'junho', 7: 'julho', 8: 'agosto',
            9: 'setembro', 10: 'outubro', 11: 'novembro', 12: 'dezembro'
        }
        return months.get(month_num, str(month_num))


import re
