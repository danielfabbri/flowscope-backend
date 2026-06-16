"""
Astrology Entity Extraction Service - Identifica entidades em perguntas de astrologia
Usa pattern matching simples sem dependências externas
"""
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AstrologyEntities:
    """Entidades extraídas de uma pergunta sobre astrologia"""
    zodiac_sign: Optional[str] = None
    topic: Optional[str] = None
    date: Optional[str] = None
    
    def has_zodiac(self) -> bool:
        return self.zodiac_sign is not None
    
    def has_topic(self) -> bool:
        return self.topic is not None


class AstrologyEntityService:
    """
    Extrai entidades de perguntas sobre astrologia.
    Identifica signos, temas (elemento, características, etc.), datas.
    NÃO requer bibliotecas externas - usa regex e pattern matching.
    """
    
    # Signos do zodíaco com variações (normalizado -> variações)
    ZODIAC_SIGNS = {
        'Áries': ['áries', 'aries', 'ariano', 'ariana'],
        'Touro': ['touro', 'taurino', 'taurina'],
        'Gêmeos': ['gêmeos', 'gemeos', 'geminiano', 'geminiana'],
        'Câncer': ['câncer', 'cancer', 'canceriano', 'canceriana'],
        'Leão': ['leão', 'leao', 'leonino', 'leonina'],
        'Virgem': ['virgem', 'virginiano', 'virginiana'],
        'Libra': ['libra', 'libriano', 'libriana'],
        'Escorpião': ['escorpião', 'escorpiao', 'escorpiano', 'escorpiana'],
        'Sagitário': ['sagitário', 'sagitario', 'sagitariano', 'sagitariana'],
        'Capricórnio': ['capricórnio', 'capricornio', 'capricorniano', 'capricorniana'],
        'Aquário': ['aquário', 'aquario', 'aquariano', 'aquariana'],
        'Peixes': ['peixes', 'pisciano', 'pisciana']
    }
    
    # Temas/tópicos astrológicos (normalizado -> keywords)
    TOPICS = {
        'elemento': ['elemento', 'elementos'],
        'planeta': ['planeta', 'planetas', 'regente', 'regido', 'rege'],
        'caracteristicas': ['característica', 'caracteristicas', 'características', 'qualidade', 'qualidades', 'jeito'],
        'corpo': ['corpo', 'saúde', 'física', 'físico', 'parte'],
        'pedras': ['pedra', 'pedras', 'cristal', 'cristais'],
        'cores': ['cor', 'cores'],
        'numeros': ['número', 'números', 'numero', 'numeros', 'sorte'],
        'profissoes': ['profissão', 'profissões', 'profissao', 'profissoes', 'carreira', 'trabalho'],
        'desafios': ['desafio', 'desafios', 'defeito', 'defeitos', 'dificuldade'],
        'amor': ['amor', 'amoroso', 'romance', 'romântico', 'relacionamento'],
        'compatibilidade': ['compatível', 'compatibilidade', 'combina', 'match', 'par'],
        'periodo': ['data', 'datas', 'período', 'periodo', 'quando', 'vai de', 'nasce', 'nasci', 'nasceu'],
        'amizade': ['amigo', 'amigos', 'amizade'],
        'familia': ['família', 'familia', 'familiar'],
        'financas': ['dinheiro', 'finanças', 'financas', 'financeiro'],
        'espiritualidade': ['espiritual', 'espiritualidade', 'místico', 'mistico']
    }
    
    # Meses do ano (para extração de datas)
    MONTHS = {
        'janeiro': 1,
        'fevereiro': 2,
        'março': 3, 'marco': 3,
        'abril': 4,
        'maio': 5,
        'junho': 6,
        'julho': 7,
        'agosto': 8,
        'setembro': 9,
        'outubro': 10,
        'novembro': 11,
        'dezembro': 12
    }
    
    # Mapeamento de período para signos
    DATE_TO_SIGN = [
        ((1, 20), (2, 18), 'Aquário'),
        ((2, 19), (3, 20), 'Peixes'),
        ((3, 21), (4, 19), 'Áries'),
        ((4, 20), (5, 20), 'Touro'),
        ((5, 21), (6, 20), 'Gêmeos'),
        ((6, 21), (7, 22), 'Câncer'),
        ((7, 23), (8, 22), 'Leão'),
        ((8, 23), (9, 22), 'Virgem'),
        ((9, 23), (10, 22), 'Libra'),
        ((10, 23), (11, 21), 'Escorpião'),
        ((11, 22), (12, 21), 'Sagitário'),
        ((12, 22), (1, 19), 'Capricórnio')
    ]
    
    def __init__(self):
        """Inicializa o serviço de extração de entidades astrológicas"""
        # Criar lookup reverso para signos (variação -> normalizado)
        self.sign_lookup = {}
        for normalized_name, variations in self.ZODIAC_SIGNS.items():
            for variation in variations:
                self.sign_lookup[variation.lower()] = normalized_name
        
        # Criar lookup reverso para tópicos (keyword -> normalizado)
        self.topic_lookup = {}
        for normalized_topic, keywords in self.TOPICS.items():
            for keyword in keywords:
                self.topic_lookup[keyword.lower()] = normalized_topic
    
    def extract(self, text: str) -> AstrologyEntities:
        """
        Extrai entidades astrológicas de um texto
        
        Args:
            text: Texto para extrair entidades (pergunta do usuário)
            
        Returns:
            AstrologyEntities com entidades encontradas
        """
        if not text:
            return AstrologyEntities()
        
        text_lower = text.lower()
        entities = AstrologyEntities()
        
        # Extrair signo do zodíaco explícito
        entities.zodiac_sign = self._extract_zodiac_sign(text_lower)
        
        # Extrair tópico
        entities.topic = self._extract_topic(text_lower)
        
        # Extrair data (formato DD/MM, DD-MM, ou por extenso)
        entities.date = self._extract_date(text_lower)
        # logger.info(f"[EXTRACT] Após _extract_date: entities.date = {entities.date}")
        
        # Se não encontrou signo mas tem data, determinar signo pela data
        if not entities.zodiac_sign and entities.date:
            entities.zodiac_sign = self._get_sign_from_date(entities.date)
            # logger.info(f"[EXTRACT] Signo determinado pela data: {entities.zodiac_sign}")
        
        # Se perguntou sobre mês (sem dia específico), inferir contexto
        # IMPORTANTE: só fazer isso se NÃO extraímos uma data completa antes
        if not entities.date:
            # logger.info(f"[EXTRACT] Tentando _extract_month_only porque entities.date está vazio")
            month_only = self._extract_month_only(text_lower)
            if not entities.zodiac_sign and month_only:
                # Para perguntas como "qual signo de outubro"
                entities.date = f"15/{month_only:02d}"  # Usar dia 15 como representativo
                entities.zodiac_sign = self._get_sign_from_date(entities.date)
                # logger.info(f"[EXTRACT] Usando dia 15 representativo: {entities.date}, signo: {entities.zodiac_sign}")
        else:
            pass
            # logger.info(f"[EXTRACT] Pulando _extract_month_only porque entities.date já está preenchido: {entities.date}")
        
        # logger.info(f"[EXTRACT] Final: signo={entities.zodiac_sign}, topic={entities.topic}, date={entities.date}")
        return entities
    
    def _extract_zodiac_sign(self, text: str) -> Optional[str]:
        """Extrai signo do zodíaco do texto"""
        # Procurar por qualquer variação de signo
        words = re.findall(r'\b\w+\b', text)
        
        for word in words:
            if word in self.sign_lookup:
                return self.sign_lookup[word]
        
        return None
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extrai tópico astrológico do texto"""
        # Procurar por palavras-chave de tópicos
        words = re.findall(r'\b\w+\b', text)
        
        # Priorizar matches mais específicos primeiro
        for word in words:
            if word in self.topic_lookup:
                return self.topic_lookup[word]
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extrai data no formato DD/MM, DD-MM, ou por extenso (DD de MMMM)"""
        # Pattern 1: DD/MM ou DD-MM
        date_pattern = r'\b(\d{1,2})[/-](\d{1,2})\b'
        match = re.search(date_pattern, text)
        
        if match:
            day, month = match.groups()
            result = f"{day.zfill(2)}/{month.zfill(2)}"
            # logger.info(f"[EXTRACT_DATE] Extraída data padrão: {result} de '{text}'")
            return result
        
        # Pattern 2: "DD de MMMM" (ex: "5 de julho", "23 de outubro")
        # logger.info(f"[EXTRACT_DATE] Testando pattern 2 para: '{text}'")
        # logger.info(f"[EXTRACT_DATE] MONTHS dict tem {len(self.MONTHS)} entradas")
        for month_name, month_num in self.MONTHS.items():
            pattern = rf'\b(\d{{1,2}})\s+de\s+{month_name}\b'
            match = re.search(pattern, text)
            # logger.info(f"[EXTRACT_DATE] Testando {month_name}: pattern={pattern}, match={match is not None}")
            if match:
                day = match.group(1)
                result = f"{day.zfill(2)}/{month_num:02d}"
                # logger.info(f"[EXTRACT_DATE] Extraída data por extenso: {result} (padrão: '{pattern}') de '{text}'")
                return result
        
        # logger.info(f"[EXTRACT_DATE] Nenhuma data encontrada em: '{text}'")
        return None
    
    def _extract_month_only(self, text: str) -> Optional[int]:
        """Extrai apenas o mês quando mencionado sozinho"""
        # Procura por nomes de meses no texto
        for month_name, month_num in self.MONTHS.items():
            if month_name in text:
                return month_num
        return None
    
    def _get_sign_from_date(self, date_str: str) -> Optional[str]:
        """
        Determina o signo baseado em uma data DD/MM
        
        Args:
            date_str: Data no formato DD/MM
            
        Returns:
            Nome do signo ou None se data inválida
        """
        try:
            parts = date_str.split('/')
            if len(parts) != 2:
                return None
            
            day = int(parts[0])
            month = int(parts[1])
            
            # Verificar cada período de signo
            for (start_month, start_day), (end_month, end_day), sign in self.DATE_TO_SIGN:
                # Caso especial: Capricórnio (atravessa virada do ano)
                if start_month > end_month:
                    if (month == start_month and day >= start_day) or (month == end_month and day <= end_day):
                        return sign
                else:
                    if (month == start_month and day >= start_day) or \
                       (month == end_month and day <= end_day) or \
                       (start_month < month < end_month):
                        return sign
            
            return None
        except (ValueError, IndexError):
            return None
    
    def get_all_zodiac_signs(self) -> List[str]:
        """Retorna lista de todos os signos (nomes normalizados)"""
        return list(self.ZODIAC_SIGNS.keys())
    
    def get_all_topics(self) -> List[str]:
        """Retorna lista de todos os tópicos"""
        return list(self.TOPICS.keys())
    
    def normalize_zodiac_sign(self, sign_variation: str) -> Optional[str]:
        """Normaliza variação de signo para nome padrão"""
        clean_sign = re.sub(r'[^\w]', '', sign_variation.lower())
        return self.sign_lookup.get(clean_sign)
