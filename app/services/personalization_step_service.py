"""
Personalization Step Service
Personalizes responses based on user profile and preferences
"""
import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict
from datetime import datetime
from app.services.base_service import BasePipelineService
from app.core.logger import get_logger

logger = get_logger(__name__)


class PersonalizationStepService(BasePipelineService):
    """
    Pipeline step that personalizes responses
    
    Input: DataFrame with user data and responses
    Output: DataFrame with personalized responses
    """
    
    def __init__(self):
        # User profiles storage
        self.user_profiles: Dict[str, Dict] = defaultdict(lambda: {
            'preferences': {},
            'history': [],
            'interests': [],
            'interaction_count': 0,
            'created_at': datetime.now().isoformat()
        })
        
        # Response templates for personalization
        self.personalized_templates = {}
        
        logger.info("PersonalizationStepService initialized")
    
    def execute(self, data: Optional[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute personalization on DataFrame
        
        Args:
            data: DataFrame with responses and user info
            config: Configuration with:
                - user_id_column: column with user ID
                - response_column: column with response to personalize
                - user_profile_features: list of features to use
                - adaptation_strategy: strategy (collaborative, content_based, hybrid)
                - update_profile: whether to update user profile
                - personalization_level: level (low, medium, high)
                - output_personalized_column: name for personalized response
                - output_profile_column: name for user profile snapshot
        
        Returns:
            DataFrame with personalized responses
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be empty for personalization")
        
        user_id_column = config.get('user_id_column', 'user_id')
        response_column = config.get('response_column', 'response')
        profile_features = config.get('user_profile_features', [
            'historico', 'preferencias', 'localizacao'
        ])
        strategy = config.get('adaptation_strategy', 'content_based')
        update_profile = config.get('update_profile', True)
        personalization_level = config.get('personalization_level', 'medium')
        output_personalized = config.get('output_personalized_column', 'personalized_response')
        output_profile = config.get('output_profile_column', 'user_profile')
        
        # Set default user ID if not present
        if user_id_column not in data.columns:
            data[user_id_column] = 'anonymous'
        
        if response_column not in data.columns:
            raise ValueError(f"Response column '{response_column}' not found")
        
        logger.info(f"Personalizing responses for {len(data)} users")
        logger.info(f"Strategy: {strategy}, Level: {personalization_level}")
        
        # Process each row
        personalized_responses = []
        profile_snapshots = []
        
        for idx, row in data.iterrows():
            user_id = str(row[user_id_column])
            response = str(row[response_column])
            
            # Get user profile
            profile = self.user_profiles[user_id]
            
            # Personalize response
            personalized = self._personalize_response(
                response, profile, strategy, personalization_level, row
            )
            
            # Update profile if enabled
            if update_profile:
                self._update_profile(user_id, row, profile_features)
            
            personalized_responses.append(personalized)
            profile_snapshots.append(json.dumps({
                'user_id': user_id,
                'interaction_count': profile['interaction_count'],
                'preferences': profile['preferences']
            }))
        
        # Add new columns to DataFrame
        result = data.copy()
        result[output_personalized] = personalized_responses
        result[output_profile] = profile_snapshots
        
        logger.info(f"Personalization completed")
        return result
    
    def _personalize_response(self, response: str, profile: Dict, strategy: str,
                             level: str, row: pd.Series) -> str:
        """Personalize response based on user profile"""
        if level == 'low':
            # Minimal personalization
            return response
        
        personalized = response
        
        # Add user name if available
        if 'nome' in profile['preferences']:
            nome = profile['preferences']['nome']
            if nome and nome not in personalized:
                personalized = f"{nome}, {personalized}"
        
        # Adapt based on interaction count
        if level in ['medium', 'high']:
            interaction_count = profile['interaction_count']
            
            if interaction_count == 0:
                # First interaction - be welcoming
                personalized = self._add_welcome_message(personalized)
            elif interaction_count > 10:
                # Regular user - be more informal
                personalized = self._make_informal(personalized)
        
        # Add contextual personalization
        if level == 'high':
            # Check interests
            if profile['interests']:
                personalized = self._add_interest_context(personalized, profile['interests'])
        
        # Strategy-specific personalization
        if strategy == 'collaborative':
            personalized = self._collaborative_personalization(personalized, profile)
        elif strategy == 'content_based':
            personalized = self._content_based_personalization(personalized, profile)
        elif strategy == 'hybrid':
            personalized = self._hybrid_personalization(personalized, profile)
        
        return personalized
    
    def _update_profile(self, user_id: str, row: pd.Series, features: List[str]):
        """Update user profile with interaction data"""
        profile = self.user_profiles[user_id]
        
        # Increment interaction count
        profile['interaction_count'] += 1
        
        # Update preferences from features
        for feature in features:
            if feature in row and pd.notna(row[feature]):
                value = row[feature]
                if feature == 'historico':
                    # Add to history
                    if 'message' in row:
                        profile['history'].append({
                            'message': str(row['message']),
                            'timestamp': datetime.now().isoformat()
                        })
                        # Keep last 50
                        profile['history'] = profile['history'][-50:]
                elif feature == 'preferencias':
                    # Update preferences
                    try:
                        if isinstance(value, str):
                            prefs = json.loads(value)
                        else:
                            prefs = value
                        profile['preferences'].update(prefs)
                    except (json.JSONDecodeError, TypeError):
                        pass
                else:
                    # Store other features
                    profile['preferences'][feature] = value
        
        # Extract interests from message
        if 'message' in row:
            message = str(row['message']).lower()
            # Simple keyword-based interest extraction
            interests_keywords = {
                'futebol': ['jogo', 'time', 'gol', 'campeonato'],
                'resultado': ['placar', 'resultado', 'quanto'],
                'escalação': ['escalação', 'jogadores', 'time titular'],
                'estatísticas': ['estatísticas', 'números', 'dados']
            }
            
            for interest, keywords in interests_keywords.items():
                if any(kw in message for kw in keywords):
                    if interest not in profile['interests']:
                        profile['interests'].append(interest)
    
    def _add_welcome_message(self, response: str) -> str:
        """Add welcome message for new users"""
        welcomes = [
            "Olá! Bem-vindo! ",
            "Oi! É um prazer te ajudar! ",
            "Olá! Fico feliz em conversar com você! "
        ]
        import random
        return random.choice(welcomes) + response
    
    def _make_informal(self, response: str) -> str:
        """Make response more informal for regular users"""
        # Simple replacements
        informal_map = {
            'Olá': 'Oi',
            'você': 'vc',
            'Por favor': 'Por favor',
        }
        
        for formal, informal in informal_map.items():
            response = response.replace(formal, informal)
        
        return response
    
    def _add_interest_context(self, response: str, interests: List[str]) -> str:
        """Add context based on user interests"""
        # Could add relevant suggestions based on interests
        return response
    
    def _collaborative_personalization(self, response: str, profile: Dict) -> str:
        """Personalize using collaborative filtering approach"""
        # Would use similar users' preferences
        return response
    
    def _content_based_personalization(self, response: str, profile: Dict) -> str:
        """Personalize based on user's content preferences"""
        # Adapt response style based on user's past interactions
        return response
    
    def _hybrid_personalization(self, response: str, profile: Dict) -> str:
        """Combine collaborative and content-based"""
        response = self._collaborative_personalization(response, profile)
        response = self._content_based_personalization(response, profile)
        return response
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get user profile"""
        return self.user_profiles.get(user_id, {})
    
    def reset_user_profile(self, user_id: str):
        """Reset user profile"""
        if user_id in self.user_profiles:
            del self.user_profiles[user_id]
            logger.info(f"Reset profile for user: {user_id}")


# Singleton instance
_instance = None

def get_personalization_step_service():
    global _instance
    if _instance is None:
        _instance = PersonalizationStepService()
    return _instance
