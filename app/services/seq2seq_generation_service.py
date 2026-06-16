"""
Serviço de geração de texto usando Seq2Seq LSTM (Level 2).
Gera respostas naturais a partir de fatos estruturados da KB.

Arquitetura:
- Encoder LSTM: processa fatos estruturados
- Decoder LSTM com Attention: gera texto natural
- Fallback para templates (Level 1) se modelo não disponível

Level 2 da evolução generativa:
Level 1: Templates + entity extraction (atual backup)
Level 2: Seq2Seq LSTM (este módulo)
Level 3: Transformer fine-tuned (futuro)
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.core.logger import get_logger

logger = get_logger(__name__)

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    KERAS_AVAILABLE = True
    KerasModel = keras.Model
except ImportError:
    logger.warning("TensorFlow/Keras não disponível. Instale com: pip install tensorflow")
    KERAS_AVAILABLE = False
    KerasModel = type(None)  # Placeholder


class Seq2SeqGenerationService:
    """
    Serviço de geração Seq2Seq LSTM para respostas de astrologia.
    
    Gera texto natural a partir de fatos estruturados mantendo precisão factual.
    """
    
    def __init__(self):
        backend_dir = Path(__file__).parent.parent.parent
        self.models_dir = backend_dir / "data" / "models" / "seq2seq"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Modelo LSTM
        self.encoder_model: Optional[KerasModel] = None
        self.decoder_model: Optional[KerasModel] = None
        
        # Tokenizers
        self.input_tokenizer: Optional[Tokenizer] = None
        self.output_tokenizer: Optional[Tokenizer] = None
        
        # Vocabulário
        self.max_input_length = 50
        self.max_output_length = 100
        
        # Configuração
        self.latent_dim = 256  # Dimensão do LSTM
        self.embedding_dim = 128
        
        logger.info("Seq2SeqGenerationService initialized (Level 2)")
    
    def generate_from_facts(
        self,
        signo: str,
        topic: str,
        facts: Dict[str, str],
        temperature: float = 0.8
    ) -> Optional[str]:
        """
        Gera resposta natural a partir de fatos estruturados.
        
        Args:
            signo: Nome do signo
            topic: Tópico (elemento, planeta, caracteristicas, etc.)
            facts: Dicionário com fatos da KB
            temperature: Controle de aleatoriedade (0.0-1.0)
            
        Returns:
            Texto gerado ou None se modelo não disponível
        """
        logger.info(f"generate_from_facts() chamado - signo={signo}, topic={topic}, facts={facts}")
        
        if not self._is_model_loaded():
            logger.warning("Modelo Seq2Seq não carregado, usando fallback")
            return None
        
        logger.info("Modelos carregados OK, tentando gerar...")
        
        try:
            # Construir input estruturado
            input_text = self._build_input_text(signo, topic, facts)
            logger.info(f"Input estruturado: {input_text}")
            
            # Gerar resposta
            generated_text = self._generate_text(input_text, temperature)
            logger.info(f"Texto gerado: {generated_text}")
            
            if generated_text:
                logger.info(f"Gerado texto Seq2Seq: '{generated_text[:50]}...'")
                return generated_text
            else:
                logger.warning("_generate_text() retornou None ou vazio")
                return None
            
        except Exception as e:
            logger.error(f"Erro ao gerar com Seq2Seq: {e}", exc_info=True)
            return None
    
    def _build_input_text(self, signo: str, topic: str, facts: Dict[str, str]) -> str:
        """
        Constrói texto de input estruturado para o encoder.
        
        Formato NOVO (v2 - tokenizável): "signo áries topic elemento facts fogo"
        (minúsculas, espaçado para tokenização palavra-por-palavra)
        
        Formato ANTIGO (v1 - bug): "SIGNO:Áries TOPIC:elemento FACTS:Fogo"
        (tokens compostos que impediam generalização)
        
        Importante: Durante o treino, cada exemplo tinha apenas O VALOR do tópico,
        não múltiplos valores concatenados. Ex: "facts fogo" não "facts texto longo fogo mais texto"
        """
        # Usar apenas o valor correspondente ao tópico (como no treino)
        fact_value = facts.get(topic, "")
        
        # Fallback: se não tem o tópico exato, tenta pegar o primeiro valor não-texto_original
        if not fact_value or fact_value == "dados não encontrados":
            for key, value in facts.items():
                if key != 'texto_original' and value:
                    fact_value = value
                    break
        
        # NOVO FORMATO: minúsculas, espaçado (permite tokenização palavra-por-palavra)
        input_text = f"signo {signo.lower()} topic {topic} facts {str(fact_value).lower()}"
        return input_text
    
    def _generate_text(self, input_text: str, temperature: float) -> str:
        """
        Gera texto usando os modelos encoder-decoder.
        
        Args:
            input_text: Texto de input estruturado
            temperature: Controle de aleatoriedade
            
        Returns:
            Texto gerado pelo decoder
        """
        if not KERAS_AVAILABLE or not self.encoder_model or not self.decoder_model:
            raise RuntimeError("Modelos não carregados")
        
        # Tokenizar input
        input_seq = self.input_tokenizer.texts_to_sequences([input_text])
        input_seq = pad_sequences(input_seq, maxlen=self.max_input_length, padding='post')
        
        logger.info(f"[DEBUG-GENERATION] Input tokenized: {input_seq[0][:10]}... (shape={input_seq.shape})")
        
        # Encoder: obter estado
        states_value = self.encoder_model.predict(input_seq, verbose=0)
        logger.info(f"[DEBUG-GENERATION] Encoder states shape: {[s.shape for s in states_value]}")
        
        # Decoder: gerar token por token
        target_seq = np.zeros((1, 1))
        start_token_idx = self.output_tokenizer.word_index.get('<start>', 1)
        target_seq[0, 0] = start_token_idx
        logger.info(f"[DEBUG-GENERATION] Start token index: {start_token_idx}")
        
        generated_text = []
        for step in range(self.max_output_length):
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value,
                verbose=0
            )
            
            # Selecionar próximo token
            if temperature == 0.0:
                # Greedy decoding: escolher token com maior probabilidade
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
            else:
                # Temperature sampling
                output_tokens = output_tokens[0, -1, :] / temperature
                exp_tokens = np.exp(output_tokens - np.max(output_tokens))
                probs = exp_tokens / np.sum(exp_tokens)
                
                # Amostrar próximo token
                sampled_token_index = np.random.choice(len(probs), p=probs)
            
            # Converter para palavra
            sampled_word = None
            for word, index in self.output_tokenizer.word_index.items():
                if index == sampled_token_index:
                    sampled_word = word
                    break
            
            if step < 5:  # Log primeiras 5 palavras
                logger.info(f"[DEBUG-GENERATION] Step {step}: token_idx={sampled_token_index}, word='{sampled_word}'")
            
            if sampled_word is None or sampled_word == '<end>':
                logger.info(f"[DEBUG-GENERATION] Stopping: sampled_word={sampled_word}")
                break
            
            if sampled_word != '<start>':
                generated_text.append(sampled_word)
            
            # Atualizar target sequence
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            
            # Atualizar estados
            states_value = [h, c]
        
        result = ' '.join(generated_text)
        logger.info(f"[DEBUG-GENERATION] Generated {len(generated_text)} words: {result[:100]}...")
        return result
    
    def train_model(
        self,
        training_data: List[Tuple[str, str]],
        epochs: int = 50,
        batch_size: int = 64,
        validation_split: float = 0.1
    ) -> Dict[str, any]:
        """
        Treina o modelo Seq2Seq com dados de treino.
        
        Args:
            training_data: Lista de tuplas (input_facts, output_text)
            epochs: Número de épocas
            batch_size: Tamanho do batch
            validation_split: Fração para validação
            
        Returns:
            Histórico de treino
        """
        if not KERAS_AVAILABLE:
            raise RuntimeError("TensorFlow não disponível")
        
        logger.info(f"Iniciando treino Seq2Seq com {len(training_data)} exemplos")
        
        # Separar inputs e outputs
        input_texts = [pair[0] for pair in training_data]
        output_texts = [f"<start> {pair[1]} <end>" for pair in training_data]
        
        # Criar tokenizers
        self.input_tokenizer = Tokenizer(filters='', lower=False)
        self.input_tokenizer.fit_on_texts(input_texts)
        
        self.output_tokenizer = Tokenizer(filters='', lower=True)
        self.output_tokenizer.fit_on_texts(output_texts)
        
        # Converter para sequências
        input_sequences = self.input_tokenizer.texts_to_sequences(input_texts)
        output_sequences = self.output_tokenizer.texts_to_sequences(output_texts)
        
        # Padding
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_input_length, padding='post')
        output_sequences = pad_sequences(output_sequences, maxlen=self.max_output_length, padding='post')
        
        # Criar targets (output deslocado em 1)
        decoder_input_data = output_sequences[:, :-1]
        decoder_target_data = output_sequences[:, 1:]
        
        # Vocabulários
        num_encoder_tokens = len(self.input_tokenizer.word_index) + 1
        num_decoder_tokens = len(self.output_tokenizer.word_index) + 1
        
        logger.info(f"Vocabulário encoder: {num_encoder_tokens} tokens")
        logger.info(f"Vocabulário decoder: {num_decoder_tokens} tokens")
        
        # Construir modelo de treino
        training_model = self._build_training_model(
            num_encoder_tokens,
            num_decoder_tokens
        )
        
        # Treinar
        history = training_model.fit(
            [input_sequences, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1
        )
        
        # Salvar modelo de treino e vocabulários
        self._save_training_model(training_model, num_encoder_tokens, num_decoder_tokens)
        
        # Construir modelos de inferência a partir do modelo salvo
        self._build_inference_models_from_saved(num_encoder_tokens, num_decoder_tokens)
        
        # Salvar modelos de inferência
        self._save_models()
        
        logger.info("Treino Seq2Seq concluído!")
        return history.history
    
    def _build_training_model(self, num_encoder_tokens: int, num_decoder_tokens: int) -> KerasModel:
        """Constrói o modelo de treino (encoder + decoder juntos)"""
        # Encoder
        encoder_inputs = layers.Input(shape=(None,))
        encoder_embedding = layers.Embedding(num_encoder_tokens, self.embedding_dim)(encoder_inputs)
        encoder_lstm = layers.LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = layers.Input(shape=(None,))
        decoder_embedding = layers.Embedding(num_decoder_tokens, self.embedding_dim)(decoder_inputs)
        decoder_lstm = layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Modelo completo
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _save_training_model(self, training_model: KerasModel, num_encoder_tokens: int, num_decoder_tokens: int):
        """Salva o modelo de treino completo e metadados"""
        training_model_path = self.models_dir / "training_model.h5"
        training_model.save(training_model_path)
        
        metadata = {
            'num_encoder_tokens': num_encoder_tokens,
            'num_decoder_tokens': num_decoder_tokens
        }
        metadata_path = self.models_dir / "training_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Salvar tokenizers também
        tokenizers = {
            'input_tokenizer': self.input_tokenizer,
            'output_tokenizer': self.output_tokenizer,
            'max_input_length': self.max_input_length,
            'max_output_length': self.max_output_length,
            'latent_dim': self.latent_dim,
            'embedding_dim': self.embedding_dim
        }
        tokenizer_path = self.models_dir / "tokenizers.pkl"
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizers, f)
        
        logger.info(f"Modelo de treino salvo em {training_model_path}")
        logger.info(f"Tokenizers salvos em {tokenizer_path}")
    
    def _build_inference_models_from_saved(self, num_encoder_tokens: int, num_decoder_tokens: int):
        """Constrói modelos de inferência carregando e extraindo do modelo de treino salvo"""
        training_model_path = self.models_dir / "training_model.h5"
        training_model = keras.models.load_model(training_model_path)
        
        # Forçar build das camadas passando dados dummy
        import numpy as np
        dummy_encoder_input = np.zeros((1, 10), dtype=np.int32)
        dummy_decoder_input = np.zeros((1, 10), dtype=np.int32)
        training_model.predict([dummy_encoder_input, dummy_decoder_input], verbose=0)
        
        logger.info(f"Camadas do modelo de treino:")
        for i, layer in enumerate(training_model.layers):
            weights = layer.get_weights()
            logger.info(f"  Layer {i} ({layer.name}): {len(weights)} weight arrays")
        
        # Encoder de inferência
        encoder_inputs = layers.Input(shape=(None,))
        encoder_embedding = layers.Embedding(num_encoder_tokens, self.embedding_dim)
        encoder_embedded = encoder_embedding(encoder_inputs)
        
        encoder_lstm = layers.LSTM(self.latent_dim, return_state=True)
        _, state_h, state_c = encoder_lstm(encoder_embedded)
        
        self.encoder_model = keras.Model(encoder_inputs, [state_h, state_c])
        
        # Copiar pesos do modelo de treino (corrigindo índices das camadas)
        encoder_embedding.set_weights(training_model.layers[2].get_weights())  # Layer 2 é embedding
        encoder_lstm.set_weights(training_model.layers[4].get_weights())       # Layer 4 é LSTM
        
        # Decoder de inferência
        decoder_inputs = layers.Input(shape=(None,))
        decoder_state_input_h = layers.Input(shape=(self.latent_dim,))
        decoder_state_input_c = layers.Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_embedding = layers.Embedding(num_decoder_tokens, self.embedding_dim)
        decoder_embedded = decoder_embedding(decoder_inputs)
        
        decoder_lstm = layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedded,
            initial_state=decoder_states_inputs
        )
        
        decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        decoder_states = [state_h, state_c]
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
        
        # Copiar pesos do modelo de treino (corrigindo índices das camadas)
        decoder_embedding.set_weights(training_model.layers[3].get_weights())  # Layer 3 é embedding_1
        decoder_lstm.set_weights(training_model.layers[5].get_weights())       # Layer 5 é lstm_1
        decoder_dense.set_weights(training_model.layers[6].get_weights())      # Layer 6 é dense
        
        logger.info("Modelos de inferência construídos a partir do modelo de treino")
    
    def _build_inference_models(self, training_model: KerasModel, num_encoder_tokens: int, num_decoder_tokens: int):
        """Constrói modelos de inferência separados copiando pesos do modelo treinado"""
        # Encoder de inferência
        encoder_inputs = layers.Input(shape=(None,))
        encoder_embedding = layers.Embedding(num_encoder_tokens, self.embedding_dim)
        encoder_embedded = encoder_embedding(encoder_inputs)
        
        encoder_lstm = layers.LSTM(self.latent_dim, return_state=True)
        _, state_h, state_c = encoder_lstm(encoder_embedded)
        
        self.encoder_model = keras.Model(encoder_inputs, [state_h, state_c])
        
        # Copiar pesos do encoder treinado
        encoder_embedding.set_weights(training_model.layers[1].get_weights())
        encoder_lstm.set_weights(training_model.layers[2].get_weights())
        
        # Decoder de inferência
        decoder_inputs = layers.Input(shape=(None,))
        decoder_state_input_h = layers.Input(shape=(self.latent_dim,))
        decoder_state_input_c = layers.Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_embedding = layers.Embedding(num_decoder_tokens, self.embedding_dim)
        decoder_embedded = decoder_embedding(decoder_inputs)
        
        decoder_lstm = layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedded,
            initial_state=decoder_states_inputs
        )
        
        decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        decoder_states = [state_h, state_c]
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
        
        # Copiar pesos do decoder treinado
        decoder_embedding.set_weights(training_model.layers[3].get_weights())
        decoder_lstm.set_weights(training_model.layers[4].get_weights())
        decoder_dense.set_weights(training_model.layers[5].get_weights())
    
    def _save_models(self):
        """Salva modelos e tokenizers"""
        try:
            # Salvar modelos Keras
            encoder_path = self.models_dir / "encoder_model.h5"
            decoder_path = self.models_dir / "decoder_model.h5"
            
            self.encoder_model.save(encoder_path)
            self.decoder_model.save(decoder_path)
            
            # Salvar tokenizers
            tokenizers = {
                'input_tokenizer': self.input_tokenizer,
                'output_tokenizer': self.output_tokenizer,
                'max_input_length': self.max_input_length,
                'max_output_length': self.max_output_length,
                'latent_dim': self.latent_dim,
                'embedding_dim': self.embedding_dim
            }
            
            tokenizer_path = self.models_dir / "tokenizers.pkl"
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizers, f)
            
            logger.info(f"Modelos salvos em {self.models_dir}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelos: {e}")
            raise
    
    def load_models(self) -> bool:
        """
        Carrega modelos e tokenizers salvos.
        
        Returns:
            True se carregou com sucesso, False caso contrário
        """
        if not KERAS_AVAILABLE:
            logger.warning("TensorFlow não disponível")
            return False
        
        try:
            encoder_path = self.models_dir / "encoder_model.h5"
            decoder_path = self.models_dir / "decoder_model.h5"
            tokenizer_path = self.models_dir / "tokenizers.pkl"
            
            logger.info(f"Verificando arquivos: encoder={encoder_path.exists()}, decoder={decoder_path.exists()}, tokenizers={tokenizer_path.exists()}")
            
            if not encoder_path.exists() or not decoder_path.exists() or not tokenizer_path.exists():
                logger.info("Modelos Seq2Seq não encontrados")
                return False
            
            # Carregar modelos
            logger.info("Carregando encoder...")
            self.encoder_model = keras.models.load_model(encoder_path)
            logger.info("Carregando decoder...")
            self.decoder_model = keras.models.load_model(decoder_path)
            
            # Carregar tokenizers
            logger.info("Carregando tokenizers...")
            with open(tokenizer_path, 'rb') as f:
                tokenizers = pickle.load(f)
            
            logger.info(f"Tokenizers carregados: {list(tokenizers.keys())}")
            logger.info(f"input_tokenizer type: {type(tokenizers.get('input_tokenizer'))}")
            logger.info(f"output_tokenizer type: {type(tokenizers.get('output_tokenizer'))}")
            
            self.input_tokenizer = tokenizers['input_tokenizer']
            self.output_tokenizer = tokenizers['output_tokenizer']
            self.max_input_length = tokenizers['max_input_length']
            self.max_output_length = tokenizers['max_output_length']
            self.latent_dim = tokenizers['latent_dim']
            self.embedding_dim = tokenizers['embedding_dim']
            
            logger.info(f"Após atribuição: self.input_tokenizer={self.input_tokenizer is not None}, self.output_tokenizer={self.output_tokenizer is not None}")
            logger.info("Modelos Seq2Seq carregados com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {e}", exc_info=True)
            return False
    
    def _is_model_loaded(self) -> bool:
        """Verifica se os modelos estão carregados"""
        result = (
            KERAS_AVAILABLE and
            self.encoder_model is not None and
            self.decoder_model is not None and
            self.input_tokenizer is not None and
            self.output_tokenizer is not None
        )
        
        if not result:
            logger.warning(f"_is_model_loaded() falhou: KERAS_AVAILABLE={KERAS_AVAILABLE}, "
                          f"encoder={self.encoder_model is not None}, "
                          f"decoder={self.decoder_model is not None}, "
                          f"input_tokenizer={self.input_tokenizer is not None}, "
                          f"output_tokenizer={self.output_tokenizer is not None}")
        
        return result
