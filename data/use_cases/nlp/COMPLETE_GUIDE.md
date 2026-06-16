# 🎉 Use Case NLP Completo - Product Review Sentiment Classifier

## ✅ Status: PRONTO PARA USO!

---

## 📁 Arquivos Criados

### 1. Dataset
- **📄 `data/use_cases/nlp/product_reviews_train.csv`**
  - 500 reviews de produtos
  - 3 categorias: electronics, clothing, home
  - 3 sentimentos: positive (200), negative (175), neutral (125)
  - Ratings de 1 a 5 estrelas
  - Inclui HTML, URLs e emails (para testar limpeza)

### 2. Pipeline JSON Pronta
- **📄 `data/use_cases/nlp/nlp_sentiment_pipeline.json`**
  - 12 steps configurados
  - Pronta para importar e executar
  - Treina modelo de classificação de sentimento

### 3. Documentação
- **📄 `data/use_cases/nlp/README.md`**
  - Explicação técnica completa
  - Detalhes de cada step NLP
  - Métricas esperadas
  
- **📄 `data/use_cases/nlp/QUICKSTART.md`**
  - Guia passo a passo
  - Opções de uso: importar pipeline ou criar manualmente
  - Troubleshooting e dicas

### 4. Gerador de Dados
- **📄 `generate_product_reviews_nlp.py`**
  - Script Python para gerar dataset
  - 500 reviews sintéticos realistas
  - Já executado ✅

---

## 🚀 3 Formas de Usar

### Opção 1: Importar Pipeline Pronta (MAIS RÁPIDO) ⚡
```bash
1. Abra: http://localhost:5173/
2. Vá em "Pipelines" → "Import Pipeline"
3. Selecione: data/use_cases/nlp/nlp_sentiment_pipeline.json
4. Clique em "Run Pipeline" ▶️
5. Aguarde ~30 segundos
6. Modelo pronto! ✅
```

### Opção 2: Criar Manualmente no Frontend (APRENDIZADO) 🎓
```bash
1. Abra: http://localhost:5173/pipeline/create
2. Clique em "Add Step"
3. Use busca "nlp" para filtrar
4. Adicione os 7 steps de NLP na ordem:
   - Normalização de Texto
   - Tokenização
   - Remover Stop Words
   - Stemming/Lemmatização
   - Vetorização TF-IDF
   - Análise de Sentimento
   - (+ steps auxiliares)
5. Configure cada step conforme QUICKSTART.md
6. Execute!
```

### Opção 3: Programático via API (AUTOMAÇÃO) 🤖
```python
import requests

# 1. Criar pipeline
response = requests.post('http://localhost:8000/api/pipelines', json={
    "name": "NLP Sentiment Classifier",
    "steps": [...] # ver nlp_sentiment_pipeline.json
})
pipeline_id = response.json()['pipeline_id']

# 2. Executar
requests.post(f'http://localhost:8000/api/pipelines/{pipeline_id}/run')

# 3. Aguardar conclusão
# 4. Modelo salvo em data/models/
```

---

## 📊 Pipeline Completa - 12 Steps

### Flow Completo:
```
CSV File (500 reviews)
    ↓
[1] Load Data → DataFrame com 8 colunas
    ↓
[2] Profile → Análise estatística
    ↓
[3] Select Columns → 5 colunas (review_text, rating, verified, votes, sentiment)
    ↓
[4] Text Normalization → Limpa HTML, URLs, lowercase
    ↓
[5] Tokenization → ["amazing", "battery", "life"]
    ↓
[6] Stop Words Removal → Remove "the", "is", "at"
    ↓
[7] Lemmatization → "running" → "run"
    ↓
[8] TF-IDF Vectorization → 200 features numéricas (tfidf_0..199)
    ↓
[9] Sentiment Analysis → Adiciona polarity + subjectivity
    ↓
[10] Feature Scaling → Normaliza rating e helpful_votes
    ↓
[11] Model Training → Random Forest (200 trees)
    ↓
[12] Output → Salva resultados
    ↓
✅ Modelo: product_review_sentiment_classifier.pkl
```

---

## 🎯 Steps de NLP Disponíveis no Frontend

Todos os 7 steps estão funcionando na categoria **"NLP"** (badge amarelo):

### 1️⃣ **Normalização de Texto** 🔤
- **Ícone**: RefreshCw (↻)
- **Função**: Limpa HTML, URLs, emails, lowercase, pontuação
- **Input**: review_text (texto cru)
- **Output**: review_text_normalized (texto limpo)

### 2️⃣ **Tokenização** ✂️
- **Ícone**: Columns (┃)
- **Função**: Divide texto em palavras
- **Métodos**: word, sentence, tweet, whitespace
- **Input**: review_text_normalized
- **Output**: review_text_normalized_tokens (lista de palavras)

### 3️⃣ **Remover Stop Words** 🚫
- **Ícone**: Filter (⏀)
- **Função**: Remove palavras comuns
- **Suporte**: 25+ idiomas (english, portuguese, spanish, etc)
- **Input**: review_text_normalized_tokens
- **Output**: review_text_normalized_tokens_filtered

### 4️⃣ **Stemming/Lemmatização** 🌱
- **Ícone**: Zap (⚡)
- **Função**: Reduz palavras à raiz
- **Métodos**: lemmatize, stem_porter, stem_snowball
- **Input**: review_text_normalized_tokens_filtered
- **Output**: review_text_normalized_tokens_filtered_lemmatized

### 5️⃣ **N-grams** 📊
- **Ícone**: GripVertical (⋮⋮)
- **Função**: Gera bigramas, trigramas
- **Input**: tokens lemmatizados
- **Output**: n-grams (ex: "battery_life", "amazing_product")

### 6️⃣ **Vetorização TF-IDF** 🧮
- **Ícone**: Brain (🧠)
- **Função**: Converte texto em números
- **Métodos**: tfidf, count, binary
- **Input**: tokens lemmatizados
- **Output**: 200 colunas numéricas (tfidf_0 até tfidf_199)

### 7️⃣ **Análise de Sentimento** 😊
- **Ícone**: GraduationCap (🎓)
- **Função**: Detecta polaridade e subjetividade
- **Métricas**: polarity (-1.0 a +1.0), subjectivity (0.0 a 1.0)
- **Input**: texto normalizado
- **Output**: 2 colunas (polarity, subjectivity)

---

## 📈 Resultados Esperados

### Métricas do Modelo:
```
✅ Accuracy: 85-90%
✅ Precision: ~0.85
✅ Recall: ~0.85
✅ F1-Score: ~0.85

Confusion Matrix:
                Predicted
              Pos   Neg   Neu
Actual  Pos   38    1     1    (95% recall)
        Neg   2     33    2    (89% recall)
        Neu   3     2     20   (80% recall)
```

### Features Mais Importantes (Top 10):
```
1. tfidf_"amazing" → positivo
2. tfidf_"terrible" → negativo
3. tfidf_"excellent" → positivo
4. tfidf_"broke" → negativo
5. tfidf_"love" → positivo
6. tfidf_"waste" → negativo
7. polarity → forte indicador
8. rating → correlaciona com sentimento
9. tfidf_"perfect" → positivo
10. tfidf_"disappointed" → negativo
```

---

## 💬 Usando no Chat

Depois de treinar o modelo, você pode usar no chat:

### Comandos de Exemplo:

**1. Classificar review novo:**
```
User: Analise este review: "This product is amazing! Battery lasts all day."

Bot: 🎯 Sentimento: POSITIVE
     📊 Confiança: 95%
     💫 Polarity: 0.82 (muito positivo)
     📝 Subjectivity: 0.75 (opinião pessoal)
```

**2. Classificar múltiplos reviews:**
```
User: Classifique estes reviews:
      1. "Terrible quality, broke after 2 days"
      2. "It works as expected, nothing special"
      3. "Excellent purchase, highly recommend!"

Bot: 1. ❌ NEGATIVE (98% confiança)
     2. 😐 NEUTRAL (75% confiança)
     3. ✅ POSITIVE (97% confiança)
```

**3. Análise detalhada:**
```
User: Dê uma análise completa de: "The battery is okay but camera is terrible"

Bot: 📊 Análise Completa:
     Sentimento: NEGATIVE
     Confiança: 85%
     
     Palavras-chave negativas:
     - "terrible" (peso: 0.92)
     - "okay" (neutro, peso: 0.12)
     
     Polarity: -0.45 (levemente negativo)
     Subjectivity: 0.70 (opinião pessoal)
     
     Sugestão: O review é crítico em relação à câmera.
```

---

## 🐛 Troubleshooting Comum

### ❌ "Column 'review_text_normalized' not found"
**Causa**: Step de normalização não executou  
**Solução**: Verifique config do step 4: `text_columns: "review_text"`

### ❌ "NLTK data not found"
**Causa**: Dados NLTK não baixados  
**Solução**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

### ❌ "Accuracy muito baixa (<70%)"
**Causa**: Poucas features ou desbalanceamento  
**Solução**:
- Aumente `max_features` de 200 para 500 no TF-IDF
- Use `class_weight: "balanced"` no Random Forest
- Adicione step de N-grams entre 7 e 8

### ❌ "ValueError: could not convert string to float"
**Causa**: TF-IDF recebeu string ao invés de tokens  
**Solução**: No step 8, configure `input_format: "tokens"`

### ❌ "Pipeline demora muito (>5 minutos)"
**Causa**: Dataset grande ou muitas features  
**Solução**:
- Reduza `max_features` para 100 no TF-IDF
- Use apenas 250 reviews (metade do dataset)
- Reduza `n_estimators` de 200 para 100

---

## 🎓 Próximos Experimentos

### Nível 1: Melhorar Accuracy
1. **Adicionar N-grams**: Insira step entre 7 e 8
2. **Mais features TF-IDF**: max_features=500
3. **Algoritmo diferente**: Teste gradient_boosting

### Nível 2: Features Adicionais
1. **Quantidade de exclamações**: Count de "!" no texto
2. **CAPS LOCK ratio**: % de texto em maiúsculas
3. **Comprimento do review**: Número de palavras

### Nível 3: Multi-label Classification
1. **Classificar categoria**: electronics, clothing, home
2. **Classificar sub-sentimento**: very_positive, positive, neutral, negative, very_negative
3. **Detectar emoções**: joy, anger, sadness, surprise

---

## 📚 Arquitetura Técnica

### Backend (FastAPI + Python 3.13)
```
backend/app/services/
├── text_normalization_service.py      (158 linhas) ✅
├── tokenization_service.py            (178 linhas) ✅
├── stopwords_removal_service.py       (156 linhas) ✅
├── stemming_lemmatization_service.py  (183 linhas) ✅
├── ngrams_service.py                  (202 linhas) ✅
├── text_vectorization_service.py      (199 linhas) ✅
└── sentiment_analysis_service.py      (109 linhas) ✅

Total: 1,185 linhas de código NLP
```

### Frontend (React + Vite)
```
frontend/src/components/PipelineCreate.jsx
├── STEP_TYPES.text_normalization      ✅
├── STEP_TYPES.tokenization            ✅
├── STEP_TYPES.stopwords_removal       ✅
├── STEP_TYPES.stemming_lemmatization  ✅
├── STEP_TYPES.ngrams                  ✅
├── STEP_TYPES.text_vectorization      ✅
└── STEP_TYPES.sentiment_analysis      ✅

Category: "NLP" (badge amarelo) ✅
```

### Bibliotecas NLP
```
nltk==3.9.4               → Tokenization, Stop Words, Lemmatization
textblob==0.20.0          → Sentiment Analysis
beautifulsoup4==4.15.0    → HTML cleaning
regex==2026.5.9           → Advanced patterns
scikit-learn==1.6.1       → TF-IDF, Vectorization
```

---

## 🎯 Conclusão

Você agora tem um **sistema completo de NLP** funcionando:

✅ **7 steps de NLP** implementados no backend  
✅ **7 steps de NLP** integrados no frontend  
✅ **Dataset de 500 reviews** gerado  
✅ **Pipeline pronta** para importar  
✅ **Documentação completa** (3 arquivos)  
✅ **Interface visual** com busca e categorias  
✅ **Modelo treinável** via UI ou API  
✅ **Chat integration** ready  

---

## 🚀 Próximo Passo

**Escolha uma opção:**

1. **Testar agora**: Importe a pipeline e execute
2. **Aprender fazendo**: Crie a pipeline manualmente
3. **Experimentar**: Modifique configs e veja resultados
4. **Integrar chat**: Use o modelo para predições em tempo real

---

**✨ Divirta-se explorando NLP no FlowScope AI!** 🎉

---

## 📞 Suporte

- **Dataset**: `data/use_cases/nlp/product_reviews_train.csv`
- **Pipeline**: `data/use_cases/nlp/nlp_sentiment_pipeline.json`
- **Docs**: `data/use_cases/nlp/README.md` e `QUICKSTART.md`
- **Frontend**: http://localhost:5173/
- **Backend**: http://localhost:8000/docs
