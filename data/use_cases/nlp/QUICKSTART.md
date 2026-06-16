# 🚀 Quick Start Guide - NLP Sentiment Classification

## ⚡ Modo Rápido (5 minutos)

### Opção 1: Importar Pipeline Pronta

1. **Abra o frontend**: http://localhost:5173/

2. **Vá para Pipelines** → "Import Pipeline"

3. **Selecione o arquivo**: `data/use_cases/nlp/nlp_sentiment_pipeline.json`

4. **Clique em "Run Pipeline"** ▶️

5. **Aguarde o treinamento** (~30 segundos)

6. **Modelo pronto!** ✅ Salvo como `product_review_sentiment_classifier`

---

## 🎓 Modo Aprendizado (Criar manualmente)

### Passo 1: Criar Nova Pipeline

1. Vá para **"Create New Pipeline"**
2. Nome: `NLP Sentiment Classifier`
3. Descrição: `Classifica sentimento de reviews usando NLP`

### Passo 2: Adicionar Steps (nesta ordem)

#### 📥 Step 1: Ingestão de Dados
- **Tipo**: Ingestão de Dados
- **Config**:
  - Ingestion Type: `File Upload`
  - File Format: `csv`
  - File Path: `data/use_cases/nlp/product_reviews_train.csv`

#### 📊 Step 2: Data Profiling
- **Tipo**: Profiling (adicione via drag no canvas)
- **Config**: Deixe padrão (generate_report: true)

#### ✂️ Step 3: Seleção de Colunas
- **Tipo**: Seleção de Colunas
- **Config**:
  - Columns to Keep: `review_text,rating,verified_purchase,helpful_votes,sentiment`

#### 🔤 Step 4: Normalização de Texto ⭐
- **Tipo**: Normalização de Texto (categoria NLP)
- **Config**:
  - Text Columns: `review_text`
  - ✅ Lowercase
  - ✅ Remove HTML
  - ✅ Remove URLs
  - ✅ Remove Emails
  - ✅ Normalize Whitespace
  - Output Suffix: `_normalized`

#### ✂️ Step 5: Tokenização ⭐
- **Tipo**: Tokenização (categoria NLP)
- **Config**:
  - Text Columns: `review_text_normalized`
  - Method: `word`
  - Min Token Length: `2`

#### 🚫 Step 6: Remover Stop Words ⭐
- **Tipo**: Remover Stop Words (categoria NLP)
- **Config**:
  - Text Columns: `review_text_normalized_tokens`
  - Language: `english`

#### 🌱 Step 7: Lemmatização ⭐
- **Tipo**: Stemming/Lemmatização (categoria NLP)
- **Config**:
  - Text Columns: `review_text_normalized_tokens_filtered`
  - Method: `lemmatize`

#### 📊 Step 8: Vetorização TF-IDF ⭐
- **Tipo**: Vetorização TF-IDF (categoria NLP)
- **Config**:
  - Text Columns: `review_text_normalized_tokens_filtered_lemmatized`
  - Method: `tfidf`
  - Max Features: `200`
  - Min DF: `2`
  - Max DF: `0.9`

#### 😊 Step 9: Análise de Sentimento ⭐
- **Tipo**: Análise de Sentimento (categoria NLP)
- **Config**:
  - Text Columns: `review_text_normalized`
  - Metrics: `polarity,subjectivity`

#### 🎯 Step 10: Feature Engineering
- **Tipo**: Engenharia de Features
- **Config**:
  - Transformation Type: `scaling`
  - Scaling Columns: `rating,helpful_votes`
  - Scaling Method: `standard`

#### 🤖 Step 11: Treinamento
- **Tipo**: Model Training (categoria Machine Learning)
- **Config**:
  - Model Name: `product_review_sentiment_classifier`
  - Problem Type: `classification`
  - Target Column: `sentiment`
  - Algorithm: `random_forest`
  - N Estimators: `200`
  - Max Depth: `20`
  - Class Weight: `balanced`

#### 💾 Step 12: Saída
- **Tipo**: Saída de Dados
- **Config**: Padrão (include_metadata: true)

### Passo 3: Salvar e Executar

1. **Clique em "Create Pipeline"** (ou "Save" se editando)
2. **Clique em "Run Pipeline"** ▶️
3. **Monitore o progresso** na tela

---

## 📊 Resultados Esperados

Após executar a pipeline, você verá:

### Métricas do Modelo:
```
✅ Accuracy: ~85-90%
✅ Precision: ~0.85
✅ Recall: ~0.85
✅ F1-Score: ~0.85

Classes:
- positive: F1 ~0.90
- negative: F1 ~0.85
- neutral: F1 ~0.75
```

### Arquivos Gerados:
- ✅ `data/models/product_review_sentiment_classifier_metadata.json`
- ✅ `data/models/product_review_sentiment_classifier.pkl`
- ✅ `data/pipeline_data/[pipeline_id]/step_12_output.csv`

---

## 🎯 Usando o Modelo no Chat

Depois do treinamento, vá para o Chat e teste:

### Exemplo 1 - Review Positivo:
```
👤 User: Analise este review: "Amazing product! Battery lasts all day and camera is excellent. Highly recommend!"

🤖 Bot: 
Sentimento: POSITIVE ✅
Confiança: 95%
Polarity: 0.82
Subjectivity: 0.75
```

### Exemplo 2 - Review Negativo:
```
👤 User: O que acha de: "Terrible quality, broke after 2 days. Total waste of money!"

🤖 Bot:
Sentimento: NEGATIVE ❌
Confiança: 98%
Polarity: -0.88
Subjectivity: 0.80
```

### Exemplo 3 - Review Neutro:
```
👤 User: Classifique: "It works as expected, nothing special."

🤖 Bot:
Sentimento: NEUTRAL 😐
Confiança: 75%
Polarity: 0.0
Subjectivity: 0.35
```

---

## 🐛 Troubleshooting

### Erro: "Column 'review_text_normalized' not found"
❌ **Causa**: Step de normalização não executou
✅ **Solução**: Verifique se o step 4 está configurado corretamente

### Erro: "ValueError: could not convert string to float"
❌ **Causa**: TF-IDF recebeu formato errado
✅ **Solução**: Verifique se `input_format: "tokens"` está configurado no step 8

### Erro: "Accuracy muito baixa (<70%)"
❌ **Causa**: Poucas features ou desequilíbrio de classes
✅ **Solução**: 
- Aumente `max_features` para 500 no TF-IDF
- Garanta `class_weight: "balanced"` no treinamento

### Pipeline demora muito
❌ **Causa**: Dataset grande ou muitas features
✅ **Solução**:
- Reduza `max_features` no TF-IDF para 100
- Use apenas 250 reviews (sample do dataset)

---

## 📚 Entendendo o Fluxo

```
Texto Cru
   ↓
[Normalização] → Remove HTML, URLs, lowercase
   ↓
[Tokenização] → Divide em palavras
   ↓
[Stop Words] → Remove "the", "is", "at"
   ↓
[Lemmatização] → "running" → "run"
   ↓
[TF-IDF] → Converte em 200 números
   ↓
[Features Extra] → Adiciona polarity, subjectivity
   ↓
[Modelo] → Random Forest classifica
   ↓
Predição: Positive/Negative/Neutral
```

---

## 🎯 Próximos Experimentos

1. **Melhorar Accuracy**:
   - Adicione N-grams (step entre 7 e 8)
   - Aumente max_features para 500
   - Teste outros algoritmos (gradient_boosting)

2. **Adicionar Features**:
   - Quantidade de exclamações (!!!)
   - Quantidade de CAPS LOCK
   - Comprimento do review

3. **Multi-label**:
   - Classifique também a categoria (electronics, clothing, home)
   - Adicione target_column: "category"

---

**✨ Divirta-se explorando NLP!** 🚀

---

## 📖 Referências

- **Dataset**: 500 reviews sintéticos de produtos
- **Arquivo**: `data/use_cases/nlp/product_reviews_train.csv`
- **Pipeline**: `data/use_cases/nlp/nlp_sentiment_pipeline.json`
- **Documentação Completa**: `data/use_cases/nlp/README.md`
