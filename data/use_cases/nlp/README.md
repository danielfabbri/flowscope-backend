# 🤖 NLP Use Case: Product Review Sentiment Classification

## 📋 Overview
Este use case demonstra uma pipeline completa de NLP para classificar sentimentos de reviews de produtos em **3 categorias**: Positive, Negative, Neutral.

## 🎯 Objetivo
Treinar um modelo de Machine Learning capaz de:
- Ler reviews de produtos em texto natural
- Processar e limpar o texto
- Extrair features relevantes usando técnicas de NLP
- Classificar automaticamente o sentimento do review

## 📊 Dataset
**Arquivo**: `product_reviews_train.csv`

### Colunas:
- `review_id`: ID único do review
- `product_name`: Nome do produto avaliado
- `category`: Categoria do produto (electronics, clothing, home)
- `rating`: Avaliação em estrelas (1-5)
- `review_text`: **Texto do review** (entrada principal para NLP)
- `review_date`: Data do review
- `verified_purchase`: Se é compra verificada (boolean)
- `helpful_votes`: Número de votos "útil"
- `sentiment`: **Target** (positive, negative, neutral)

### Estatísticas:
- **Total de reviews**: 500
- **Distribuição**:
  - Positive: 40% (~200 reviews)
  - Negative: 35% (~175 reviews)
  - Neutral: 25% (~125 reviews)

## 🔄 Pipeline NLP - Passo a Passo

### Step 1: 📥 **Ingestão de Dados**
**Tipo**: `ingestion`

Carrega o arquivo CSV com os reviews.

**Config**:
```json
{
  "ingestion_type": "file_upload",
  "file_format": "csv",
  "file_path": "data/use_cases/nlp/product_reviews_train.csv"
}
```

---

### Step 2: 📊 **Data Profiling**
**Tipo**: `profiling`

Analisa as características do dataset para entender a distribuição.

**Config**:
```json
{
  "generate_report": true
}
```

**O que observar**:
- Distribuição das classes (sentiment)
- Tamanho médio dos reviews
- Palavras mais frequentes
- Valores ausentes

---

### Step 3: 🔤 **Seleção de Colunas**
**Tipo**: `column_selection`

Seleciona apenas as colunas relevantes para o modelo.

**Config**:
```json
{
  "columns_to_keep": "review_text,rating,verified_purchase,helpful_votes,sentiment"
}
```

**Por que**: Removemos colunas irrelevantes (review_id, product_name, category, review_date) que não ajudam na predição.

---

### Step 4: 🧹 **Normalização de Texto**
**Tipo**: `text_normalization`

Limpa e normaliza o texto dos reviews.

**Config**:
```json
{
  "text_columns": "review_text",
  "lowercase": true,
  "remove_html": true,
  "remove_urls": true,
  "remove_emails": true,
  "remove_punctuation": false,
  "normalize_whitespace": true,
  "output_suffix": "_normalized"
}
```

**Resultado**: Nova coluna `review_text_normalized` com texto limpo.

**Exemplo**:
- **Antes**: `<p>Honestly, Amazing battery life! Check www.reviews.com</p>`
- **Depois**: `honestly amazing battery life`

---

### Step 5: ✂️ **Tokenização**
**Tipo**: `tokenization`

Divide o texto em palavras (tokens).

**Config**:
```json
{
  "text_columns": "review_text_normalized",
  "method": "word",
  "min_token_length": 2,
  "output_suffix": "_tokens"
}
```

**Resultado**: Coluna `review_text_normalized_tokens` com lista de palavras.

**Exemplo**:
- **Antes**: `honestly amazing battery life`
- **Depois**: `["honestly", "amazing", "battery", "life"]`

---

### Step 6: 🚫 **Remover Stop Words**
**Tipo**: `stopwords_removal`

Remove palavras comuns que não agregam significado.

**Config**:
```json
{
  "text_columns": "review_text_normalized_tokens",
  "language": "english",
  "input_format": "list",
  "output_suffix": "_filtered"
}
```

**Resultado**: Coluna `review_text_normalized_tokens_filtered`.

**Exemplo**:
- **Antes**: `["honestly", "amazing", "battery", "life"]`
- **Depois**: `["amazing", "battery", "life"]`

**Stop words removidas**: "honestly", "the", "is", "at", "a", "an", etc.

---

### Step 7: 🌱 **Lemmatização**
**Tipo**: `stemming_lemmatization`

Reduz palavras às suas formas base.

**Config**:
```json
{
  "text_columns": "review_text_normalized_tokens_filtered",
  "method": "lemmatize",
  "language": "english",
  "input_format": "list",
  "output_suffix": "_lemmatized"
}
```

**Resultado**: Coluna `review_text_normalized_tokens_filtered_lemmatized`.

**Exemplo**:
- **Antes**: `["running", "batteries", "worked"]`
- **Depois**: `["run", "battery", "work"]`

---

### Step 8: 📊 **Vetorização TF-IDF**
**Tipo**: `text_vectorization`

Converte texto em features numéricas para o modelo.

**Config**:
```json
{
  "text_columns": "review_text_normalized_tokens_filtered_lemmatized",
  "method": "tfidf",
  "max_features": 200,
  "min_df": 2,
  "max_df": 0.9,
  "ngram_range": "[1, 2]",
  "input_format": "tokens",
  "feature_prefix": "tfidf"
}
```

**Resultado**: 200 novas colunas numéricas (tfidf_0 até tfidf_199).

**O que é TF-IDF**:
- **TF** (Term Frequency): Quão frequente é uma palavra no documento
- **IDF** (Inverse Document Frequency): Quão rara é a palavra no corpus
- Palavras importantes ficam com valores altos

---

### Step 9: 😊 **Análise de Sentimento (Features Adicionais)**
**Tipo**: `sentiment_analysis`

Adiciona métricas de sentimento como features extras.

**Config**:
```json
{
  "text_columns": "review_text_normalized",
  "metrics": "polarity,subjectivity",
  "input_format": "text"
}
```

**Resultado**: Novas colunas:
- `review_text_normalized_polarity`: -1.0 (negativo) a +1.0 (positivo)
- `review_text_normalized_subjectivity`: 0.0 (objetivo) a 1.0 (subjetivo)

---

### Step 10: 🎯 **Feature Engineering (Opcional)**
**Tipo**: `feature_engineering`

Cria features adicionais dos dados numéricos.

**Config**:
```json
{
  "transformation_type": "scaling",
  "scaling_columns": "rating,helpful_votes",
  "scaling_method": "standard"
}
```

**Por que**: Normaliza ratings e votes para mesma escala das features TF-IDF.

---

### Step 11: 🤖 **Treinamento do Modelo**
**Tipo**: `ml_model_training`

Treina modelo de classificação.

**Config**:
```json
{
  "model_name": "product_review_sentiment_classifier",
  "problem_type": "classification",
  "target_column": "sentiment",
  "algorithm": "random_forest",
  "test_size": 0.2,
  "n_estimators": 200,
  "max_depth": 20,
  "class_weight": "balanced"
}
```

**Resultado**: Modelo salvo em `data/models/product_review_sentiment_classifier_metadata.json`.

**Métricas esperadas**:
- Accuracy: ~85-90%
- F1-Score: ~0.85
- Precision/Recall balanceados

---

### Step 12: 💾 **Saída**
**Tipo**: `output`

Salva os dados processados.

**Config**:
```json
{
  "include_metadata": true
}
```

---

## 🎯 Como Usar no Chat

Depois de treinar o modelo, você pode fazer predições via chat:

**Exemplo 1**:
```
User: "Analise este review: This product is amazing! Battery lasts forever and camera is excellent."

Bot: Analisando o review...
Sentimento: POSITIVE (95% confiança)
Polarity: 0.85
Subjectivity: 0.78
```

**Exemplo 2**:
```
User: "Classifique: Terrible quality, broke after 2 days. Total waste of money."

Bot: Sentimento: NEGATIVE (98% confiança)
Polarity: -0.92
```

---

## 📈 Entendendo os Steps

### Por que cada step é importante?

1. **Normalização**: Texto cru tem ruído (HTML, URLs) que confunde o modelo
2. **Tokenização**: Modelos entendem palavras individuais, não frases
3. **Stop Words**: "the", "is", "a" aparecem em todos os reviews, não diferenciam sentimento
4. **Lemmatização**: "working", "worked", "works" → "work" (reduz vocabulário)
5. **TF-IDF**: Transforma texto em números que o modelo pode processar
6. **Sentiment Features**: Dá dicas extras sobre a polaridade do texto

### Fluxo de Dados:

```
Texto Cru → Limpo → Tokens → Sem Stop Words → Lemmatizado → Vetorizado → Modelo → Predição
```

---

## 🚀 Próximos Passos

1. **Gerar o dataset**: Execute `python generate_product_reviews_nlp.py`
2. **Criar a pipeline**: Use o frontend para montar os steps acima
3. **Treinar o modelo**: Execute a pipeline completa
4. **Testar no chat**: Use reviews novos para validar

---

## 🔍 Troubleshooting

**Problema**: "Accuracy muito baixa (<70%)"
- Solução: Aumente `max_features` no TF-IDF (200 → 500)

**Problema**: "Modelo sempre prediz mesma classe"
- Solução: Use `class_weight: "balanced"` no treinamento

**Problema**: "Muito tempo para processar"
- Solução: Reduza `max_features` ou use menos reviews (sample)

---

## 📚 Referências

- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **Lemmatization**: WordNet Lemmatizer
- **Stop Words**: NLTK English Stop Words
- **Sentiment**: TextBlob Sentiment Analysis

---

**✨ Boa sorte com seu modelo de NLP!** 🚀
