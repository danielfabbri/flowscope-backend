# 🏠 Pipeline de Regressão - Previsão de Preço de Imóveis

## 🎯 Objetivo

**Prever o preço de venda de imóveis** com base em suas características (tamanho, localização, condição, etc.) e **identificar imóveis com bom custo-benefício** (preço previsto > preço real = subvalorizados) para recomendar aos compradores.

**Resultado final:** Lista de imóveis subvalorizados com previsão de preço, dados do corretor e link para contato.

---

## 📊 O Problema de Negócio

Você é um analista de dados de uma imobiliária. Seu objetivo é:
1. **Treinar um modelo** que prevê preços baseado em características dos imóveis
2. **Identificar oportunidades**: Imóveis que estão sendo vendidos abaixo do preço previsto
3. **Facilitar o contato**: Adicionar informações do corretor para negociação rápida

---

## 🛠️ Passo a Passo da Pipeline

### **Step 1: Data Ingestion** 📥
**O que fazer:**
- Clique em "+ Create New Pipeline"
- Nome: `Property Price Prediction`
- Description: `Predict property sale prices and identify undervalued properties`
- Adicione o step **Data Ingestion**
- Clique em "Test Connection & Load Columns"

**Configuração:**
```
Data Source Type: JSON
File Path: data/use_cases/regression/property_sales.json
```

**O que esperar:**
- ✅ 100 imóveis carregados
- ✅ 22 colunas detectadas (property_id, property_type, bedrooms, sale_price, etc.)

---

### **Step 2: Column Selection** 🎯
**O que fazer:**
- Adicione step **Column Selection**
- Selecione as colunas relevantes para o modelo

**Configuração:**
```
Columns to Keep:
property_id
property_type
neighborhood
bedrooms
bathrooms
square_feet
lot_size
age
stories
condition
has_pool
has_fireplace
has_ac
has_basement
school_rating
distance_downtown
crime_rate
sale_price
```

**O que esperar:**
- ✅ 18 colunas mantidas (removemos heating_type, parking_type, year_built, days_on_market)
- ✅ Dados mais focados nas features relevantes

---

### **Step 3: Feature Engineering - Label Encoding (Property Type)** 🏷️
**O que fazer:**
- Adicione step **Feature Engineering**
- Transformação: **Label Encoding**

**Configuração:**
```
Transformation Type: Label Encoding
Columns to Encode: property_type
```

**O que esperar:**
- ✅ `property_type` convertido para números:
  - Apartment = 0
  - Condo = 1
  - House = 2
  - Townhouse = 3

---

### **Step 4: Feature Engineering - Label Encoding (Neighborhood)** 🏘️
**O que fazer:**
- Adicione step **Feature Engineering**
- Transformação: **Label Encoding**

**Configuração:**
```
Transformation Type: Label Encoding
Columns to Encode: neighborhood
```

**O que esperar:**
- ✅ `neighborhood` convertido para números (0-7)

---

### **Step 5: Feature Engineering - Label Encoding (Condition)** ⭐
**O que fazer:**
- Adicione step **Feature Engineering**
- Transformação: **Label Encoding**

**Configuração:**
```
Transformation Type: Label Encoding
Columns to Encode: condition
```

**O que esperar:**
- ✅ `condition` convertido para números:
  - Poor = 0, Fair = 1, Good = 2, Excellent = 3

---

### **Step 6: Feature Engineering - Normalization** 📏
**O que fazer:**
- Adicione step **Feature Engineering**
- Transformação: **Scaling (Normalization)**

**Configuração:**
```
Transformation Type: Scaling
Scaler Type: StandardScaler
Columns to Scale:
square_feet
lot_size
age
school_rating
distance_downtown
crime_rate
```

**O que esperar:**
- ✅ Valores numéricos normalizados (média=0, desvio=1)
- ✅ Exemplo: square_feet 2500 → 0.543, 1200 → -0.892

---

### **Step 7: ML Modeling - Random Forest Regression** 🤖
**O que fazer:**
- Adicione step **ML Modeling**
- Tipo: **Regression**

**Configuração:**
```
Model Type: Regression
Algorithm: Random Forest Regressor
Target Column: sale_price
Exclude Features: property_id
Test Size: 0.2
Random State: 42
N Estimators: 100
Max Depth: 15
```

**O que esperar:**
- ✅ Modelo treinado com 80 imóveis (20 para teste)
- ✅ Nova coluna: `predicted_price` (previsão do modelo)
- ✅ Nova coluna: `price_difference` (sale_price - predicted_price)
- ✅ Métricas:
  - MAE (Mean Absolute Error): ~$25,000 (quanto o modelo erra em média)
  - RMSE (Root Mean Squared Error): ~$35,000
  - R² Score: ~0.85-0.92 (85-92% de explicação da variância)
- ✅ Feature Importance: Top features que influenciam preço
  - Esperado: square_feet, neighborhood, condition, bedrooms

---

### **Step 8: Feature Engineering - Create Price Gap Column** 💰
**O que fazer:**
- Adicione step **Feature Engineering**
- Transformação: **Feature Creation**

**Configuração:**
```
Transformation Type: Feature Creation
Formula Name: price_gap
Formula: predicted_price - sale_price
```

**O que esperar:**
- ✅ Nova coluna `price_gap`:
  - **Positivo** = Imóvel subvalorizado (vale mais do que está sendo vendido)
  - **Negativo** = Imóvel sobrevalorizado
- Exemplo: 
  - sale_price = $300,000, predicted_price = $350,000 → price_gap = +$50,000 ✅
  - sale_price = $450,000, predicted_price = $400,000 → price_gap = -$50,000 ❌

---

### **Step 9: Row Filtering - Undervalued Properties** 🎯
**O que fazer:**
- Adicione step **Row Filtering**
- Filtrar apenas imóveis com bom custo-benefício

**Configuração:**
```
Filter Column: price_gap
Operator: >
Filter Value: 20000
Keep Matching: true
```

**O que esperar:**
- ✅ Apenas imóveis com `price_gap > $20,000` (subvalorizados em mais de $20K)
- ✅ Dataset reduzido para ~20-30 imóveis (os melhores negócios)

---

### **Step 10: Data Enrichment - Add Agent Details** 📞
**O que fazer:**
- Adicione step **Data Enrichment**
- JOIN com dados dos corretores

**Configuração:**
```
Source Type: CSV
Source Path: data/use_cases/regression/property_details.csv
Join Key (Left): property_id
Join Key (Right): property_id
Join Type: left
Columns to Add: (deixe vazio para adicionar todas)
```

**O que esperar:**
- ✅ Novas colunas adicionadas:
  - `agent_name`: Nome do corretor
  - `agent_phone`: Telefone para contato
  - `agent_email`: E-mail do corretor
  - `listing_url`: Link direto do imóvel

---

### **Step 11: Output - Export Results** 💾
**O que fazer:**
- Adicione step **Output**
- Exportar lista final

**Configuração:**
```
Output Type: CSV
Output Path: data/output/undervalued_properties.csv
```

**O que esperar:**
- ✅ Arquivo CSV gerado com colunas principais:
  - property_id, property_type, neighborhood, bedrooms, bathrooms, square_feet
  - sale_price, predicted_price, price_gap
  - agent_name, agent_phone, agent_email, listing_url
- ✅ Pronto para compartilhar com equipe de vendas!

---

## ▶️ Executar a Pipeline

1. Clique no botão **"Save Pipeline"** (salva configuração)
2. Clique no botão **"Run Pipeline"** (executa processamento)
3. **Observe em tempo real:**
   - ✅ Steps completando na fila à esquerda
   - ✅ Status mudando: pending → running → completed
   - ✅ Dados transformados em cada etapa

---

## 📈 Interpretando os Resultados

### Métricas do Modelo
- **MAE (Mean Absolute Error)**: Erro médio em dólares
  - Se MAE = $30,000 → modelo erra em média $30K para mais ou menos
- **R² Score**: Quanto o modelo explica da variância (0-1)
  - 0.90 = 90% excelente, 0.70 = 70% bom, <0.50 = ruim

### Feature Importance
Mostra quais características mais influenciam o preço:
- **square_feet** (35%): Tamanho é crucial
- **neighborhood** (20%): Localização importa muito
- **condition** (15%): Estado do imóvel
- **bedrooms** (10%): Número de quartos
- **school_rating** (8%): Escolas próximas

### Price Gap (Oportunidades)
- **price_gap > $50K**: 🌟 Excelente oportunidade
- **price_gap $20K-$50K**: ✅ Boa oportunidade
- **price_gap $0-$20K**: ⚠️ Preço justo
- **price_gap < $0**: ❌ Sobrevalorizado (filtrado)

---

## 🎯 Resultado Final

Você terá uma **lista de imóveis subvalorizados** com:
- ✅ Preço real vs. preço previsto pelo modelo
- ✅ Diferença em dólares (quanto está "barato")
- ✅ Dados completos do corretor para contato imediato
- ✅ Link direto para a listagem

**Exemplo de resultado:**
```
property_id: P045
property_type: House
neighborhood: Riverside
bedrooms: 4
sale_price: $320,000
predicted_price: $375,000
price_gap: +$55,000 🎉
agent_name: Sarah Johnson
agent_phone: +1-555-2045
agent_email: sarah.johnson@realty.com
```

**Interpretação:** Casa de 4 quartos em Riverside está $55K abaixo do preço previsto pelo modelo. Contate Sarah Johnson urgentemente!

---

## 💡 Dicas

1. **Durante a execução**, clique nos steps completados para ver os dados transformados
2. **Verifique as métricas** do ML Modeling (R², MAE, Feature Importance)
3. **Ajuste o threshold** do Row Filtering se quiser mais/menos imóveis
4. **Experimente diferentes algoritmos**: Linear Regression (mais simples), Random Forest (mais preciso)
5. **Compare resultados**: Rode a pipeline 2x com algoritmos diferentes e compare as previsões

---

## 🚀 Próximos Passos

Depois de criar esta pipeline, você pode:
- 📊 Criar uma pipeline de **classificação** com customer_churn_dataset.json
- 🔄 Ajustar features (adicionar/remover colunas)
- 🎯 Testar diferentes thresholds de price_gap
- 📈 Comparar Random Forest vs. Linear Regression
- 🏆 Criar uma pipeline para prever **dias no mercado** (days_on_market)

---

**Boa sorte! 🎉 Veja o processo de Machine Learning acontecer ao vivo! 🚀**
