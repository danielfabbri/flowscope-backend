# 🗺️ Roteiro da Pipeline: Customer Segmentation

## 📋 **Visão Geral**

Este roteiro guia você **passo a passo** na criação de uma pipeline completa de clusterização de clientes, desde a ingestão dos dados sujos até a identificação de segmentos acionáveis.

---

## 🎯 **Pipeline Steps**

### **Step 1: Data Ingestion** 📥
**Objetivo**: Carregar o dataset de clientes

**Configuração**:
```
Nome: Data Ingestion
Tipo: ingestion
Ingestion Type: file_upload
File Path: C:\dev\flowscope\data\use_cases\clustering\customer_segmentation.json
File Format: json
```

**O que acontece**: 200 clientes carregados com todos os problemas de dados

**Verificar**: Você deve ver 19 colunas e 200 linhas

---

### **Step 2: Data Profiling** 🔍
**Objetivo**: Entender a qualidade dos dados

**Configuração**:
```
Nome: Data Profiling
Tipo: profiling
```

**O que acontece**: Geração de estatísticas de:
- Valores nulos por coluna
- Tipos de dados
- Distribuições
- Outliers

**O que procurar**:
- ⚠️ 42 nomes inválidos
- ⚠️ 58 idades inválidas
- ⚠️ 61 valores negativos em métricas
- ⚠️ 15 IDs duplicados

---

### **Step 3: Data Cleaning - Remover Duplicados** 🧹
**Objetivo**: Eliminar registros duplicados

**Configuração**:
```
Nome: Remove Duplicates
Tipo: cleaning
Clean Method: drop_duplicates
Subset Columns: customer_id
Keep: first
```

**Resultado esperado**: ~185 linhas (15 duplicados removidos)

---

### **Step 4: Data Cleaning - Valores Nulos** 💧
**Objetivo**: Tratar valores nulos

**Configuração**:
```
Nome: Handle Missing Values
Tipo: cleaning
Handle Missing: fill
Fill Strategy: median (para numéricas)
Drop Threshold: 0.8
```

**Estratégia**:
- Numéricas: preencher com mediana
- Categóricas: preencher com "Unknown" ou moda
- Se > 80% null: remover coluna

**Resultado esperado**: ~180-185 linhas completas

---

### **Step 5: Column Selection** 🎯
**Objetivo**: Selecionar apenas features úteis para clustering

**Configuração**:
```
Nome: Select Features for Clustering
Tipo: column_selection
Action: keep
Columns to Keep:
  - customer_id (para identificação)
  - age
  - total_purchases
  - total_spend
  - avg_order_value
  - days_since_last_purchase
  - email_clicks
  - email_opens
  - support_tickets
  - returns_count
  - referral_count
```

**Por que remover outras?**:
- `name`, `email`: não são numéricas
- `country`, `gender`, `favorite_category`: precisam de encoding
- `registration_date`, `last_purchase_date`: já temos `days_since_last_purchase`
- `has_premium`: inconsistente demais

**Resultado esperado**: 11 colunas (1 ID + 10 features)

---

### **Step 6: Data Cleaning - Valores Negativos** ⚠️
**Objetivo**: Corrigir valores impossíveis (negativos)

**Configuração**:
```
Nome: Fix Negative Values
Tipo: cleaning
Handle Missing: fill
Custom Rules:
  - age < 0 → null (remover depois)
  - age > 120 → null
  - total_purchases < 0 → 0
  - total_spend < 0 → 0
  - email_clicks < 0 → 0
  - email_opens < 0 → 0
  - support_tickets < 0 → 0
  - returns_count < 0 → 0
  - referral_count < 0 → 0
```

**IMPORTANTE**: O FlowScope pode não ter regras customizadas. **Alternativa**:
1. Use **Row Filtering** para remover linhas com valores negativos:
   - Filter Column: `age`
   - Operator: `>=`
   - Value: `0`
   - Keep Matching: `true`

2. Repita para cada coluna problemática

**Resultado esperado**: Dados limpos, sem negativos

---

### **Step 7: Feature Engineering - Criar RFM Score** 🎨
**Objetivo**: Criar features derivadas mais significativas

**Configuração**:
```
Nome: Create RFM Features
Tipo: feature_engineering
Transformation Type: feature_creation

Feature 1 - Recency Score:
  - Feature Name: recency_score
  - Formula: 365 - days_since_last_purchase
  (quanto maior, melhor - comprou recentemente)

Feature 2 - Engagement Score:
  - Feature Name: engagement_score
  - Formula: email_clicks + email_opens
  (soma total de interações)

Feature 3 - Return Rate:
  - Feature Name: return_rate
  - Formula: returns_count / total_purchases
  (% de devoluções)
```

**NOTA**: Se o FlowScope não suportar múltiplas features, crie **3 steps separados** de Feature Engineering.

**Resultado esperado**: 14 colunas (11 anteriores + 3 novas)

---

### **Step 8: Data Cleaning - Remove Outliers Extremos** 🔥
**Objetivo**: Remover outliers que quebram o modelo

**Configuração**:
```
Nome: Remove Extreme Outliers
Tipo: cleaning
Outlier Method: IQR ou Z-Score
Threshold: 3 (desvios padrão)
Columns: total_spend, total_purchases
```

**OU use Row Filtering**:
```
Exemplo: Remover total_spend > 1,000,000
- Filter Column: total_spend
- Operator: <
- Value: 1000000
- Keep Matching: true
```

**Resultado esperado**: ~170-180 linhas (outliers absurdos removidos)

---

### **Step 9: Feature Engineering - Normalização** 📏
**Objetivo**: Colocar todas as features na mesma escala (0-1)

**Configuração**:
```
Nome: Normalize Features
Tipo: feature_engineering
Transformation Type: scaling
Scaling Method: standard ou minmax
Columns to Scale:
  - age
  - total_purchases
  - total_spend
  - avg_order_value
  - days_since_last_purchase
  - email_clicks
  - email_opens
  - support_tickets
  - returns_count
  - referral_count
  - recency_score
  - engagement_score
  - return_rate
```

**POR QUE?** K-Means é sensível a escala! Se `total_spend` vai de 0 a 100,000 e `age` de 0 a 80, o modelo vai ignorar `age`.

**Resultado esperado**: Todas as colunas numéricas entre -3 e 3 (standard) ou 0 e 1 (minmax)

---

### **Step 10: ML Modeling - Clustering** 🤖
**Objetivo**: Criar os clusters de clientes!

**Configuração**:
```
Nome: K-Means Clustering
Tipo: ml_modeling
Model Type: clustering
Algorithm: kmeans
Number of Clusters: 4
Exclude Features: customer_id
Random State: 42
```

**Como escolher número de clusters?**
- Comece com 4-5
- Teste com 3, 4, 5, 6
- Veja qual faz mais sentido de negócio

**Resultado esperado**: 
- Nova coluna `cluster` com valores 0, 1, 2, 3
- Cada cliente atribuído a um cluster

---

### **Step 11: Sorting - Ver Clusters Ordenados** 📊
**Objetivo**: Ordenar para análise

**Configuração**:
```
Nome: Sort by Cluster
Tipo: sorting
Sort Column: cluster
Sort Order: ascending
Use Absolute: false
```

**Resultado esperado**: Todos os clientes do Cluster 0 juntos, depois Cluster 1, etc.

---

### **Step 12: Output - Salvar Resultado** 💾
**Objetivo**: Preparar dados para exportação

**Configuração**:
```
Nome: Final Output
Tipo: output
Output Format: csv ou parquet
Include Metadata: true
```

**Resultado esperado**: Dataset final com os clusters prontos para ação!

---

## 📊 **Analisando os Resultados**

Após a pipeline, você terá **4 clusters**. Para entender cada um:

### **Análise Manual**

1. **Filtre por cada cluster** (use Row Filtering)
2. **Calcule médias** das métricas por cluster:
   ```
   Cluster 0:
   - Avg total_purchases: ?
   - Avg total_spend: ?
   - Avg days_since_last_purchase: ?
   ```

3. **Dê nomes aos clusters**:
   - Cluster 0: "VIP Champions" (alta frequência, alto valor)
   - Cluster 1: "At Risk" (alto valor histórico, não compra há tempo)
   - Cluster 2: "Growing" (frequência média crescente)
   - Cluster 3: "Dormant" (baixo engajamento)

### **Métricas de Qualidade**

- **Silhouette Score**: Quão bem definidos são os clusters? (0.5+ é bom)
- **Inertia**: Soma das distâncias ao centroide (menor = melhor)
- **Cluster Size**: Clusters muito desbalanceados? (ex: 1 com 170, outros com 10)

---

## 🎯 **Próximos Passos**

1. **Exportar dados** com clusters
2. **Criar dashboard** visual dos clusters
3. **Definir ações** de marketing para cada cluster
4. **Implementar campanhas** segmentadas
5. **Medir resultados** (conversão, ROI, etc.)

---

## 💡 **Dicas Importantes**

### **Se algo der errado**:

❌ **"Clustering não rodou"**
→ Verifique se todas as colunas são numéricas
→ Remova `customer_id` das features
→ Certifique-se de normalizar os dados

❌ **"Todos os clientes no mesmo cluster"**
→ Dados não normalizados
→ Poucas features (adicione mais)
→ Tente número diferente de clusters

❌ **"Muitos valores null ainda"**
→ Volte ao Step 4 e seja mais agressivo
→ Considere remover colunas com > 50% null

❌ **"Pipeline muito lenta"**
→ Normal com 200 registros, deve levar 1-2 min
→ Verifique se backend está rodando

---

## 📝 **Checklist Final**

- [ ] Dataset carregado (200 clientes)
- [ ] Duplicados removidos (~15 linhas)
- [ ] Nulos tratados
- [ ] Valores negativos corrigidos
- [ ] Features selecionadas (10-13 numéricas)
- [ ] Outliers extremos removidos
- [ ] Dados normalizados (escala 0-1 ou -3 a 3)
- [ ] K-Means aplicado (4-5 clusters)
- [ ] Cluster column criada
- [ ] Resultados analisados

---

## 🎓 **Conceitos Aprendidos**

✅ Data Profiling
✅ Data Cleaning (duplicados, nulls, outliers)
✅ Feature Engineering (criação, normalização)
✅ Unsupervised Learning (K-Means)
✅ RFM Analysis
✅ Business Translation (clusters → ações)

**Parabéns! Você completou um projeto end-to-end de Data Science!** 🎉
