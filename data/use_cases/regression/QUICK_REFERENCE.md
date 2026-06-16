# 🚀 Quick Reference - Regression Pipeline

## 🎯 Objetivo
**Encontrar imóveis subvalorizados** (preço real < preço previsto) para recomendar aos compradores.

---

## 📋 Checklist Rápido

### 1️⃣ Data Ingestion
```
Type: JSON
Path: data/use_cases/regression/property_sales.json
```

### 2️⃣ Column Selection
Manter: property_id, property_type, neighborhood, bedrooms, bathrooms, square_feet, lot_size, age, stories, condition, has_pool, has_fireplace, has_ac, has_basement, school_rating, distance_downtown, crime_rate, sale_price

### 3️⃣ Label Encoding (property_type)
```
Columns: property_type
```

### 4️⃣ Label Encoding (neighborhood)
```
Columns: neighborhood
```

### 5️⃣ Label Encoding (condition)
```
Columns: condition
```

### 6️⃣ Scaling
```
Scaler: StandardScaler
Columns: square_feet, lot_size, age, school_rating, distance_downtown, crime_rate
```

### 7️⃣ ML Modeling - Regression
```
Algorithm: Random Forest Regressor
Target: sale_price
Exclude: property_id
Test Size: 0.2
N Estimators: 100
Max Depth: 15
```

### 8️⃣ Feature Creation
```
Formula Name: price_gap
Formula: predicted_price - sale_price
```

### 9️⃣ Row Filtering
```
Column: price_gap
Operator: >
Value: 20000
```

### 🔟 Data Enrichment
```
Source: data/use_cases/regression/property_details.csv
Join Key: property_id (both)
Join Type: left
```

### 1️⃣1️⃣ Output
```
Type: CSV
Path: data/output/undervalued_properties.csv
```

---

## ✅ Resultado Esperado

**~20-30 imóveis** com preço_gap > $20,000 (subvalorizados)

**Colunas finais:**
- Características: property_type, neighborhood, bedrooms, square_feet, etc.
- Preços: sale_price, predicted_price, price_gap
- Contato: agent_name, agent_phone, agent_email, listing_url

**Métricas do modelo:**
- R² Score: ~0.85-0.92 (bom!)
- MAE: ~$25,000-35,000 (erro médio)

---

## 💡 Interpretação

**price_gap = predicted_price - sale_price**

- `+$50,000` = 🌟 Excelente oportunidade (vale $50K a mais!)
- `+$20,000` = ✅ Boa oportunidade
- `$0` = Preço justo
- `-$20,000` = ❌ Sobrevalorizado (filtrado)

---

Para guia completo: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
