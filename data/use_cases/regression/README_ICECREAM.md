# 🍦 Ice Cream Sales Prediction - Use Case

## 📊 Business Problem

Uma **sorveteria** quer prever quantos sorvetes vai vender para:
- 📦 **Gerenciar estoque** (não faltar nem sobrar)
- 👥 **Escalar equipe** (mais funcionários em dias de alta demanda)
- 💰 **Planejar compras** (ingredientes, embalagens)
- 📈 **Otimizar marketing** (promoções em dias estratégicos)

**Pergunta:** *"Quantos sorvetes venderemos amanhã se a temperatura for 30°C?"*

---

## 🎯 Solução: Machine Learning com Model Persistence

### **Tipo de Problema:** Regression
- **Target (Y):** `ice_creams_sold` (número contínuo)
- **Features (X):** temperatura, dia_semana, feriado, promoção, clima, localização

### **Algoritmo:** Random Forest Regressor
- Robusto a outliers
- Captura relações não-lineares
- Fornece feature importance

---

## 📁 Datasets Gerados

```
data/use_cases/regression/
├── ice_cream_sales_training.json      # 160 dias (80%) - treinar modelo
├── ice_cream_sales_inference.json     # 40 dias (20%) - simular predições
└── ice_cream_sales_complete.json      # 200 dias - validação completa
```

### **Estatísticas:**
- **Período:** 200 dias (1 ano de histórico)
- **Temperatura:** 14-41°C (sazonalidade realista)
- **Vendas:** 10-368 sorvetes/dia
- **Receita Total:** $81,795
- **Correlação Temperatura↔Vendas:** **0.911** ⭐ (correlação muito forte!)

### **Features do Dataset:**

| Feature | Tipo | Descrição | Impacto nas Vendas |
|---------|------|-----------|-------------------|
| `temperature_celsius` | float | Temperatura do dia | 🔥 **ALTO** - Principal fator |
| `is_weekend` | boolean | Final de semana? | 🔼 +20% vendas |
| `is_holiday` | boolean | Feriado? | 🔼 +30% vendas |
| `has_promotion` | boolean | Tem promoção? | 🔼 +25% vendas |
| `weather` | string | Sunny/Cloudy/Rainy | 🌞 Sunny +15%, ☁️ Cloudy -5%, 🌧️ Rainy -40% |
| `store_location` | string | Downtown/Beach | 🏖️ Beach +30% no verão |
| `day_of_week` | int | 0-6 (0=Monday) | Contextual |
| `month` | int | 1-12 | Sazonalidade |

### **Target:**
- `ice_creams_sold` - Número de sorvetes vendidos (10-368)

---

## 🧠 Raciocínio do Modelo Machine Learning

### **Por que Regression?**
Queremos prever um **número contínuo** (quantidade de sorvetes), não uma categoria.

### **Por que Random Forest?**
1. ✅ **Captura relações complexas** - temperatura tem efeito não-linear (muito calor = vendas explodem)
2. ✅ **Múltiplas features** - considera temperatura + dia + feriado + clima simultaneamente
3. ✅ **Robusto** - funciona bem mesmo com alguns dados imperfeitos
4. ✅ **Feature Importance** - mostra quais fatores mais influenciam vendas

### **Relação Temperatura → Vendas:**

```
Temperatura   |  Vendas Esperadas  |  Padrão
15°C          |  ~20 sorvetes      |  Inverno frio
20°C          |  ~60 sorvetes      |  Primavera amena
25°C          |  ~100 sorvetes     |  Dia agradável
30°C          |  ~150 sorvetes     |  Verão quente
35°C+         |  ~250+ sorvetes    |  Verão escaldante! 🔥
```

**+ Multiplicadores:**
- 🎉 Feriado: **×1.3**
- 🌴 Final de semana: **×1.2**
- 💰 Promoção: **×1.25**
- ☀️ Dia ensolarado: **×1.15**
- 🌧️ Dia chuvoso: **×0.6**

**Exemplo real:**
```
Dia: Sábado (weekend=✅)
Feriado: Não
Temperatura: 32°C
Clima: Sunny
Promoção: Sim
Localização: Beach

Cálculo:
Base = (32 - 15) × 8 = 136 sorvetes
× Weekend (1.2) = 163
× Sunny (1.15) = 187
× Promotion (1.25) = 234
× Beach verão (1.3) = 304 sorvetes! 🍦
```

---

## 🚀 Workflow MLOps

### **PIPELINE 1: Training (Treinar e Salvar Modelo)**

#### **Objetivo:** 
Treinar modelo com dados históricos e salvar para reutilização.

#### **Steps:**

```json
{
  "name": "Ice Cream Sales - Training",
  "steps": [
    {
      "name": "1. Load Historical Data",
      "type": "ingestion",
      "config": {
        "ingestion_type": "file_upload",
        "file_path": "../data/use_cases/regression/ice_cream_sales_training.json",
        "file_format": "json"
      }
    },
    {
      "name": "2. Clean Data",
      "type": "cleaning",
      "config": {
        "remove_nulls": true,
        "remove_duplicates": true
      }
    },
    {
      "name": "3. Train Model",
      "type": "model_training",
      "config": {
        "model_name": "ice_cream_sales_predictor",
        "model_type": "regression",
        "algorithm": "random_forest",
        "target_column": "ice_creams_sold",
        "exclude_features": "date,day_name,revenue_usd",
        "test_size": 0.2,
        "random_state": 42,
        "hyperparameters": {
          "n_estimators": 200,
          "max_depth": 15
        },
        "auto_version": true,
        "save_predictions": true
      }
    }
  ]
}
```

#### **Resultado Esperado:**
- ✅ Modelo salvo: `backend/data/models/ice_cream_sales_predictor.pkl`
- ✅ Metadata: `ice_cream_sales_predictor_metadata.json`
- ✅ **R² Score: 0.85-0.92** (excelente! 🎉)
- ✅ **MAE: 10-15 sorvetes** (erro aceitável)
- ✅ Feature Importance: `temperature_celsius` será #1

---

### **PIPELINE 2: Inference (Prever Novos Dias)**

#### **Objetivo:** 
Carregar modelo treinado e prever vendas para próximos dias.

#### **Steps:**

```json
{
  "name": "Ice Cream Sales - Inference",
  "steps": [
    {
      "name": "1. Load Weather Forecast",
      "type": "ingestion",
      "config": {
        "ingestion_type": "file_upload",
        "file_path": "../data/use_cases/regression/ice_cream_sales_inference.json",
        "file_format": "json"
      }
    },
    {
      "name": "2. Clean Data",
      "type": "cleaning",
      "config": {
        "remove_nulls": true,
        "remove_duplicates": true
      }
    },
    {
      "name": "3. Load Trained Model",
      "type": "model_loading",
      "config": {
        "model_name": "ice_cream_sales_predictor",
        "validate_features": true,
        "cache_model": true
      }
    },
    {
      "name": "4. Predict Sales",
      "type": "model_inference",
      "config": {
        "prediction_column": "predicted_ice_creams",
        "include_probabilities": false,
        "include_error": true,
        "compare_with_actual": false
      }
    }
  ]
}
```

#### **Output:**
Para cada dia futuro:
- `predicted_ice_creams` - Quantos sorvetes venderemos
- `prediction_error` - Margem de erro
- **Exemplo:** "Amanhã (32°C, Sábado, Sunny) → **234 sorvetes**"

---

## 📊 Métricas de Sucesso

### **Modelo Bom:**
- ✅ **R² > 0.80** - Explica >80% da variação nas vendas
- ✅ **MAE < 20** - Erro médio menor que 20 sorvetes
- ✅ **Feature importance:** temperatura deve ser top 1

### **Modelo Excelente:**
- 🌟 **R² > 0.90** - Explica >90% da variação
- 🌟 **MAE < 15** - Erro muito baixo
- 🌟 **Predictions úteis:** Gerente consegue planejar estoque com confiança

---

## 💡 Insights de Negócio

Depois de treinar, você pode responder:

### **1. Planejamento de Estoque:**
```
Previsão: 5 dias
- Segunda (22°C, cloudy) → 65 sorvetes
- Terça (25°C, sunny) → 110 sorvetes
- Quarta (28°C, sunny) → 140 sorvetes
- Quinta (30°C, sunny, promoção) → 190 sorvetes ⚠️
- Sexta (32°C, sunny, weekend) → 220 sorvetes ⚠️

Ação: Comprar ingredientes para 750 sorvetes!
```

### **2. Otimização de Promoções:**
```
Pergunta: "Vale a pena fazer promoção quinta-feira (30°C)?"
Resposta do modelo:
- SEM promoção: 150 sorvetes × $5 = $750
- COM promoção: 190 sorvetes × $4 = $760
💡 SIM! Promoção aumenta lucro.
```

### **3. Gestão de Equipe:**
```
Se previsão > 150 sorvetes → Chamar funcionário extra
Se previsão < 50 sorvetes → Apenas 1 funcionário
```

---

## 🧪 Como Testar no FlowScope

### **Passo 1: Training**
1. Acesse FlowScope: http://localhost:5173
2. Create New Pipeline
3. Nome: "Ice Cream Sales - Training"
4. Adicione os 3 steps (Ingestion, Cleaning, Model Training)
5. Configure conforme JSON acima
6. Execute!
7. Verifique: `backend/data/models/ice_cream_sales_predictor.pkl` foi criado

### **Passo 2: Inference**
1. Create New Pipeline
2. Nome: "Ice Cream Sales - Inference"
3. Adicione os 4 steps (Ingestion, Cleaning, Loading, Inference)
4. Configure conforme JSON acima
5. Execute!
6. Veja predições na aba "Data"

### **Passo 3: Validar**
Compare `predicted_ice_creams` com vendas reais usando o dataset completo.

---

## 🎓 Conceitos ML Aprendidos

### **1. Regression vs Classification**
- ❌ Classification: "Vai chover sim/não?" (categoria)
- ✅ Regression: "Quantos mm de chuva?" (número)

### **2. Train/Test Split**
- 80% treino (160 dias) - ensinar o modelo
- 20% teste (40 dias) - avaliar se aprendeu

### **3. Feature Engineering**
- Transformar dados brutos em features úteis
- Ex: `date` → `day_of_week`, `is_weekend`, `month`

### **4. Model Persistence**
- Treinar UMA vez → Usar INFINITAS vezes
- Economiza tempo e recursos
- Pattern de produção!

### **5. Feature Importance**
- Quais fatores mais influenciam?
- Resposta: temperatura (0.75), weekend (0.10), weather (0.08)...

---

## 🚀 Próximos Passos

Depois de dominar este use case:

1. **Adicionar mais features:**
   - Vendas do dia anterior
   - Eventos especiais na cidade
   - Preço da gasolina (afeta passeios)

2. **Experimentar outros algoritmos:**
   - XGBoost (pode ser melhor!)
   - Linear Regression (baseline simples)

3. **Hyperparameter Tuning:**
   - Encontrar o melhor `n_estimators`
   - Otimizar `max_depth`

4. **Deploy em produção:**
   - API REST para consultar previsões
   - Dashboard para visualizar trends

---

## 📚 Referências

- **Dataset:** Gerado sinteticamente com correlação realista
- **Modelo:** scikit-learn RandomForestRegressor
- **Padrão:** MLOps com Model Persistence
- **Inspiração:** Problemas reais de varejo

---

**🎉 Agora você tem um use case completo de Regression com Model Persistence!**

Teste, experimente, e entenda como ML resolve problemas reais de negócio! 🍦📈
