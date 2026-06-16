# 🛍️ Customer Segmentation - E-commerce Clustering

## 🎯 **Objetivo**

Identificar **grupos de clientes com comportamentos similares** para criar estratégias de marketing personalizadas e aumentar o ROI (Retorno sobre Investimento) das campanhas.

## 📖 **Contexto do Negócio**

Uma empresa de e-commerce possui **milhares de clientes** com diferentes perfis de compra, engajamento e valor. A equipe de marketing precisa:

- 📧 **Personalizar campanhas de email** para cada perfil
- 💰 **Otimizar orçamento** focando em clientes de alto valor
- 🎁 **Criar ofertas específicas** para reativar clientes inativos
- ⭐ **Identificar clientes VIP** para programas de fidelidade
- 🚨 **Detectar clientes em risco** de churn

## 🔍 **O Que Estamos Clusterizando?**

Vamos agrupar clientes baseados em:

### **Dimensões RFM** (análise clássica de marketing)
- **R**ecency: Quanto tempo desde a última compra?
- **F**requency: Quantas vezes comprou?
- **M**onetary: Quanto gastou no total?

### **Dimensões Comportamentais**
- Engajamento com emails (clicks, opens)
- Categoria favorita de produtos
- Taxa de retorno de produtos
- Tickets de suporte abertos

### **Dimensões Demográficas**
- Idade, gênero, país
- Status de cliente premium

## 📊 **Dataset: `customer_segmentation.json`**

**200 registros** de clientes com **19 features**.

### **Features Disponíveis**

| Feature | Descrição | Tipo | Problemas Esperados |
|---------|-----------|------|---------------------|
| `customer_id` | ID único do cliente | String | ⚠️ Duplicados |
| `name` | Nome do cliente | String | ❌ Nulls, vazios, "UNKNOWN" |
| `email` | Email do cliente | String | ❌ Malformados, nulls |
| `age` | Idade do cliente | Integer | ❌ Negativos, > 120, nulls |
| `gender` | Gênero | String | ❌ Inconsistências (M/Male/male) |
| `country` | País | String | ❌ Nulls, "???" |
| `registration_date` | Data de cadastro | Date | ❌ Datas futuras, inválidas |
| `last_purchase_date` | Data da última compra | Date | ❌ Nulls |
| `total_purchases` | Total de compras | Integer | ❌ Negativos!, absurdos |
| `total_spend` | Valor total gasto | Float | ❌ Negativos!, inconsistente |
| `avg_order_value` | Ticket médio | Float | ❌ Inconsistente com total |
| `days_since_last_purchase` | Dias desde última compra | Integer | ❌ Negativos |
| `favorite_category` | Categoria favorita | String | ❌ Nulls, "N/A" |
| `email_clicks` | Clicks em emails | Integer | ❌ Negativos |
| `email_opens` | Aberturas de email | Integer | ❌ Negativos |
| `support_tickets` | Tickets de suporte | Integer | ❌ Negativos |
| `returns_count` | Número de devoluções | Integer | ❌ Maior que total_purchases! |
| `has_premium` | Cliente premium? | Boolean | ❌ Múltiplos formatos (True/Yes/1) |
| `referral_count` | Indicações feitas | Integer | ❌ Negativos |

### **Estatísticas dos Problemas**

- 🔴 **42 clientes** com nome nulo/vazio/inválido
- 🔴 **58 clientes** com idade inválida (< 0 ou > 120)
- 🔴 **61 clientes** com valores negativos em métricas
- 🔴 **15 IDs duplicados**
- 🔴 ~20% de valores null distribuídos

## 🎯 **Resultados Esperados do Clustering**

Após a pipeline, esperamos identificar **4-5 grupos distintos**:

### 🏆 **Cluster 1: VIP Champions**
- Alta frequência, alto valor, comprou recentemente
- **Ação**: Programa VIP, acesso antecipado, descontos exclusivos

### 💎 **Cluster 2: High Value at Risk**
- Alto valor histórico, mas não compra há muito tempo
- **Ação**: Campanhas de reativação urgentes, cupons agressivos

### 🌱 **Cluster 3: Growing Customers**
- Frequência média, valor crescente, engajados
- **Ação**: Incentivar próxima compra, cross-sell/up-sell

### 😴 **Cluster 4: Dormant / Low Engagement**
- Poucas compras, baixo valor, não abre emails
- **Ação**: Campanhas de reengajamento, pesquisa de satisfação

### 🆕 **Cluster 5: New Customers**
- Recém cadastrados, poucas compras
- **Ação**: Onboarding, educação sobre produtos, primeira recompra

## 📈 **Métricas de Sucesso**

Após implementar as ações baseadas nos clusters:

- 📊 **+25% em taxa de conversão** de campanhas segmentadas
- 📧 **+40% em open rate** de emails personalizados
- 💰 **+15% em LTV** (Lifetime Value) dos clientes
- 🔄 **-30% em taxa de churn** dos clientes de alto valor

## 🛠️ **Pipeline Steps Necessários**

Ver arquivo: `ROTEIRO_PIPELINE.md`

## 📚 **Conceitos Aplicados**

- **K-Means Clustering**: Algoritmo de agrupamento
- **RFM Analysis**: Framework clássico de segmentação
- **Feature Engineering**: Criação de features relevantes
- **Data Cleaning**: Limpeza de dados sujos
- **Normalization**: Normalização para K-Means funcionar bem
- **Elbow Method**: Determinar número ideal de clusters

## 🎓 **Aprendizados**

Este case demonstra:
1. Como dados reais são **bagunçados**
2. Importância da **limpeza de dados** (80% do trabalho!)
3. Como **feature engineering** impacta resultados
4. Valor de negócio de **unsupervised learning**
5. Como traduzir clusters em **ações de marketing**

---

**💡 Dica**: Este é um dataset sintético para aprendizado. Em produção, você teria **milhões** de registros e precisaria de ferramentas como Spark/Databricks para processar!
