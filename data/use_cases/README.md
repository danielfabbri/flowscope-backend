# FlowScope AI - Datasets

Datasets organizados por caso de uso de Machine Learning.

## Estrutura

```
data/use_cases/
├── classification/          # Datasets para problemas de classificação
│   ├── customer_churn_dataset.json
│   └── customer_contacts.csv
│
└── regression/             # Datasets para problemas de regressão
    ├── property_sales.json
    └── property_details.csv
```

---

## Classification: Customer Churn Prediction

### customer_churn_dataset.json
**Objetivo:** Prever se um cliente vai cancelar o serviço (churned = 0 ou 1)

**100 registros** com 20 colunas:
- `customer_id`: ID único do cliente (C001-C100)
- `age`: Idade do cliente
- `gender`: Gênero (M/F)
- `region`: Região geográfica
- `acquisition_channel`: Canal de aquisição
- `days_as_customer`: Dias como cliente
- `loyalty_tier`: Nível de fidelidade (Bronze/Silver/Gold/Platinum)
- `total_purchases`: Total de compras realizadas
- `total_spent`: Valor total gasto
- `avg_order_value`: Valor médio por pedido
- `days_since_last_purchase`: Dias desde última compra
- `favorite_category`: Categoria favorita
- `has_loyalty_card`: Possui cartão fidelidade (true/false)
- `has_app`: Possui app instalado (true/false)
- `accepts_marketing`: Aceita marketing (true/false)
- `num_complaints`: Número de reclamações
- `satisfaction_score`: Score de satisfação (1-5)
- `support_contacts`: Contatos com suporte
- `returns_rate`: Taxa de devoluções (0-1)
- **`churned`**: TARGET - Cliente cancelou? (0 = Não, 1 = Sim)

**Distribuição:**
- 58% churned (classe 1)
- 42% ativos (classe 0)

### customer_contacts.csv
**Objetivo:** Dataset auxiliar para enrichment (JOIN)

**100 registros** com 3 colunas:
- `customer_id`: ID do cliente (chave para JOIN)
- `phone`: Telefone de contato
- `email`: E-mail do cliente

---

## Regression: Property Price Prediction

### property_sales.json
**Objetivo:** Prever o preço de venda de imóveis (sale_price)

**100 registros** com 22 colunas:
- `property_id`: ID único do imóvel (P001-P100)
- `property_type`: Tipo (House/Apartment/Condo/Townhouse)
- `neighborhood`: Bairro
- `bedrooms`: Número de quartos
- `bathrooms`: Número de banheiros
- `square_feet`: Área em pés quadrados
- `lot_size`: Tamanho do terreno
- `year_built`: Ano de construção
- `age`: Idade do imóvel (anos)
- `stories`: Número de andares
- `condition`: Condição (Excellent/Good/Fair/Poor)
- `heating_type`: Tipo de aquecimento
- `parking_type`: Tipo de estacionamento
- `has_pool`: Possui piscina (true/false)
- `has_fireplace`: Possui lareira (true/false)
- `has_ac`: Possui ar condicionado (true/false)
- `has_basement`: Possui porão (true/false)
- `school_rating`: Avaliação das escolas próximas (1-10)
- `distance_downtown`: Distância do centro (milhas)
- `crime_rate`: Taxa de criminalidade (por 1000 pessoas)
- `days_on_market`: Dias no mercado
- **`sale_price`**: TARGET - Preço de venda (USD)

**Estatísticas:**
- Preço mínimo: ~$110,000
- Preço máximo: ~$990,000
- Preço médio: ~$379,000

**Distribuição de tipos:**
- House: 21%
- Apartment: 25%
- Condo: 35%
- Townhouse: 19%

### property_details.csv
**Objetivo:** Dataset auxiliar para enrichment (JOIN)

**100 registros** com 5 colunas:
- `property_id`: ID do imóvel (chave para JOIN)
- `agent_name`: Nome do corretor
- `agent_phone`: Telefone do corretor
- `agent_email`: E-mail do corretor
- `listing_url`: URL da listagem

---

## Uso nos Pipelines

### Pipeline de Classificação
1. **Data Ingestion**: Carregar `customer_churn_dataset.json`
2. **Feature Engineering**: Normalizar, encodar variáveis categóricas
3. **ML Modeling**: Treinar modelo de classificação (Random Forest, Logistic Regression)
4. **Row Filtering**: Filtrar clientes com alta probabilidade de churn
5. **Data Enrichment**: JOIN com `customer_contacts.csv` para adicionar telefone/email
6. **Output**: Exportar lista de clientes em risco com contatos

### Pipeline de Regressão
1. **Data Ingestion**: Carregar `property_sales.json`
2. **Feature Engineering**: Normalizar valores numéricos, encodar property_type
3. **ML Modeling**: Treinar modelo de regressão (Random Forest, Linear Regression)
4. **Row Filtering**: Filtrar imóveis por faixa de preço prevista
5. **Data Enrichment**: JOIN com `property_details.csv` para adicionar informações do corretor
6. **Output**: Exportar previsões com detalhes dos corretores

---

## Regenerar Dados

Para gerar novos datasets com valores diferentes:

```bash
# Classification
python generate_extended_data.py

# Regression
python generate_regression_data.py
```
