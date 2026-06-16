# 🚀 Space Explorer Chatbot - Guia de Uso

## 📋 Visão Geral

Você treinou com sucesso o chatbot **Space Explorer** que pode:
- ✅ Classificar intenções de perguntas sobre o espaço
- ✅ Buscar conhecimento relevante em uma base de 50 documentos
- ✅ Gerar respostas contextualizadas

## 🎯 Como Usar o Chatbot

### Opção 1: Pipeline de Conversação (Recomendado)

1. **Vá para a página de Pipelines** (Menu lateral → Pipelines)
2. **Clique em "Criar Pipeline"**
3. **Selecione o template**: "Space Explorer - Conversação"
4. **Execute o pipeline** e digite sua pergunta quando solicitado

#### Exemplos de Perguntas:

**Cumprimentos:**
- "Olá!"
- "Bom dia"

**Perguntas sobre Planetas:**
- "O que é Marte?"
- "Qual é o maior planeta do Sistema Solar?"
- "Júpiter tem anéis?"

**Perguntas sobre Estrelas:**
- "O que é uma estrela?"
- "Como o Sol funciona?"
- "O que acontece quando uma estrela morre?"

**Perguntas sobre Galáxias:**
- "O que é a Via Láctea?"
- "Quantas galáxias existem?"
- "O que é uma galáxia espiral?"

**Perguntas sobre Buracos Negros:**
- "O que é um buraco negro?"
- "Buracos negros são perigosos?"
- "Como se forma um buraco negro?"

**Perguntas sobre Exploração Espacial:**
- "Quem foi o primeiro homem no espaço?"
- "Quando chegamos à Lua?"
- "O que é a Estação Espacial Internacional?"

**Despedidas:**
- "Obrigado!"
- "Até logo"

---

## 🔧 Opção 2: Teste Individual de Componentes

### Testar Classificação de Intenção

1. Crie um CSV com perguntas:
```csv
text
O que é um buraco negro?
Olá, tudo bem?
Qual é o maior planeta?
```

2. Crie pipeline com passos:
   - **Carregar Dados** (file_upload)
   - **Classificar Intenção** (intent_classification, model: `space_explorer`)

### Testar Busca Semântica

1. Crie pipeline com passos:
   - **Entrada do Usuário** (user_input)
   - **Buscar Conhecimento** (semantic_search, kb: `space_explorer_kb`, top_k: 5)

---

## 📊 Detalhes Técnicos

### Modelo Treinado: `space_explorer`
- **Tipo**: Classificação de Intenção
- **Algoritmo**: Naive Bayes + TF-IDF
- **Classes**: 9 intenções
  - `greeting` - Cumprimentos
  - `question_planet` - Perguntas sobre planetas
  - `question_star` - Perguntas sobre estrelas
  - `question_galaxy` - Perguntas sobre galáxias
  - `question_blackhole` - Perguntas sobre buracos negros
  - `question_space_exploration` - Exploração espacial
  - `question_general` - Perguntas gerais
  - `thanks` - Agradecimentos
  - `goodbye` - Despedidas

### Base de Conhecimento: `space_explorer_kb`
- **Documentos**: 50 textos educacionais
- **Tópicos**: Sistema Solar, planetas, estrelas, galáxias, buracos negros, exploração espacial
- **Tecnologia**: Embeddings semânticos (Sentence Transformers)

---

## 🎨 Personalizando Respostas

Você pode editar o pipeline para ajustar:

1. **Número de documentos retornados** (`top_k` no semantic_search)
2. **Limite de similaridade** (`min_similarity` - quanto menor, mais resultados)
3. **Estilo de resposta** (`response_style` no response_generation)

---

## 🐛 Solução de Problemas

**Problema**: Pipeline não encontra o modelo
- **Solução**: Verifique se o modelo `space_explorer` aparece na página de Modelos

**Problema**: Busca não retorna resultados
- **Solução**: Reduza o `min_similarity` para 0.2 ou menos

**Problema**: Intenção classificada incorretamente
- **Solução**: Adicione mais exemplos de treinamento no CSV e retreine o modelo

---

## 📚 Próximos Passos

1. **Testar o Chatbot** - Use o pipeline de conversação
2. **Avaliar Respostas** - Verifique se as respostas fazem sentido
3. **Melhorar o Modelo** - Adicione mais exemplos de treinamento se necessário
4. **Expandir Conhecimento** - Adicione mais documentos na base de conhecimento
5. **Integrar API** - Use o chatbot via API REST (endpoint `/chat`)

---

**✨ Divirta-se explorando o universo com seu chatbot espacial! 🌌**
