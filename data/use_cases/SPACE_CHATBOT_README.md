# 🌌 Space Explorer Chatbot - Dataset de Treinamento

## 📚 Visão Geral

Este é um dataset educacional para treinar um chatbot especializado em **conceitos do espaço sideral e astronomia**. Perfeito para criar um assistente de ensino sobre o universo!

---

## 📂 Arquivos do Dataset

### 1. `space_chatbot_intent_training.csv`
**68 exemplos de treinamento** com as seguintes intenções:

| Intenção | Descrição | Exemplos |
|----------|-----------|----------|
| `greeting` | Saudações | 5 exemplos |
| `question_planet` | Perguntas sobre planetas | 13 exemplos |
| `question_star` | Perguntas sobre estrelas | 10 exemplos |
| `question_galaxy` | Perguntas sobre galáxias | 7 exemplos |
| `question_blackhole` | Perguntas sobre buracos negros | 6 exemplos |
| `question_space_exploration` | Perguntas sobre exploração espacial | 8 exemplos |
| `question_general` | Perguntas gerais sobre espaço | 11 exemplos |
| `thanks` | Agradecimentos | 4 exemplos |
| `goodbye` | Despedidas | 4 exemplos |

### 2. `space_chatbot_knowledge_base.csv`
**50 documentos informativos** cobrindo:

| Categoria | Tópicos | Documentos |
|-----------|---------|-----------|
| **Planetas** | Sistema Solar, Mercúrio, Vênus, Terra, Marte, Júpiter, Saturno, Urano, Netuno, Plutão | 10 docs |
| **Estrelas** | Formação, ciclo de vida, Sol, supernovas, características | 9 docs |
| **Galáxias** | Via Láctea, formação, tipos, futuro | 5 docs |
| **Buracos Negros** | Formação, efeitos, mitos, evaporação | 5 docs |
| **Exploração Espacial** | História, missões, ISS, futuro | 7 docs |
| **Universo** | Big Bang, expansão, tamanho, forma | 4 docs |
| **Outros** | Asteroides, cometas, eclipses, matéria escura, constelações | 10 docs |

---

## 🚀 Como Usar Este Dataset

### Passo 1: Criar Pipeline de Treinamento

1. No FlowScope AI, clique em **Create New Pipeline**
2. Escolha o template **"Chatbot Inteligente - Treinamento"**
3. Nomeie o pipeline: `Space Explorer - Training`

### Passo 2: Configurar Steps de Treinamento

#### Step 1: Carregar Intenções de Treinamento
```
File Path: data/use_cases/space_chatbot_intent_training.csv
```

#### Step 2: Treinar Classificador de Intenção
```
Text Column: text
Intent Column: intent
Model Name: space_explorer_model  ⭐ IMPORTANTE: Lembre este nome!
Test Size: 0.2
```

#### Step 3: Carregar Base de Conhecimento
```
File Path: data/use_cases/space_chatbot_knowledge_base.csv
```

#### Step 4: Indexar Conhecimento
```
Text Column: text
Index Name: space_explorer_kb  ⭐ IMPORTANTE: Lembre este nome!
ID Column: id
Metadata Columns: category, topic
```

### Passo 3: Executar Treinamento

1. Clique em **Create Pipeline**
2. Execute o pipeline (botão ▶️ Play)
3. Aguarde o treinamento completar (aprox. 30-60 segundos)
4. Verifique as métricas de acurácia no resultado

**Métricas esperadas:**
- Acurácia: ~85-95% (depende da divisão train/test)
- 9 classes de intenção balanceadas
- 50 documentos indexados com sucesso

---

## 💬 Como Usar o Chatbot Treinado

### Passo 1: Criar Pipeline de Uso

1. Clique em **Create New Pipeline**
2. Escolha o template **"Chatbot Inteligente - Uso"**
3. Nomeie: `Space Explorer - Chat`

### Passo 2: Configurar Steps de Uso

#### Step 2: Classificar Intenção
```
Model Name: space_explorer_model  (mesmo nome do treinamento)
```

#### Step 4: Busca Semântica
```
Index Name: space_explorer_kb  (mesmo nome do treinamento)
```

#### Step 6: Gerar Resposta

Configure templates personalizados (opcional):
```json
{
  "greeting": "🌟 Olá, explorador espacial! Estou aqui para ensinar sobre o universo. O que você gostaria de saber?",
  "question_planet": "🪐 Sobre planetas: {context}. Ficou claro?",
  "question_star": "⭐ Sobre estrelas: {context}",
  "question_galaxy": "🌌 Sobre galáxias: {context}",
  "question_blackhole": "⚫ Sobre buracos negros: {context}",
  "question_space_exploration": "🚀 Sobre exploração espacial: {context}",
  "question_general": "🌠 {context}",
  "thanks": "🙌 Que bom que gostou! Continuar aprendendo sobre o universo? 🔭",
  "goodbye": "👋 Até logo, futuro astronauta! Continue explorando o cosmos!"
}
```

### Passo 3: Testar o Chatbot

Execute o pipeline com perguntas como:

| Pergunta | Intenção Esperada | Resposta Esperada |
|----------|------------------|-------------------|
| "Olá!" | greeting | Saudação amigável |
| "O que são planetas?" | question_planet | Explicação sobre planetas |
| "Como o Sol vai morrer?" | question_star | Explicação sobre ciclo de vida estelar |
| "O que é um buraco negro?" | question_blackhole | Explicação sobre buracos negros |
| "Existe vida em Marte?" | question_space_exploration | Info sobre exploração de Marte |
| "O que é o Big Bang?" | question_general | Explicação sobre origem do universo |

---

## 🎓 Conceitos Cobertos no Dataset

### Astronomia Básica
✅ Sistema Solar e seus 8 planetas  
✅ Características de cada planeta  
✅ Luas, anéis e satélites naturais  

### Astrofísica
✅ Formação e ciclo de vida de estrelas  
✅ Supernovas e explosões estelares  
✅ Buracos negros e gravidade extrema  
✅ Galáxias e estrutura do universo  

### Cosmologia
✅ Big Bang e origem do universo  
✅ Expansão do universo e energia escura  
✅ Tamanho e forma do cosmos  
✅ Matéria escura  

### Exploração Espacial
✅ História das missões espaciais  
✅ Apollo 11 e chegada à Lua  
✅ Rovers em Marte (Perseverance, Curiosity)  
✅ Estação Espacial Internacional  
✅ Futuro da colonização espacial  

### Fenômenos Astronômicos
✅ Eclipses solares e lunares  
✅ Meteoros (estrelas cadentes)  
✅ Asteroides e cometas  
✅ Constelações  

---

## 🔍 Exemplos de Perguntas que o Chatbot Pode Responder

### Sobre Planetas
- "Qual é o maior planeta do Sistema Solar?"
- "Por que Marte é vermelho?"
- "Quantas luas tem Saturno?"
- "Por que Plutão não é mais considerado planeta?"

### Sobre Estrelas
- "O que faz as estrelas brilharem?"
- "O Sol vai explodir um dia?"
- "O que é uma supernova?"
- "Por que as estrelas piscam?"

### Sobre Galáxias
- "Como se chama nossa galáxia?"
- "A Via Láctea vai colidir com outra galáxia?"
- "Quantas estrelas existem na Via Láctea?"

### Sobre Buracos Negros
- "Como um buraco negro é formado?"
- "Buracos negros engolem tudo?"
- "O que acontece se cair em um buraco negro?"

### Sobre Exploração Espacial
- "Quando o homem foi à Lua?"
- "Existe vida em outros planetas?"
- "Quanto tempo leva para chegar em Marte?"
- "O que é a Estação Espacial Internacional?"

### Sobre o Universo
- "O que foi o Big Bang?"
- "O universo está se expandindo?"
- "Qual o tamanho do universo?"
- "O que é matéria escura?"

---

## 📊 Estatísticas do Dataset

### Intenções de Treinamento
- **Total de exemplos**: 68
- **Número de classes**: 9
- **Balanceamento**: Relativamente balanceado (4-13 exemplos por classe)
- **Idioma**: Português brasileiro

### Base de Conhecimento
- **Total de documentos**: 50
- **Comprimento médio**: ~150-200 palavras por documento
- **Categorias**: 20+ categorias temáticas
- **Tópicos**: 30+ tópicos específicos
- **Nível**: Educacional - adequado para estudantes e entusiastas

---

## 🎯 Possíveis Melhorias

### Para Melhor Acurácia
1. **Adicionar mais exemplos** por intenção (alvo: 20-30 por classe)
2. **Balancear melhor** as classes (mesma quantidade de exemplos)
3. **Incluir variações** de linguagem (formal, informal, gírias)

### Para Expandir o Conhecimento
1. **Adicionar mais tópicos**:
   - Exoplanetas e zonas habitáveis
   - Telescópios e observação astronômica
   - História da astronomia
   - Mitologia das constelações
   - Relatividade e física quântica

2. **Incluir imagens/visualizações** (futuro)
3. **Adicionar quizzes interativos** (futuro)

### Para Melhorar Respostas
1. **Usar LLM local** (Ollama com Llama 3) em vez de templates
2. **Adicionar contexto conversacional** (lembrar perguntas anteriores)
3. **Implementar follow-up questions** automáticas

---

## 🚀 Próximos Passos

Após treinar o modelo:

1. ✅ Execute perguntas de teste
2. ✅ Ajuste templates de resposta conforme necessário
3. ✅ Integre com interface de chat em `/chat`
4. ✅ Compartilhe com estudantes e entusiastas de astronomia!

---

## 🌟 Exemplos de Uso Educacional

### Em Sala de Aula
- Professor pode usar como assistente para responder dúvidas
- Alunos podem explorar conceitos no próprio ritmo
- Ferramenta de estudo para provas de ciências

### Aprendizado Independente
- Curiosos podem fazer perguntas livremente
- Ótimo para crianças e adolescentes interessados em espaço
- Complemento para leitura de livros de astronomia

### Divulgação Científica
- Ferramenta para clubes de astronomia
- Apoio em eventos de observação de estrelas
- Material educativo para planetários e museus

---

## 📖 Referências e Fontes

As informações neste dataset são baseadas em:
- NASA (National Aeronautics and Space Administration)
- ESA (European Space Agency)
- Dados científicos consensuais da comunidade astronômica
- Missões espaciais reais (Apollo, Voyager, Hubble, James Webb, Mars rovers)

**Nota**: Todas as informações foram simplificadas para fins educacionais mantendo precisão científica.

---

## 🛠️ Tecnologias Utilizadas

- **FlowScope AI** - Plataforma de ML pipeline
- **scikit-learn** - Classificação de intenções (TF-IDF + Naive Bayes)
- **spaCy** - Extração de entidades nomeadas
- **sentence-transformers** - Busca semântica (all-MiniLM-L6-v2)
- **Python 3.13** - Linguagem de programação

---

## 📝 Licença

Este dataset educacional é de **uso livre para fins educacionais e não comerciais**. 

Use, modifique e compartilhe para ensinar e aprender sobre o universo! 🌌✨

---

**Feito com ❤️ e ⭐ para explorar o cosmos!**

*"O universo não é apenas mais estranho do que imaginamos, é mais estranho do que podemos imaginar." - J.B.S. Haldane*
