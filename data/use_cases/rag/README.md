# 🤖 Sistema RAG (Retrieval-Augmented Generation)

Sistema de chat com contexto que responde perguntas baseado em conhecimento carregado.

## 📚 O que é RAG?

RAG permite que você:
1. **Carregue conhecimento** (texto ou planilha)
2. **Faça perguntas**
3. **Receba respostas** baseadas no contexto

## 🎯 Formatos Suportados

### 1. Arquivo TXT (Texto Corrido)
- Sistema divide em chunks automaticamente
- Ideal para: documentos, manuais, artigos

**Exemplo:** `loja_roupas.txt`

### 2. Arquivo CSV (Perguntas e Respostas)
- Modo Q&A: colunas `pergunta` e `resposta`
- Modo Documentos: coluna única com textos

**Exemplo:** `faq_loja.csv`

## 🚀 Como Usar

### Via API

#### 1. Carregar conhecimento de TXT

```bash
POST http://localhost:8000/chat/rag/load-text
{
  "file_path": "data/use_cases/rag/loja_roupas.txt",
  "chunk_size": 200,
  "name": "catalogo_produtos"
}
```

#### 2. Carregar conhecimento de CSV (modo Q&A)

```bash
POST http://localhost:8000/chat/rag/load-csv
{
  "file_path": "data/use_cases/rag/faq_loja.csv",
  "question_column": "pergunta",
  "answer_column": "resposta",
  "name": "faq_loja"
}
```

#### 3. Fazer perguntas

```bash
POST http://localhost:8000/chat/rag/ask
{
  "question": "Qual a cor da camisa?",
  "top_k": 3,
  "min_similarity": 0.1
}
```

**Resposta:**
```json
{
  "status": "success",
  "answer": "A camisa básica é vermelha. Também temos camisa polo azul, camisa social branca e camisa estampada floral em verde e rosa.",
  "confidence": 0.95,
  "contexts": [...]
}
```

#### 4. Ver estatísticas

```bash
GET http://localhost:8000/chat/rag/stats
```

### Via Interface Web

1. **Carregue conhecimento** usando a API
2. **Vá para página de Chat**
3. **Selecione modelo RAG** (se disponível)
4. **Faça perguntas naturalmente!**

## 📝 Exemplos de Perguntas

Com `faq_loja.csv` carregado:
- ❓ "Oi, tudo bem?" → "Olá! Tudo ótimo, e você?"
- ❓ "Qual a cor da camisa?" → "A camisa básica é vermelha..."
- ❓ "Quanto custa?" → "A camisa básica vermelha custa R$ 45,00..."
- ❓ "Vocês fazem entrega?" → "Sim! Frete grátis para compras acima de R$ 150,00..."

Com `loja_roupas.txt` carregado:
- ❓ "Fale sobre as jaquetas" → (retorna chunk sobre jaquetas)
- ❓ "Como funciona a troca?" → (retorna política de troca)
- ❓ "Qual o horário?" → (retorna horário de funcionamento)

## 🔧 Parâmetros Ajustáveis

### `top_k` (padrão: 3)
- Número de contextos a buscar
- Maior = mais contexto, mas pode incluir irrelevante

### `min_similarity` (padrão: 0.1)
- Similaridade mínima (0.0 a 1.0)
- Maior = respostas mais precisas, mas pode não encontrar nada

### `chunk_size` (padrão: 200, apenas TXT)
- Tamanho dos chunks em palavras
- Menor = busca mais precisa
- Maior = mais contexto por chunk

## 🎨 Como Funciona

1. **Indexação:** TF-IDF vetoriza o conhecimento
2. **Busca:** Similaridade de cosseno entre pergunta e documentos
3. **Resposta:** Retorna o contexto mais relevante

## 🆚 TXT vs CSV

| Característica | TXT | CSV (Q&A) |
|----------------|-----|-----------|
| **Estrutura** | Texto livre | Pergunta + Resposta |
| **Precisão** | Média | Alta |
| **Flexibilidade** | Alta | Média |
| **Ideal para** | Documentos longos | FAQs, atendimento |

## 💡 Dicas

✅ **Use CSV** quando tiver perguntas frequentes conhecidas  
✅ **Use TXT** quando tiver documentação extensa  
✅ **Combine ambos!** Carregue FAQ + manual completo  
✅ **Teste `top_k` e `min_similarity`** para ajustar qualidade

## 🔄 Trocar Conhecimento

Para carregar novo conhecimento, basta chamar `load-text` ou `load-csv` novamente. O conhecimento anterior será substituído.

## 📊 Monitoramento

Use `/chat/rag/stats` para ver:
- Quantos documentos estão carregados
- Tipo de conhecimento (Q&A vs documentos)
- Nome do conhecimento atual
