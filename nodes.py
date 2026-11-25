# Carrega variáveis do arquivo .env mais próximo (pasta do projeto)
def _carregar_env():
    from pathlib import Path
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for pasta in Path(__file__).resolve().parents:
        if (pasta / ".env").exists():
            load_dotenv(pasta / ".env")
            break


_carregar_env()

import os
from langchain_community.vectorstores import AzureSearch
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from config import embeddings, llm
from prompt import agent_prompt
from state import AgentState

# --- CONFIGURAÇÃO DA CONEXÃO AZURE ---
try:
    vectorstore = AzureSearch(
        azure_search_endpoint=os.environ["ENDPOINT_SEARCH"],
        azure_search_key=os.environ["KEY_SEARCH"],
        index_name=os.environ["INDEX_NAME"],
        embedding_function=embeddings
    )
except Exception as e:
    print(f"🔴 Erro ao conectar no Azure Search: {e}")
    vectorstore = None

# --- FUNÇÃO DO CLASSIFICADOR (ROUTER) ---
def should_search_documents(query: str) -> bool:
    """
    Pergunta para a LLM se vale a pena buscar documentos para essa frase.
    Retorna True (Buscar) ou False (Não Buscar).
    """
    router_template = """
    Você é um classificador de intenções para um Assistente de Sinistros da Acme Seguros.
    Analise a entrada do usuário e responda APENAS com uma das palavras: "BUSCAR" ou "PULAR".

    CRITÉRIOS:
    - Responda **BUSCAR** se for: Dúvidas sobre seguros, sinistros, apólices, documentos, regras, pagamentos, ressarcimento, manuais.
    - Responda **PULAR** se for: Saudações (oi, bom dia), Conversa fiada (receitas, piadas, clima), Perguntas de identidade (quem é você, ajuda), ou nonsense (teste, asdjasd).

    ENTRADA DO USUÁRIO: "{input}"
    
    SUA CLASSIFICAÇÃO:
    """
    
    try:
        prompt = PromptTemplate.from_template(router_template)
        chain = prompt | llm
        result = chain.invoke({"input": query})
        
        # Limpa a resposta para garantir (tira espaços e deixa maiúsculo)
        decision = result.content.strip().upper()
        
        print(f"🤖 Decisão do Router para '{query}': {decision}")
        
        return "BUSCAR" in decision
        
    except Exception as e:
        print(f"⚠️ Erro no Router (Padrão = Buscar): {e}")
        return True # Na dúvida, busca.

# --- NÓ DE BUSCA (RETRIEVE) ---
def retrieve_node(state: AgentState):
    query = state["query"]
    
    # 1. PERGUNTA SE PRECISA BUSCAR
    precisa_buscar = should_search_documents(query)
    
    # Se o Router disser que não precisa, retorna vazio imediatamente
    if not precisa_buscar:
        print(f"⏩ Router decidiu PULAR a busca para: '{query}'")
        return {"context": []}

    # 2. VERIFICAÇÃO DE CONEXÃO
    if not vectorstore:
        return {"context": []}

    # 3. BUSCA VETORIAL 
    documents = vectorstore.similarity_search(query, k=4)
    
    # DEBUG
    print(f"\n🔍 --- BUSCA AZURE REALIZADA ---")
    return {"context": documents}

# --- NÓ DE GERAÇÃO (GENERATE) ---
def generate_node(state: AgentState):
    query = state["query"]
    context = state["context"] 
    messages = state["messages"] 
    
    history_text = "\n".join(
        [f"{msg.type.upper()}: {msg.content}" for msg in messages[:-1]] 
    )
    
    if context:
        context_text = "\n\n".join([doc.page_content for doc in context])
    else:
        context_text = "" 
    
    chain = agent_prompt | llm
    response = chain.invoke({
        "context": context_text, 
        "question": query,
        "history": history_text 
    })
    
    sources_data = []
    for doc in context:
        sources_data.append({
            "filepathField": doc.metadata.get("filepathField", "Desconhecido"),
            "content": doc.page_content
        })

    final_message = AIMessage(
        content=response.content,
        name="writer",
        additional_kwargs={"sources": sources_data} 
    )
    
    return {"messages": [final_message]}
