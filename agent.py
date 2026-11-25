from __future__ import annotations

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

import sys
import os
import logging

# --- 1. configuração de caminho (path) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 2. demais importações ---
import re
import pandas as pd
from uuid import uuid4
from typing import Dict, List, Optional

# bibliotecas externas
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SISTEMA DE FEEDBACK E LOGS ---
FEEDBACK_AVAILABLE = False
try:
    from feedback_local import update_feedback
    from feedback_local import log_interaction
    FEEDBACK_AVAILABLE = True
    logger.info("✅ sistema de feedback carregado")
except ImportError as e:
    logger.warning(f"⚠️ sistema de feedback não disponível: {e}")
    def update_feedback(*args, **kwargs): pass
    def log_interaction(*args, **kwargs): pass

AGENT_NAME = os.path.basename(current_dir)

# --- importação do estilo visual ---
try:
    from streamlit_style import (
        init_page_config,
        apply_app_style,
        build_institutional_header,
        build_footer
    )
except ImportError as e:
    st.error(f"ERRO CRÍTICO: não foi possível importar 'streamlit_style'. detalhe: {e}")
    st.stop()

# --- configuração inicial ---

try:
    from graph_builder import graph 
except ImportError:
    class MockGraph:
        def get_state(self, *args, **kwargs): return {}
        def stream(self, *args, **kwargs): return []
    graph = MockGraph()

# --- 4. funções de dados (azure search) ---
@st.cache_data(ttl=3600)
def get_document_catalog():
    """busca catálogo de manuais no azure search."""
    try:
        endpoint = os.environ.get("ENDPOINT_SEARCH")
        key = os.environ.get("KEY_SEARCH")
        index = os.environ.get("INDEX_NAME")
        
        if not all([endpoint, key, index]):
            return pd.DataFrame()

        client = SearchClient(endpoint=endpoint, index_name=index, credential=AzureKeyCredential(key))
        results = client.search(search_text="*", top=3000)
        
        docs, seen = [], set()
        for res in results:
            fname = (
                res.get("source") or 
                res.get("filepathField") or 
                res.get("title") or 
                res.get("metadata_storage_name") or 
                res.get("url") or
                res.get("name")
            )
            
            if fname and fname not in seen:
                clean_name = os.path.basename(fname)
                clean_name = clean_name.replace(".pdf", "").replace("_", " ").title()
                year_match = re.search(r"(202[0-9])", fname)
                ref = f"Ref. {year_match.group(1)}" if year_match else "Geral/Vigente"
                docs.append({
                    "Manual / Norma": clean_name, 
                    "Referência Temporal": ref, 
                    "Arquivo Técnico": fname
                })
                seen.add(fname)
        
        if docs:
            return pd.DataFrame(docs).sort_values("Manual / Norma")
        return pd.DataFrame() 

    except Exception as e:
        print(f"Erro detalhado no catálogo: {e}")
        return pd.DataFrame()

# --- 5. css local (estilo dos cards) ---
def _inject_component_styles():
    st.markdown("""
    <style>
        .status-banner {
            background-color: #FFFFFF;
            color: #2D3748;
            padding: 16px 24px;
            border-radius: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 24px;
            border: 1px solid #EDF2F7;
            display: flex; align-items: center; justify-content: space-between;
            font-family: 'Segoe UI', sans-serif !important;
        }
        .status-text {
            color: #1E3A8A;
            font-weight: 700; letter-spacing: 0.5px;
            text-transform: uppercase; font-size: 0.9rem;
        }
        .status-pill {
            background-color: #1E3A8A;
            color: white !important;
            padding: 6px 16px; border-radius: 6px;
            font-weight: 600; font-size: 0.85rem;
        }
        .manual-card h3 {
            color: #1E3A8A !important;
            font-family: 'Segoe UI', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.25rem !important;
            margin-bottom: 12px !important;
        }
        .manual-card p, .manual-card li, .manual-card span {
            color: #2D3748 !important;
            font-family: 'Segoe UI', sans-serif !important;
            line-height: 1.6;
        }
        .file-item {
            display: flex; align-items: center; gap: 10px; 
            padding: 8px 0; border-bottom: 1px solid #EDF2F7;
            color: #2D3748;
            font-family: 'Segoe UI', sans-serif !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 6. componentes de interface (fontes) ---

def is_generic_query(query: str) -> bool:
    """
    Detecta se a pergunta é genérica/interação social para evitar mostrar fontes.
    """
    if not query: return True
    
    query = query.lower().strip()
    
    # Lista de termos que indicam interação básica sem necessidade de documentos
    generic_terms = [
        "oi", "ola", "olá", "bom dia", "boa tarde", "boa noite",
        "tudo bem", "quem é você", "quem e voce", "o que você faz",
        "o que voce faz", "ajuda", "help", "teste",
        "sobre o que posso perguntar", "quais assuntos", "o que você sabe"
    ]
    
    # Se a query for muito curta (< 5 chars) ou contiver termos genéricos
    is_short_greeting = len(query) < 5
    is_common_phrase = any(term in query for term in generic_terms)
    
    return is_short_greeting or is_common_phrase

def _render_sources(sources: List[Dict], query: str = "") -> None:
    """
    Renderiza a lista de fontes consultadas no rodapé da mensagem.
    Filtra interações genéricas (chit-chat).
    """
    if is_generic_query(query):
        return

    if not sources: return

    unique_files = set()
    clean_list = []

    for source in sources:
        # Lógica mista (novo e antigo)
        raw_name = source.get("source")
        if not raw_name:
            raw_name = source.get("filepathField")
        if not raw_name:
            raw_name = source.get("filename") or "Desconhecido"
        
        name = os.path.basename(raw_name)

        if name and name != "Desconhecido" and name != "Documento sem nome" and name not in unique_files:
            unique_files.add(name)
            clean_list.append(name)

    if not clean_list: return

    with st.expander("📚 Fontes Consultadas", expanded=False):
        for name in clean_list:
            st.markdown(f"""
            <div class="file-item">
                <span style="font-size: 1.2em;">📄</span>
                <span style="font-weight: 500;">{name}</span>
            </div>""", unsafe_allow_html=True)

# --- função auxiliar catalogo local ---
def get_local_file_catalog():
    import os
    import pandas as pd
    import streamlit as st 
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, "data", "atualizacao", "documentos_atualizacao.xlsx")
    
    if not os.path.exists(excel_path):
        return pd.DataFrame()

    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
        if "Data do Documento" in df.columns:
            df["Data do Documento"] = pd.to_datetime(df["Data do Documento"], errors='coerce')
            df = df.sort_values(by="Data do Documento", ascending=False)
            df["Data do Documento"] = df["Data do Documento"].dt.strftime('%d/%m/%Y').fillna("-")
        df = df.fillna("-").astype(str)
        return df
    except Exception as e:
        return pd.DataFrame()

# --- 7. página do guia rápido (manual) ---
def render_manual_page():
    tab1, tab2 = st.tabs(["📍 Comece por Aqui", "📂 Base de Conhecimento"])
    
    with tab1:
        st.markdown("""
        <div class="status-banner">
            <div class="status-text">ESPECIALISTA</div>
            <div class="status-pill">Status: Online 🟢</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="manual-card">
            <h3>✅ Capacidades</h3>
            <p style="margin-bottom: 15px;">
                Atuo como seu apoiador estratégico na tomada de decisão. Minha missão é varrer a complexidade técnica dos manuais 
                para entregar a <b>fundamentação técnica necessária</b>.
            </p>
            <p>
                Sou uma inteligência focada exclusivamente em consulta documental: <b>não acesso dados de apólices em tempo real e não realizo transações</b>. 
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="manual-card">
            <h3>🔍 O que perguntar?</h3>
            <ul style="padding-left: 20px;">
                <li style="margin-bottom:10px;"><b>Coberturas e Riscos:</b> validação técnica tanto para Auto (ex: vidros, carrocerias) quanto para RE (ex: danos elétricos, vendaval).</li>
                <li style="margin-bottom:10px;"><b>Procedimentos:</b> regras para liquidação, prazos e franquias.</li>
                <li><b>Documentação:</b> checklists obrigatórios customizados para cada natureza de sinistro.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="manual-card" style="margin-bottom: 10px; padding-bottom: 10px;">
            <h3>💡 Exemplos de Perguntas</h3>
            <p style="font-style: italic; font-size: 0.9em; margin-bottom: 10px;">Clique no ícone de copiar no canto direito de cada pergunta:</p>
        </div>
        """, unsafe_allow_html=True)

        exemplos = [
            "Como identificar se é uma apólice de Extrato Unificado?",
            "Qual checklist técnico de Exclusões e Impedimentos para o serviço de Instalação de Suporte de TV?",
            "Como funciona a Locação de Veículo pelo Reclamante?",
            "O prestador cobrou R$ 450,00 de mão de obra para um serviço geral. Qual valor a seguradora cobre e quanto o segurado deve pagar?"
        ]

        for pergunta in exemplos:
            st.code(pergunta, language="text")
        
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="manual-card">
            <h3>🚫 Limitações</h3>
            <ul style="padding-left: 20px;">
                <li style="margin-bottom:8px;">Minhas respostas baseiam-se estritamente nos documentos ingeridos na ferramenta.</li>
                <li style="margin-bottom:8px;">Caso identifique algum manual desatualizado, contate a equipe de IA Generativa.</li>
                <li style="margin-bottom:8px;"><b>Subjetividade:</b> não analiso fotos, dinâmicas de acidente ou áudios. Análise apenas textual.</li>
                <li><b>Ação Financeira:</b> eu sugiro e embaso, mas a decisão final é sua.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📄 Documentos Disponíveis")
        st.markdown("Estes arquivos foram adicionados ao projeto e são usados pelo agente como referência.")
        
        with st.spinner("Lendo planilha de documentos..."):
            df = get_local_file_catalog()
        
        if df is not None and not df.empty:
            st.dataframe(
                df, 
                hide_index=True, 
                use_container_width=True, 
                height=500,
                column_config={
                    "Nome do Documento": st.column_config.TextColumn("Nome do Documento", width="large"),
                    "Data do Documento": st.column_config.TextColumn("Data / Vigência", width="medium"),
                }
            )
        else:
            st.warning("⚠️ O arquivo de referência na pasta 'data/atualizacao' não foi encontrado ou está vazio.")
            
# --- 8. estado do grafo ---
def _get_graph_state(thread_id: str) -> List[BaseMessage]:
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", []) if state and state.values else []

# --- 9. app principal (interface) ---
def build_page(is_authenticated: bool = True, *args, **kwargs) -> None:
    apply_app_style()
    _inject_component_styles()

    if "thread_id" not in st.session_state: st.session_state["thread_id"] = str(uuid4())
    if "current_page" not in st.session_state: st.session_state["current_page"] = "chat"
    if "edit_mode" not in st.session_state: st.session_state["edit_mode"] = False
    
    if "latest_response_id" not in st.session_state: st.session_state["latest_response_id"] = None
    
    with st.sidebar:
        st.markdown("### Navegação")
        if st.button("💬  Agente de Regulação", type="primary" if st.session_state["current_page"] == "chat" else "secondary"):
            st.session_state["current_page"] = "chat"
            st.rerun()

        if st.button("📘 Guia Rápido", type="primary" if st.session_state["current_page"] == "manual" else "secondary"):
            st.session_state["current_page"] = "manual"
            st.rerun()
        
        st.markdown("---")
        
        if st.session_state["current_page"] == "chat":
            st.markdown("### 🛠️ Controles")
            if st.button("Limpar Conversa", icon="🗑️", type="primary"):
                st.session_state["thread_id"] = str(uuid4())
                st.session_state["edit_mode"] = False
                st.session_state["latest_response_id"] = None 
                st.rerun()
            
            history = _get_graph_state(st.session_state["thread_id"])
            if history and isinstance(history[-1], AIMessage):
                last_human = next((m for m in reversed(history) if isinstance(m, HumanMessage)), None)
                if last_human:
                    if st.button("Refazer Última", type="secondary"): 
                        st.session_state["edit_mode"] = True
                        st.session_state["text_to_edit"] = last_human.content
                        st.rerun()

    if st.session_state["current_page"] == "manual":
        build_institutional_header(
            title="📘 Guia Rápido", 
            subtitle="Tudo que você precisa saber: boas práticas, orientações e manual de referência",
            show_logo=False 
        )
        render_manual_page()
        build_footer()
    else:
        build_institutional_header(
            title="Agente de Regulação Auto/RE", 
            subtitle="Seu parceiro para consulta, análise técnica e regulação de processos de sinistro."
        )
        
        thread_id = st.session_state["thread_id"]
        config = {"configurable": {"thread_id": thread_id}}
        history = _get_graph_state(thread_id)
        final_prompt = None

        if st.session_state["edit_mode"]:
            st.info("📝 **Editando pergunta:**")
            with st.form("edit_form"):
                txt = st.text_area("Sua pergunta:", value=st.session_state.get("text_to_edit", ""))
                c1, c2 = st.columns([1, 4])
                if c1.form_submit_button("🚀 Reenviar", type="primary"):
                    final_prompt = txt
                    st.session_state["edit_mode"] = False
                if c2.form_submit_button("❌ Cancelar"):
                    st.session_state["edit_mode"] = False
                    st.rerun()
        else:
            if user_input := st.chat_input("Digite sua pergunta..."):
                final_prompt = user_input

        chat_container = st.container()
        with chat_container:
            # LOOP DO HISTÓRICO
            for i, msg in enumerate(history):
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                with st.chat_message(role):
                    st.markdown(msg.content)
                    if role == "assistant": 
                        user_q = ""
                        if i > 0 and isinstance(history[i-1], HumanMessage):
                            user_q = history[i-1].content
                        
                        _render_sources(msg.additional_kwargs.get("sources", []), query=user_q)
                        
                        if i == len(history) - 1:
                            if st.session_state.get("latest_response_id"):
                                update_feedback(
                                    agent_name=AGENT_NAME, 
                                    thread_id=thread_id, 
                                    response_id=st.session_state["latest_response_id"]
                                )

            if final_prompt and not st.session_state.get("edit_mode"):
                with st.chat_message("user"): 
                    st.markdown(final_prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Formulando uma resposta..."):
                        inputs = {"messages": [HumanMessage(content=final_prompt)], "query": final_prompt}
                        events = graph.stream(inputs, config, stream_mode="values")
                        
                        final_state = {}
                        for event in events: final_state = event
                        
                        if msgs := final_state.get("messages"):
                            resp = next((m for m in reversed(msgs) if isinstance(m, AIMessage) and m.name == "writer"), None)
                            if resp:
                                st.markdown(resp.content)
                                _render_sources(resp.additional_kwargs.get("sources", []), query=final_prompt)
                                
                                if FEEDBACK_AVAILABLE:
                                    response_id = str(uuid4())
                                    st.session_state["latest_response_id"] = response_id
                                    
                                    try:
                                        log_interaction(
                                            agent_name=AGENT_NAME,
                                            thread_id=thread_id,
                                            response_id=response_id,
                                            user_question=final_prompt,
                                            chatbot_response=resp.content
                                        )
                                        st.rerun() 
                                    except Exception as e:
                                        logger.error(f"Erro ao logar: {e}")
                
                if final_prompt and "edit_form" in str(st.session_state): st.rerun()