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
import time
from glob import glob
from uuid import uuid4
from typing import List, Dict, Any
import urllib3

import config 
from config import embeddings 

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_community.vectorstores import AzureSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from langchain_community.document_loaders import Docx2txtLoader

# Silencia avisos de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- pega as chaves de acesso ---
ENDPOINT_DOC_INTEL = os.environ["ENDPOINT_DOC"]
KEY_DOC_INTEL = os.environ["KEY_DOC"]

ENDPOINT_SEARCH = os.environ["ENDPOINT_SEARCH"]
KEY_SEARCH = os.environ["KEY_SEARCH"]
INDEX_NAME = os.environ["INDEX_NAME"]

PASTA_ARQUIVOS = "./data/" 

# --- funções de ajuda ---

def limpar_texto(texto: str) -> str:
    """
    Remove as tags de formatação (HTML/XML) que a IA da Azure 
    coloca no meio do Markdown e que atrapalham a busca.
    """
    if not texto: return ""
    
    substituicoes = [
        ("<figure>", ""), ("</figure>", ""),
        ("<table>", ""), ("</table>", ""),
        ("", ""),
    ]
    
    texto_limpo = texto
    for sujo, limpo in substituicoes:
        texto_limpo = texto_limpo.replace(sujo, limpo)
        
    return texto_limpo

def file_already_indexed(filename: str) -> bool:
    """
    verifica se o arquivo já foi lido antes para não gastar tempo/dinheiro à toa.
    """
    try:
        # Adicionada verificação de conexão False para o Proxy
        client = SearchClient(
            endpoint=ENDPOINT_SEARCH, 
            index_name=INDEX_NAME, 
            credential=AzureKeyCredential(KEY_SEARCH),
            connection_verify=False
        )
        
        # pesquisa se o nome do arquivo existe nos metadados
        results = client.search(
            search_text=f"\"{filename}\"", 
            search_fields=["metadata"], 
            top=1
        )
        
        for _ in results:
            return True 
            
        return False
    except Exception:
        return False

def get_markdown_with_azure(file_path: str) -> str:
    """
    usa a ia da azure para ler o pdf.
    essa função é essencial para tabelas: ela converte o visual da tabela 
    em texto organizado (markdown) que a ia entende.
    """
    client = DocumentIntelligenceClient(
        endpoint=ENDPOINT_DOC_INTEL, 
        credential=AzureKeyCredential(KEY_DOC_INTEL),
    )

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout", 
            body=f,
            output_content_format="markdown" 
        )
        result = poller.result()
    
    return result.content

def read_word_file(file_path: str) -> str:
    """lê arquivos do word (.docx) transformando em texto simples"""
    try:
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        return "\n".join([d.page_content for d in docs])
    except Exception as e:
        print(f"erro ao ler word {file_path}: {e}")
        return ""


def ingest_files():
    print(f"🚀 Iniciando ingestão no índice: {INDEX_NAME}")

    # 1. valida se a pasta existe
    if not os.path.exists(PASTA_ARQUIVOS):
        print(f"🔴 ERRO: A pasta '{PASTA_ARQUIVOS}' não existe.")
        return

    # 2. procura pdfs e docx na pasta
    arquivos_para_processar = []
    arquivos_para_processar.extend(glob(os.path.join(PASTA_ARQUIVOS, "*.pdf")))
    arquivos_para_processar.extend(glob(os.path.join(PASTA_ARQUIVOS, "*.docx")))

    # filtra os que já foram processados
    print("🔍 Procurando arquivos novos...")
    arquivos_novos = []
    for caminho in arquivos_para_processar:
        nome = os.path.basename(caminho)
        if file_already_indexed(nome):
            print(f"⏩ {nome} já existe. Pulando.")
        else:
            arquivos_novos.append(caminho)

    if not arquivos_novos:
        print("✅ Tudo atualizado! Nada novo para ler.")
        return

    print(f"📦 Achei {len(arquivos_novos)} arquivos novos.")

    # 3. chunks
    # usamos 4000 caracteres para tentar manter tabelas grandes inteiras no mesmo bloco
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        separators=["\n## ", "\n\n", "\n", " ", ""],
        length_function=len,
    )

    # conecta com o banco de dados da azure
    vector_store = AzureSearch(
        azure_search_endpoint=ENDPOINT_SEARCH,
        azure_search_key=KEY_SEARCH,
        index_name=INDEX_NAME,
        embedding_function=embeddings,
        # Adicionado para evitar erro de conexão pelo Proxy (WinError 10054)
        additional_search_client_options={"connection_verify": False}
    )

    # 4. começa a leitura arquivo por arquivo
    for i, caminho in enumerate(arquivos_novos):
        nome_arquivo = os.path.basename(caminho)
        extensao = nome_arquivo.split('.')[-1].lower()
        print(f"[{i+1}/{len(arquivos_novos)}] 📄 Lendo: {nome_arquivo}...")
        
        texto_extraido = ""
        
        try:
            # decide qual função usar baseado no tipo de arquivo
            if extensao == "pdf":
                # aqui acontece a mágica das tabelas
                texto_extraido = get_markdown_with_azure(caminho)
            elif extensao == "docx":
                texto_extraido = read_word_file(caminho)
            
            if not texto_extraido:
                print(f"⚠️ Arquivo vazio: {nome_arquivo}")
                continue

            # Limpa o texto antes de cortar
            texto_extraido = limpar_texto(texto_extraido)

            # quebra o texto em pedaços menores
            chunks_texto = text_splitter.split_text(texto_extraido)
            
            # prepara os dados para envio com metadados
            docs_para_upload = []
            for chunk in chunks_texto:
                docs_para_upload.append({
                    "page_content": chunk,
                    "metadata": {
                        "filepathField": nome_arquivo,
                        "source": nome_arquivo,
                        "chunk_id": str(uuid4())
                    }
                })
            
            # converte para o formato que o langchain gosta
            final_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs_para_upload]
            
            # envia para a nuvem em pacotes de 10 em 10 para não travar
            total = len(final_docs)
            print(f"   💾 Gravando {total} partes do texto...")

            for k in range(0, total, 10):
                batch = final_docs[k : k + 10]
                try:
                    vector_store.add_documents(batch)
                    time.sleep(1) # pausa rápida para não sobrecarregar
                except Exception:
                    print("   ⚠️ Azure pediu para esperar. Aguardando 30s...")
                    time.sleep(30)
                    vector_store.add_documents(batch)

            print(f"   🟢 Sucesso!")

        except Exception as e:
            print(f"🔴 ERRO ao ler {nome_arquivo}: {e}")

    print("\n🏁 Processo finalizado.")

if __name__ == "__main__":
    ingest_files()