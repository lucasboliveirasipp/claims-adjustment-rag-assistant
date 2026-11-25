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
import httpx  # Cliente HTTP seguro
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# # 1. Proxy e cliente seguro

http_client = httpx.Client(trust_env=True, verify=False)

# 2. Credenciais
# Document Intelligence (PDF)

# Azure AI Search
os.environ["INDEX_NAME"] = "agente-sinistro-regulacao-v1"

# 3. Embeddings (servidor antigo)
# os.environ["API_VERSION"] = "2023-05-15"

os.environ["API_VERSION"] = "2023-05-15"


embeddings = AzureOpenAIEmbeddings(
    # azure_deployment="text-embedding-ada-002",
    azure_deployment="text-embedding-ada-002",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["OPENAI_API_KEY"],
    openai_api_version=os.environ["API_VERSION"],
    openai_api_type="azure",
    http_client=http_client
)

# 4. Chat (servidor novo)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    openai_api_version="2024-08-01-preview",
    deployment_name="gpt-4.1-mini",
    openai_api_type="azure",
    temperature=0,
    http_client=http_client
)

os.environ["GPT4"] = "gpt-4.1-mini"

print("✅ PARÂMETROS CARREGADOS (config.py)!")