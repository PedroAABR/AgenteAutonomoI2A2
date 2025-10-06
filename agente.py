# agente.py
import pandas as pd
import os # <-- 1. Importe a biblioteca OS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Carrega o CSV
df = pd.read_csv("creditcard.csv")

# Inicializa o modelo Gemini Pro
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    # 2. Passe a chave de API diretamente aqui
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    convert_system_message_to_human=True
)

# Cria o agente Pandas DataFrame
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True
)

# Faça uma pergunta ao agente
pergunta = "Quantas transações fraudulentas existem neste dataset?"
resposta = agent.invoke(pergunta)

print(resposta)