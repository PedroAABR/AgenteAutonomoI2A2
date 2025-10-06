# app.py
import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- CONSTANTES ---
NOME_ARQUIVO_GRAFICO = "temp_grafico.png"

# Função para criar e executar o agente
def rodar_agente(dataframe, pergunta, historico_chat):
    # Formata o histórico para incluir no prompt
    historico_formatado = "\n".join(historico_chat)

    prompt_com_historico = f"""Você é um agente de análise de dados.
    Baseado no histórico da conversa abaixo
    , responda à nova pergunta do usuário.
    Se o usuário pedir um gráfico, gere o código python para criá-lo e OBRIGATORIAMENTE salve a imagem no arquivo '{NOME_ARQUIVO_GRAFICO}'.

    Histórico da Conversa:
    ---
    {historico_formatado}
    ---
    Nova Pergunta: {pergunta}
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        convert_system_message_to_human=True
    )
    
    agent = create_pandas_dataframe_agent(
        llm,
        dataframe,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    # LÓGICA PARA GRÁFICOS: Apaga o gráfico anterior, se existir
    if os.path.exists(NOME_ARQUIVO_GRAFICO):
        os.remove(NOME_ARQUIVO_GRAFICO)

    # Executa o agente
    with st.spinner('Analisando os dados e gerando a resposta...'):
        resposta = agent.invoke(prompt_com_historico)
        return resposta['output']

# --- Configuração da Página do Streamlit ---
st.set_page_config(page_title="Agente de Análise de CSV", layout="wide")
st.title("🤖 Agente de Análise de Dados de CSV")

# LÓGICA PARA MEMÓRIA: Inicializa o histórico do chat no session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Upload do Arquivo ---
# Reinicia o chat se um novo arquivo for carregado
arquivo_csv = st.file_uploader("Selecione um arquivo CSV", type="csv", key="file_uploader")

if arquivo_csv is not None:
    # Lógica para limpar o histórico quando um novo arquivo é enviado
    if 'current_file' not in st.session_state or st.session_state.current_file != arquivo_csv.name:
        st.session_state.current_file = arquivo_csv.name
        st.session_state.chat_history = []
        st.success(f"Arquivo '{arquivo_csv.name}' carregado! O histórico foi reiniciado.")

    df = pd.read_csv(arquivo_csv)
    
    # Exibe o histórico do chat
    with st.chat_message("assistant"):
        st.write("Olá! Sou seu agente de análise de dados. O que você gostaria de saber sobre este arquivo?")
    
    for i, message in enumerate(st.session_state.chat_history):
        # Alterna entre mensagem do usuário e do agente
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message)
        else:
            with st.chat_message("assistant"):
                st.write(message)

    # --- Interação com o Usuário ---
    pergunta = st.chat_input("Faça sua pergunta sobre o arquivo...")
    
    if pergunta:
        with st.chat_message("user"):
            st.write(pergunta)

        with st.chat_message("assistant"):
            resposta_agente = rodar_agente(df, pergunta, st.session_state.chat_history[-4:]) # Usa apenas as 2 últimas interações (4 mensagens) como contexto
            st.write(resposta_agente)

            # LÓGICA PARA GRÁFICOS: Verifica se um gráfico foi criado e o exibe
            if os.path.exists(NOME_ARQUIVO_GRAFICO):
                st.image(NOME_ARQUIVO_GRAFICO)
                os.remove(NOME_ARQUIVO_GRAFICO) # Limpa o arquivo após exibir

        # LÓGICA PARA MEMÓRIA: Adiciona a pergunta e a resposta ao histórico
        st.session_state.chat_history.append(pergunta)
        st.session_state.chat_history.append(resposta_agente)

else:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")