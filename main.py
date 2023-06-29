from langchain import PromptTemplate, LLMChain, BasePromptTemplate
from langchain.llms import GPT4All, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import LLMChain, PromptTemplate
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import BaseMessage
from llama_index import download_loader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import os
import tempfile
from io import StringIO
from pathlib import Path
from langchain.agents import Agent
from langchain.agents import initialize_agent, Tool
from typing import List, Union, Tuple
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.prompts import BaseChatPromptTemplate, ChatMessagePromptTemplate, PromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
import streamlit as st

os.environ['OPENAI_API_KEY'] = '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'
openai_api_key = os.getenv("OPENAI_API_KEY")




st.set_page_config(page_title='🦜🔗 Ask SuSi ')
st.title('🦜🔗 Frag SuSi')


st.markdown(f"""
            <style>
            .stApp {{background-image: url("https://source.unsplash.com/jQAk1lZL5Jk"); 
                     background-attachment: fixed;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

file = ("combined.txt")
loader = TextLoader(file)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=20)
documents = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()
callbacks = [StreamingStdOutCallbackHandler()]
embeddings = OpenAIEmbeddings(model="ext-embedding-ada-002")
memory = ConversationBufferMemory()

llm = ChatOpenAI(streaming=True,
                 callbacks=callbacks,
                 temperature=0.1,
                 openai_api_key=os.getenv("OPENAI_API_KEY"),
                 model="gpt-4-0613"
                 )

memory = ConversationSummaryBufferMemory(llm=llm,
                                        output_key='answer',
                                        memory_key='chat_history',
                                        return_messages=False,)

qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                             chain_type="stuff",
                                             retriever = vectorstore.as_retriever(),  
                                             get_chat_history=lambda h:h,
                                             memory=memory,
                                             verbose = False,
                                             )

chat_history = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def generate_response(query, chat_history):
    response = qa({"question": query, "chat_history": chat_history})["answer"]
    return response

with st.form("my_form"):
    st.write("Frag Hector, unseren AI Kundensupport.")
    query = st.text_input('Wie kann man dir helfen?', placeholder='Bitte gibt deine Frage hier ein.', key='initial_question')
    submitted = st.form_submit_button("Absenden")
    
    
    if submitted:
    # Generate response and add it to the chat history
        response = generate_response(query, st.session_state.chat_history)
        st.session_state.chat_history.append((query, response))
        st.write("Antwort kommt...")


# Display the chat history
for chat_query, chat_response in st.session_state.chat_history:
    st.info(f"You: {chat_query}")
    st.success(f"Bot: {chat_response}")

# Clear chat history button
if st.button("Neue Frage"):
    st.session_state.chat_history = []