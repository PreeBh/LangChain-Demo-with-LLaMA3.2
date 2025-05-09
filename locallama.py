from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

st.title('LangChain Demo with LLaMA3.2 via Ollama')
st.write("Streamlit is working!") 

input_text = st.text_input("Ask your question:")

try:
    llm = Ollama(model="llama3.2")  # Your custom model name
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
except Exception as model_error:
    st.error(f"Model load error: {model_error}")

if input_text:
    try:
        response = chain.invoke({"question": input_text})
        st.write(response)
    except Exception as e:
        st.error(f"Error during inference: {e}")
