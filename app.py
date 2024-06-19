# # Installer FAISS avant d'utiliser les imports
# #!pip install faiss-cpu

# # Vous pouvez également utiliser faiss-gpu si vous avez une GPU compatible CUDA
# # !pip install faiss-gpu

# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
# from dotenv import load_dotenv
# from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.agents import AgentExecutor, create_tool_calling_agent

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # Ensure SpaCy and the language model are installed and loaded
# import spacy
# try:
#     nlp = spacy.load("en_core_web_sm")
# except:
#     import subprocess
#     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
#     nlp = spacy.load("en_core_web_sm")

# embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# def pdf_read(pdf_doc):
#     text = ""
#     for pdf in pdf_doc:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def vector_store(text_chunks):
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_db")

# def get_conversational_chain(tools, ques):
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key="sk-proj-VLhTzwjSc8uvBxrVI1VQT3BlbkFJPD9cL13uoD5olNuV0hto")
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer""",
#             ),
#             ("placeholder", "{chat_history}"),
#             ("human", "{input}"),
#             ("placeholder", "{agent_scratchpad}"),
#         ]
#     )
#     tool = [tools]
#     agent = create_tool_calling_agent(llm, tool, prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
#     response = agent_executor.invoke({"input": ques})
#     print(response)
#     st.write("Reply: ", response['output'])

# def user_input(user_question):
#     new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
#     retriever = new_db.as_retriever()
#     retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answer to queries from the pdf")
#     get_conversational_chain(retrieval_chain, user_question)

# def main():
#     st.set_page_config("Chat PDF")
#     st.header("RAG based Chat with PDF")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = pdf_read(pdf_doc)
#                 text_chunks = get_chunks(raw_text)
#                 vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()
# Installer FAISS avant d'utiliser les imports
#!pip install faiss-cpu

# Vous pouvez également utiliser faiss-gpu si vous avez une GPU compatible CUDA
# !pip install faiss-gpu

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from langchain.tools.retriever import create_retriever_tool

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure SpaCy and the language model are installed and loaded
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load pre-trained model for embeddings (can be customized)
from langchain.embeddings import SpacyEmbeddings

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

# Définition de la fonction create_retriever_tool
# def create_retriever_tool(retriever, tool_name, description):
#     def tool(question):
#         docs = retriever.retrieve(question)
#         if docs:
#             context = docs[0].page_content
#         else:
#             context = "No relevant context found."
#         return context
#     tool.tool_name = tool_name
#     tool.description = description
#     return tool

def get_conversational_chain(tools, ques):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    context = tools(ques)
    response = qa_pipeline(question=ques, context=context)
    st.write("Reply: ", response['answer'])

def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_tool = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answer to queries from the pdf")
    get_conversational_chain(retrieval_tool, user_question)

def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
