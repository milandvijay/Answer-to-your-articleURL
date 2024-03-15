import os 
import streamlit as st
import pickle 
from langchain import OpenAI 
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

os.environ["OPENAI-API-KEY"]='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'
llm = OpenAI(openai_api_key = os.environ["OPENAI-API-KEY"],temperature= 0.9,max_tokens=1000,model="gpt-3.5-turbo-instruct")
st.title("Answer to Your Article ")

url = st.text_input("Enter your URL")
URLs = []
URLs.append(url)

# creating the file path 
file_path = "faiss_store_openai.pkl"
# submit button 

clicked = st.button("Process the URL")

main_placefolder = st.empty()

if clicked:
    loader = UnstructuredURLLoader(urls=URLs)
    main_placefolder.text("Data Loading....started...")
    data = loader.load()

    # text splitter function and splitting 
    text_spliter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n',',','.'],
        chunk_size = 1000
    )
    docs = text_spliter.split_documents(data)

    # FAISS and embedding 
    embeddings = OpenAIEmbeddings(openai_api_key = os.environ["OPENAI-API-KEY"])
    vector = FAISS.from_documents(docs,embeddings)

    with open(file_path,'wb') as f:
        pickle.dump(vector,f)

# quetion box 
query = st.text_input("What's your question ")

if query:
    if os.path.exists(file_path):
        with open(file_path,'rb') as f :
            vector = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm = llm , retriever=vector.as_retriever())
            result = chain({'question':query},)

            st.header("Your Answer")
            st.subheader(result['answer'])







