import os
import getpass

from langchain.document_loaders import PyPDFLoader  
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain import HuggingFaceHub  
from langchain.chains import RetrievalQA
import chainlit as cl
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#loading the API key
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']

# path = input("Enter PDF file path: ")
# loader = PyPDFLoader(path)
# pages = loader.load()

# splitter = CharacterTextSplitter(separator='\n',chunk_size=500, chunk_overlap=20)
# docs = splitter.split_documents(pages)

# embeddings = HuggingFaceEmbeddings()
# doc_search = Chroma.from_documents(docs, embeddings)

async def get_pdf_text():
    pdfs = None

    while pdfs == None:
        pdfs = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=3,
            max_files=5
        ).send()

    raw_text = ""
    pages_list = []
    for pdf in pdfs:
        loader = PyPDFLoader(pdf.path)
        pages = loader.load()
        for page in pages:
            text = page.page_content
            raw_text += text
    return raw_text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b", model_kwargs={"temperature":0.5, "max_length":512})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )
    return conversation_chain

@cl.on_chat_start
async def main():
    await cl.Message(content="Uploading and processing PDF...").send()
    raw_text = await get_pdf_text()
    await cl.Message(content="PDF uploaded! Processing...").send()
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    print(vectorstore)
    conversation_chain = get_conversation_chain(vectorstore)
    cl.user_session.set("conversation_chain", conversation_chain)
    await cl.Message(content="PDF processed! You can ask your questions now.").send()

    
@cl.on_message
async def main(message:str):
    conversation_chain = cl.user_session.get("conversation_chain")
    res = await conversation_chain.acall(message.content, callbacks=
                                      [cl.AsyncLangchainCallbackHandler()])
    print(res)
    if "answer" in res:
        answer = res["answer"]
        helpful_answer = answer.split("Helpful Answer:")[1].strip()
        # QuestionIndex = helpful_answer.find("Question")
        # helpful_answer = helpful_answer[0:QuestionIndex]
        await cl.Message(content=helpful_answer).send()
    else:
        await cl.Message(content="Sorry, I couldn't find the required information.").send()
