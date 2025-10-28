import os
import getpass

from langchain.document_loaders import PyPDFLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma 
from langchain import HuggingFaceHub  
from langchain.chains import RetrievalQA
import chainlit as cl

#loading the API key
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
# print(huggingfacehub_api_token)
path = input("Enter PDF file path: ")
loader = PyPDFLoader(path)
pages = loader.load()

splitter = CharacterTextSplitter(separator='\n',chunk_size=500, chunk_overlap=20)
docs = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(docs, embeddings)

repo_id = "tiiuae/falcon-7b"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={'temperature': 0.2, 'max_length':1000}) 

@cl.on_chat_start
def main():
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever())
    cl.user_session.set("retrieval_chain", retrieval_chain)
    
@cl.on_message
async def main(message:str):
    retrieval_chain = cl.user_session.get("retrieval_chain")
    res = await retrieval_chain.acall(message.content, callbacks=
                                      [cl.AsyncLangchainCallbackHandler()])
    print(res)
    if "result" in res:
        answer = res["result"].split("Helpful Answer:")[1].strip() if "Helpful Answer:" in res["result"] else res["result"]
        await cl.Message(content=answer).send()
    else:
        await cl.Message(content="Sorry, I couldn't find the required information.").send()