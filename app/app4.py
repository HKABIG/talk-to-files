from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sentence_transformers
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template, source_template 
import PyPDF2
from io import BytesIO
from langchain.prompts import PromptTemplate

custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content

def get_sources(docs):
    source_list = set()
    for doc in docs:
        source_name = doc.metadata['source']
        page_number = doc.metadata['page']
        source_list.add(f"[ source_file, {source_name} : page_number, {page_number} ]")
    return " ".join(source_list)

def document_loader(docs):
    documents = []
    for file in docs:
        pdf_file = file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            documents.append(Document(metadata={'source': file.name, 'page': page_num}, page_content=text))
    print(documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(chunks)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    print(db)
    return db

def get_conversationchain(vectorstore):
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.2, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer') # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                memory=memory)
    return conversation_chain

def handle_question(question):
    response=st.session_state.conversation({'question': question})
    st.session_state.chat_history=response["chat_history"]
    for i,msg in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",msg.content,),unsafe_allow_html=True)
        else:
            response = msg.content[msg.content.rfind("Helpful Answer:")+15:]
            docs = st.session_state.vectorstore.similarity_search(response,k=2)
            sources = get_sources(docs)
            print(msg.content)
            st.write(bot_template.replace("{{MSG}}",response),unsafe_allow_html=True)
            st.write(source_template.replace("{{SOURCES}}",sources),unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with multiple PDFs :books:")
    question=st.text_input("Ask question from your document:")
    if question:
        handle_question(question)
    with st.sidebar:
        st.subheader("Your documents")
        docs=st.file_uploader("Upload your PDF here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #create vectorstore
                st.session_state.vectorstore = document_loader(docs)

                #create conversation chain
                st.session_state.conversation=get_conversationchain(st.session_state.vectorstore)


if __name__ == '__main__':
    main()