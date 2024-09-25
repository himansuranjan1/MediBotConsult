import streamlit as st
from streamlit_chat import message 
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Load the PDF files from the path
# Use raw string for Windows paths or escape backslashes
#loader = DirectoryLoader('D:\rag_try\temp.pdf', glob="*.pdf", loader_cls=PyPDFLoader)
loader = DirectoryLoader(r'D:\rag_try', glob="*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})


# Create FAISS vectorstore
vector_store = FAISS.from_documents(text_chunks, embeddings)

# Create LLM using CTransformers
#llm = CTransformers(model="model-00002-of-00002.safetensors", model_type="llama",
                    #config={'max_new_tokens': 128, 'temperature': 0.01})
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q5_K_S.bin", model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})


# Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the retrieval chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm, chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory
)

# Set up Streamlit interface
st.title("HealthCare ChatBot 🧑🏽‍⚕️")

# Conversation handling function
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Initialize session state for chat history
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

# Display chat history and handle user input
def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Health", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state and display chat history
initialize_session_state()
display_chat_history()
