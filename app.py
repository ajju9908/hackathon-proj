import os
import streamlit as st
import fitz  # PyMuPDF
import faiss
import torch
from langchain_ibm import WatsonxLLM
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# --- 1. Set up API Keys and Environment Variables ---
# Get your IBM Watsonx credentials from your IBM Cloud account.
os.environ["IBM_CLOUD_API_KEY"] = "YOUR_IBM_CLOUD_API_KEY"
os.environ["IBM_WATSONX_PROJECT_ID"] = "YOUR_IBM_WATSONX_PROJECT_ID"
# Hugging Face token is needed for the embedding model if it's not locally available.
os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGING_FACE_HUB_TOKEN"

# --- 2. Streamlit UI Components ---    
st.set_page_config(page_title="StudyMate AI", page_icon="📚")
st.title("📚 StudyMate: Your AI Academic Assistant")
st.markdown("Upload your study materials (PDFs) and ask natural-language questions.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Sidebar for file upload and instructions
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="file_uploader")

    if uploaded_file:
        st.success("PDF uploaded successfully! 🚀")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.info("Chat history cleared. Please upload a new document to begin.")

# --- 3. RAG Pipeline Functions ---

@st.cache_resource
def get_faiss_vector_store(uploaded_file):
    """
    Processes the uploaded PDF, chunks the text, creates embeddings, and builds a FAISS index.
    """
    try:
        # Save the uploaded file to a temporary location
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Use PyMuPDFLoader for efficient PDF text extraction
        loader = PyMuPDFLoader("temp.pdf")
        documents = loader.load_and_split()
        
        # Use a Hugging Face SentenceTransformer model for embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create a FAISS vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def create_qa_chain(vector_store):
    """
    Initializes the ConversationalRetrievalChain with IBM Watsonx's Mixtral model.
    """
    if not vector_store:
        return None

    try:
        # Configure and initialize the IBM Watsonx LLM
        params = {
            "decoding_method": "greedy",
            "max_new_tokens": 512,
            "min_new_tokens": 50,
        }
        
        # Use the Mixtral-8x7B-Instruct model
        llm = WatsonxLLM(
            model_id="mistralai/mixtral-8x7b-instruct-v0-1",
            url="https://us-south.ml.cloud.ibm.com",
            project_id=os.environ["IBM_WATSONX_PROJECT_ID"],
            params=params
        )

        # Create the conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating the Q&A chain. Make sure your IBM Watsonx API key is valid: {e}")
        return None

# --- 4. Main Application Logic ---

if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("Processing PDF and building knowledge base..."):
        faiss_store = get_faiss_vector_store(uploaded_file)
        if faiss_store:
            st.session_state.qa_chain = create_qa_chain(faiss_store)

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the document..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the chain with the user's question and chat history
                    result = st.session_state.qa_chain.invoke(
                        {"question": prompt, "chat_history": st.session_state.chat_history}
                    )
                    answer = result["answer"]
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
    else:
        st.warning("Please upload a PDF file first to start the conversation.")