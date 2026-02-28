import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback

# Load environment variables
load_dotenv()

def set_page_config():
    st.set_page_config(
        page_title="Phyto-Research Assistant (PDF Summarizer)",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for a beautiful, modern look (matching the PhD vibe)
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #2e7d32;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1b5e20;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        .main-header {
            font-size: 2.5rem;
            color: #2e7d32;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #555;
            text-align: center;
            margin-bottom: 3rem;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    set_page_config()

    st.markdown('<p class="main-header">🌱 Phyto-Research Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your agricultural research papers and instantly query their contents using GPT-4o-mini.</p>', unsafe_allow_html=True)

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar for Settings and Upload
    with st.sidebar:
        st.header("Settings")
        
        # Check for API Key (Support both local .env and Streamlit Secrets)
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        has_api_key = False
        
        if not api_key or api_key == "your_api_key_here":
            st.error("❌ OpenAI API Key is missing.")
            st.info("💡 **Local:** Add it to your `.env` file.\n\n💡 **Cloud:** Add it to your Streamlit 'Secrets'.")
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✅ API Key connected.")
            has_api_key = True
        
        st.divider()
        st.header("Upload Document")
        pdf_docs = st.file_uploader(
            "Upload your PDF Research Papers", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        process_button = st.button("Process Documents", use_container_width=True, disabled=not has_api_key)
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.vectorstore = None
            st.rerun()

    if pdf_docs and process_button:
        if not has_api_key:
            st.error("Please provide a valid OpenAI API Key in Settings.")
        else:
            with st.spinner("Processing your research papers..."):
                try:
                    # 1. Extract text from PDFs
                    raw_text = ""
                    for pdf in pdf_docs:
                        try:
                            pdf_reader = PdfReader(pdf)
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    raw_text += text + "\n"
                        except Exception as pdf_err:
                            st.warning(f"⚠️ Could not read {pdf.name}: {pdf_err}")
                    
                    if not raw_text.strip():
                        st.error("❌ Could not extract text from the provided PDFs. They might be empty, encrypted, or scanned images.")
                        return

                    # 2. Split text into chunks
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(raw_text)

                    # 3. Create vector store embeddings
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                    st.session_state.vectorstore = vectorstore
                    
                    st.success("✅ Documents processed successfully! Ask your questions below.")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during processing: {e}")
                    return

    # Display Chat History using chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if user_question := st.chat_input("Ask a question about your papers..."):
        if not has_api_key:
            st.warning("⚠️ Please configure your OpenAI API Key first.")
        elif not st.session_state.vectorstore:
            st.warning("⚠️ Please upload and process your documents first.")
        else:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing papers..."):
                    try:
                        docs = st.session_state.vectorstore.similarity_search(user_question, k=4)
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                        prompt = ChatPromptTemplate.from_template(
                            "Answer the following question based only on the provided context.\n\nContext:\n{context}\n\nQuestion: {question}"
                        )
                        chain = prompt | llm
                        
                        response = chain.invoke({"context": context_text, "question": user_question})
                        answer = response.content
                        
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")

if __name__ == '__main__':
    main()
