import streamlit as st
import os
import io
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import easyocr
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback

# Load environment variables
load_dotenv()

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'])

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
    st.markdown('<p class="sub-header">Upload your agricultural research papers (even scanned ones!) and instantly query their contents.</p>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "matrix_data" not in st.session_state:
        st.session_state.matrix_data = []

    # Sidebar for Settings and Upload
    with st.sidebar:
        st.header("Settings")
        
        # 1. Try to get key from secrets or env
        env_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        if env_key == "your_api_key_here":
            env_key = None
            
        # 2. Provide a fallback input field in the UI
        api_key_input = st.text_input(
            "Enter OpenAI API Key", 
            value=env_key if env_key else "", 
            type="password",
            help="Your key is stored only for this session."
        ).strip()
        
        has_api_key = False
        if api_key_input:
            if not api_key_input.startswith("sk-"):
                st.error("❌ Invalid Key Format: OpenAI keys usually start with 'sk-'.")
            else:
                os.environ["OPENAI_API_KEY"] = api_key_input
                st.success("✅ API Key connected.")
                has_api_key = True
                
                # Test Connection Button
                if st.button("Test API Connection", use_container_width=True):
                    with st.spinner("Testing connection..."):
                        try:
                            # Use a minimal request to test the key
                            test_llm = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=5)
                            test_llm.invoke("Hi")
                            st.success("🎉 Connection Successful!")
                        except Exception as test_err:
                            st.error(f"❌ Connection Failed: {test_err}")
                            if "insufficient_quota" in str(test_err).lower():
                                st.info("💡 **Tip:** Your API key might have run out of credit or quota.")
        else:
            st.error("❌ OpenAI API Key is missing.")
            st.info("💡 Paste your key above or add it to Streamlit Secrets.")
        
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
                    # Initialize OCR Reader
                    reader = load_ocr_reader()
                    
                    all_docs = []
                    matrix_data = [] # To store metadata for the Comparison Table

                    from langchain.schema import Document
                    
                    for pdf in pdf_docs:
                        pdf_bytes = pdf.read()
                        file_text = ""
                        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
                        
                        # AI Metadata extraction for the Matrix
                        llm_metadata = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                        metadata_prompt = ChatPromptTemplate.from_template(
                            "Extract the 'Research Methodology' and 'Main Key Findings' from the following text in 10-15 words each.\n\n"
                            "Text: {text}\n\n"
                            "Return format: Methodology: [Value] | Findings: [Value]"
                        )
                        meta_text = first_page_text if first_page_text.strip() else "No text extracted"
                        try:
                            meta_resp = (metadata_prompt | llm_metadata).invoke({"text": meta_text[:1000]})
                            meta_content = meta_resp.content
                            methodology = meta_content.split("|")[0].replace("Methodology:", "").strip()
                            findings = meta_content.split("|")[1].replace("Findings:", "").strip()
                        except:
                            methodology = "TBD"
                            findings = "TBD"

                        matrix_entry = {
                            "Source": pdf.name,
                            "Title": pdf.name.replace(".pdf", "").replace("_", " "),
                            "Methodology": methodology,
                            "Key Findings": findings,
                            "Pages": len(pdf_reader.pages)
                        }

                        for page_num, page in enumerate(pdf_reader.pages):
                            page_text = page.extract_text()
                            
                            # Fallback to OCR if page text is empty
                            if not page_text.strip():
                                images = convert_from_bytes(pdf_bytes, first_page=page_num+1, last_page=page_num+1)
                                for image in images:
                                    img_array = np.array(image)
                                    ocr_results = reader.readtext(img_array, detail=0)
                                    page_text = "\n".join(ocr_results)
                            
                            if page_text.strip():
                                all_docs.append(Document(
                                    page_content=page_text,
                                    metadata={"source": pdf.name, "page": page_num + 1}
                                ))
                        
                        matrix_data.append(matrix_entry)
                    
                    if not all_docs:
                        st.error("❌ Could not extract text from the provided PDFs.")
                        return

                    # 2. Split documents into chunks
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_documents(all_docs)

                    # 3. Create vector store embeddings
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
                    
                    # Update session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.matrix_data = matrix_data
                    
                    st.success(f"✅ {len(pdf_docs)} documents processed successfully! {len(chunks)} chunks indexed with page tracking.")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during processing: {e}")
                    return

    # Main Interface Layout with Tabs
    chat_tab, matrix_tab, tools_tab = st.tabs(["💬 Research Chat", "📊 Literature Matrix", "🎓 Educator Tools"])

    with chat_tab:
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
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing papers..."):
                        try:
                            docs = st.session_state.vectorstore.similarity_search(user_question, k=4)
                            
                            # Build context with citation info
                            context_chunks = []
                            for doc in docs:
                                source = doc.metadata.get('source', 'Unknown')
                                page = doc.metadata.get('page', '?')
                                context_chunks.append(f"--- [SOURCE: {source}, PAGE: {page}] ---\n{doc.page_content}")
                            
                            context_text = "\n\n".join(context_chunks)
                            
                            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                            prompt = ChatPromptTemplate.from_template(
                                "You are an expert academic researcher. Answer the following question based ONLY on the provided context.\n"
                                "When you use information from a source, YOU MUST cite it at the end of the sentence or paragraph like this: (Source: Filename, Page X).\n\n"
                                "Context:\n{context}\n\n"
                                "Question: {question}"
                            )
                            chain = prompt | llm
                            
                            response = chain.invoke({"context": context_text, "question": user_question})
                            answer = response.content
                            
                            st.markdown(answer)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {e}")

    with matrix_tab:
        st.header("📊 Literature Review Matrix")
        if not st.session_state.matrix_data:
            st.info("Upload documents to generate the comparison matrix.")
        else:
            import pandas as pd
            df = pd.DataFrame(st.session_state.matrix_data)
            st.table(df)
            st.caption("Auto-generated based on uploaded research papers.")

    with tools_tab:
        st.header("🎓 Educator Tools")
        if not st.session_state.vectorstore:
            st.info("Upload documents to enable assessment and analysis tools.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📝 Generate Student Quiz", use_container_width=True):
                    with st.spinner("Preparing quiz..."):
                        # Get some diverse context for the quiz
                        quiz_docs = st.session_state.vectorstore.similarity_search("core concepts and definitions", k=5)
                        quiz_context = "\n".join([d.page_content for d in quiz_docs])
                        
                        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
                        quiz_prompt = ChatPromptTemplate.from_template(
                            "Create 3 Multiple Choice Questions (MCQs) for university students based on this research:\n\n{context}\n\n"
                            "Provide the correct Answer Key at the end."
                        )
                        quiz_resp = (quiz_prompt | llm).invoke({"context": quiz_context})
                        st.markdown("### 📝 Student Assessment")
                        st.markdown(quiz_resp.content)
            
            with col2:
                if st.button("🔍 Find Research Gaps", use_container_width=True):
                    with st.spinner("Analyzing gaps..."):
                        # Query specifically for limitations and gaps
                        gap_docs = st.session_state.vectorstore.similarity_search("limitations future research goals gaps unanswered questions", k=5)
                        gap_context = "\n".join([d.page_content for d in gap_docs])
                        
                        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                        gap_prompt = ChatPromptTemplate.from_template(
                            "Based on the following research documents, identify 3 'Research Gaps' or opportunities for future studies. "
                            "Suggest these as potential PhD or Master's thesis topics.\n\n{context}"
                        )
                        gap_resp = (gap_prompt | llm).invoke({"context": gap_context})
                        st.markdown("### 🔍 Future Research Opportunities")
                        st.markdown(gap_resp.content)

            st.divider()
            if st.button("📚 Export References (APA/BibTeX)", use_container_width=True):
                with st.spinner("Formatting references..."):
                    llm = ChatOpenAI(model_name="gpt-4o-mini")
                    refs = []
                    for entry in st.session_state.matrix_data:
                        ref_prompt = f"Create a standard APA citation and a BibTeX entry for a paper titled '{entry['Title']}'. If you can guess more info from common phyto-research context, do so, else use placeholders."
                        ref_resp = llm.invoke(ref_prompt)
                        refs.append(f"### {entry['Source']}\n{ref_resp.content}")
                    
                    st.markdown("## 📑 Academic References")
                    st.markdown("\n\n".join(refs))
                    st.download_button("Download References", "\n\n".join(refs), file_name="references.txt")

if __name__ == '__main__':
    main()
