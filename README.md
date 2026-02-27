# 🌱 Phyto-Research Assistant

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=iffat336/-Phyto-Research-Assistant&branch=main&main_module=app.py)

An AI-powered research assistant designed for agricultural scientists to instantly query PDF research papers using **GPT-4o-mini** and Retrieval-Augmented Generation (RAG).

## 🚀 Features

- **Modern Chat Interface**: Fluid questioning experience using Streamlit's native chat components.
- **Multiple PDF Support**: Upload and process several research papers simultaneously.
- **Smart Chunking & Embeddings**: Uses LangChain's `CharacterTextSplitter` and `FAISS` for efficient semantic search.
- **Persistent Chat History**: Keep track of your conversation and query results with a "Clear History" option in the sidebar.
- **PhD-Ready Aesthetics**: A clean, professional green-themed UI tailored for agricultural research.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iffat336/-Phyto-Research-Assistant.git
   cd -Phyto-Research-Assistant
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up your `.env` file with your OpenAI API Key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## 📈 Usage

Run the following command to start the dashboard:
```bash
.\start_app.bat
```
Or manually:
```bash
streamlit run app.py
```

## 🙏 Special Thanks

Special thanks to **Dr. Muhammad Ammar Tufail** for his invaluable contributions and inspiration to the AI and Agriculture community.

- **GitHub**: [@AammarTufail](https://github.com/AammarTufail)
- **LinkedIn**: [Dr. Muhammad Aammar Tufail](https://www.linkedin.com/in/aammar-tufail/)
- **Community**: [Codanics](https://github.com/codanics)

---
*Built with ❤️ for Agricultural Research.*
