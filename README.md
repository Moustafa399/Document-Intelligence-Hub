# 📚 Document Intelligence Hub (Multi-Modal RAG Extractor)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Mistral](https://img.shields.io/badge/Mistral_AI-000000?style=for-the-badge&logo=mistral)

**Document Intelligence Hub** is a powerful Multi-Modal Retrieval-Augmented Generation (RAG) pipeline and application. It allows you to upload various types of documents, automatically extracts text and data (including OCR for images and scanned PDFs in both **English and Arabic**), and provides an interactive AI chat interface to query your documents instantly.

---

## 🖼️ Application Screenshots

<div align="center">
  <img src="imgs%20production/knowledge1.png" alt="Screenshot 1" width="45%" />
  <img src="imgs%20production/knowledge2.png" alt="Screenshot 2" width="45%" />
</div>

---

## ✨ Key Features

- **Multi-Format Support**: Ingest PDF, Word (`.docx`), Excel (`.xlsx`), CSV, PowerPoint (`.pptx`), and Images (`.png`, `.jpg`).
- **Advanced OCR**: Built-in Optical Character Recognition supporting **English and Arabic**, falling back to Vision-Language Models (VLM) for complex layouts.
- **Intelligent RAG Pipeline**: Chunks documents dynamically based on file type and uses advanced embeddings (Chroma/LangChain) for highly accurate semantic retrieval.
- **Interactive Chat UI**: Beautiful, responsive Streamlit frontend for uploading files and conversing with your knowledge base.
- **FastAPI Backend**: Robust API architecture for extraction, embedding, and querying.
- **Session Management**: Isolate document collections by user sessions to prevent data crossover.

---

## 🛠️ Tech Stack

- **Backend Framework**: FastAPI, Uvicorn
- **Frontend Interface**: Streamlit
- **LLM Integration**: Mistral (`mistral-medium-latest`)
- **Document Processing**: PyMuPDF, LangChain
- **OCR**: PaddleOCR / EasyOCR
- **Vector Storage**: ChromaDB
- **Database**: MongoDB (for document metadata tracking)

---

## 🚀 Getting Started

### 1. Clone the repository (if applicable)
```bash
git clone https://github.com/yourusername/Extractor_files.git
cd Extractor_files
```

### 2. Set up the Environment
Create a virtual environment and install the dependencies:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
MONGO_URI=your_mongodb_connection_string
```

### 4. Run the Application

You need to run both the FastAPI backend and the Streamlit frontend.

**Start the FastAPI Backend:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8007 --reload
```

**Start the Streamlit Frontend:**
Open a new terminal, activate your environment, and run:
```bash
streamlit run frontend.py
```

Now, open your browser and navigate to `http://localhost:8501` to access the Document Intelligence Hub!

---

## ⚙️ Configuration

You can tweak the pipeline settings in `config.py`:
- `LLM_MODEL`: Change the active AI model.
- `OCR_LANGUAGES`: Define languages for OCR (e.g., `["en", "ar"]`).
- `CHUNK_SIZE` & `CHUNK_OVERLAP`: Optimize retrieval granularity.
- `OCR_CONFIDENCE_THRESHOLD`: Set the threshold for VLM fallback.

---

## 🔌 API Endpoints

If you wish to use the API directly without the UI:

- `POST /extract`: Upload files to process and ingest them into the RAG system.
- `POST /chat`: Query your ingested documents using a `session_id`.
- `GET /health`: Health check endpoint to ensure the service is running.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is licensed under the MIT License.
