import os
import shutil
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

# Import your pipeline
from pipeline.document_pipeline import pipeline

app = FastAPI(title="Document Extractor & RAG API")

# Setup directories
UPLOAD_DIR = "temp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/extract")
def extract_documents(
    files: List[UploadFile] = File(...),
    author: str = Form("Default Author"), 
    session_id: str = Form(None)
):
    """
    Upload multiple documents, extract content, and ingest into RAG.
    After extraction, documents are automatically searchable via /chat.
    """
    documents_status = []
    json_paths = []
    
    if not session_id:
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
    
    for file in files:
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{unique_id}_{file.filename}"
        temp_path = os.path.join(UPLOAD_DIR, filename)
        
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            print(f"🚀 Processing: {file.filename} (Session: {session_id})")

            # Execute Pipeline: Extract → LLM Analysis → RAG Ingestion
            base_dir, parsed_path = pipeline(
                temp_path, 
                author=author,
                session_id=session_id
            )
            
            json_paths.append(parsed_path)
            documents_status.append({
                "filename": file.filename,
                "status": "success",
                "structured_json": parsed_path
            })

        except Exception as e:
            print(f"❌ Error processing {file.filename}: {e}")
            documents_status.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 3. Batch Save to MongoDB
    batch_mongo_id = None
    if json_paths:
        from services.db_service import save_batch_to_mongodb
        batch_mongo_id = save_batch_to_mongodb(json_paths, session_id, author)

    return {
        "session_id": session_id,
        "batch_mongo_id": batch_mongo_id,
        "processed_count": sum(1 for d in documents_status if d["status"] == "success"),
        "documents": documents_status
    }


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: str

@app.post("/chat")
def chat_with_documents(request: ChatRequest):
    """
    Ask a question against all ingested documents using the RAG system.
    """
    try:
        from services.rag_service import query_rag
        answer = query_rag(request.query, request.session_id, request.top_k)
        return {"query": request.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "Document Extractor & RAG"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
