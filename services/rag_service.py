import os
import json
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from config import ROWS_PER_CHUNK, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP

load_dotenv()

# Setup Vector Store
CHROMA_PATH = os.path.join(os.getcwd(), "temp", "chromadb")
COLLECTION_NAME = "documents_rag"

def get_embeddings():
    return MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.environ.get("MISTRAL_API_KEY")
    )

def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH
    )


def _calculate_optimal_chunk_size(rows, num_columns):
    """
    Dynamically calculate the best number of rows per chunk
    based on the actual data size. Goal: keep each chunk 
    under ~1500 characters so embeddings stay meaningful.
    """
    if not rows:
        return ROWS_PER_CHUNK  # fallback to config default
    
    # Sample a few rows to estimate average row length
    sample = rows[:min(10, len(rows))]
    avg_row_len = sum(
        len(" | ".join(str(cell) for cell in row))
        for row in sample
    ) / len(sample)
    
    # Target: ~1500 chars per chunk (sweet spot for embeddings)
    TARGET_CHUNK_CHARS = 1500
    
    if avg_row_len > 0:
        optimal = max(5, int(TARGET_CHUNK_CHARS / avg_row_len))  # minimum 5 rows
    else:
        optimal = ROWS_PER_CHUNK
    
    # Cap it: never go above config value, never below 5
    optimal = min(optimal, ROWS_PER_CHUNK)
    
    total = len(rows)
    print(f"   📐 Adaptive chunking: {total} rows, ~{avg_row_len:.0f} chars/row → {optimal} rows/chunk ({(total // optimal) + 1} chunks)")
    return optimal


def _row_based_chunks(rows, header_line, doc_id, source, data_type, location, session_id):
    """
    Split tabular rows into adaptive-sized chunks.
    Each chunk always starts with the header line so the LLM
    understands what every column represents.
    Chunk size is calculated dynamically based on actual data.
    """
    if not rows:
        return []
    
    num_columns = len(rows[0]) if rows else 0
    chunk_size = _calculate_optimal_chunk_size(rows, num_columns)
    
    chunks = []
    total_rows = len(rows)
    
    for start in range(0, total_rows, chunk_size):
        batch = rows[start:start + chunk_size]
        rows_text = "\n".join(
            " | ".join(str(cell) for cell in row) for row in batch
        )
        chunk_text = f"{header_line}\nRows {start + 1}-{start + len(batch)} of {total_rows}:\n{rows_text}"
        
        doc = Document(
            page_content=chunk_text,
            metadata={
                "session_id": session_id,
                "doc_id": doc_id,
                "source": source,
                "type": data_type,
                "location": location,
                "row_start": start + 1,
                "row_end": start + len(batch),
                "total_rows": total_rows,
                "chunk_index": start // chunk_size
            }
        )
        chunks.append(doc)
    
    return chunks


def chunk_structured_json(json_path):
    """Reads structured.json and chunks its contents for RAG."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    doc_id = data.get('source_id', 'unknown')
    source = data.get('source', 'unknown')
    session_id = data.get('session_id', 'unknown')
    
    docs_to_embed = []
    
    # 1. Chunk Text content
    if 'content' in data and data['content']:
        text = data['content']
        splitter = RecursiveCharacterTextSplitter(chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP)
        chunks = splitter.split_text(text)
        
        for idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={"session_id": session_id, "doc_id": doc_id, "source": source, "type": "text", "chunk_index": idx}
            )
            docs_to_embed.append(doc)
            
    # 2. Chunk Tables (PDF/Word/PPTX)
    if 'tables' in data:
        for t_idx, table in enumerate(data['tables']):
            location = table.get('location', f"Table {t_idx}")
            table_text = f"Table Location: {location}\nHeaders: {', '.join(str(h) for h in table.get('headers', []))}\n"
            for row in table.get('data', []):
                table_text += f"{', '.join(str(cell) for cell in row)}\n"
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP)
            chunks = splitter.split_text(table_text)
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"session_id": session_id, "doc_id": doc_id, "source": source, "type": "table", "location": location, "chunk_index": idx}
                )
                docs_to_embed.append(doc)
                
    # 3. Chunk Excel Sheets (Row-Based Chunking)
    if 'excel' in data and 'sheets' in data['excel']:
        for sheet in data['excel']['sheets']:
            sheet_name = sheet.get('sheet_name', 'Unknown Sheet')
            headers = sheet.get('headers', [])
            rows = sheet.get('data', [])
            header_line = f"Excel Sheet: {sheet_name}\nHeaders: {' | '.join(str(h) for h in headers)}"
            
            docs_to_embed.extend(
                _row_based_chunks(rows, header_line, doc_id, source, "excel_sheet", sheet_name, session_id)
            )
                
    # 4. Chunk CSV (Row-Based Chunking)
    if 'csv' in data:
        csv_info = data['csv']
        headers = csv_info.get('headers', [])
        rows = csv_info.get('data', [])
        header_line = f"CSV File: {csv_info.get('file_name', 'unknown')}\nHeaders: {' | '.join(str(h) for h in headers)}"
        
        docs_to_embed.extend(
            _row_based_chunks(rows, header_line, doc_id, source, "csv", csv_info.get('file_name', 'unknown'), session_id)
        )

    return docs_to_embed

def ingest_to_rag(json_path):
    """Full ingestion pipeline: Chunk -> Embed -> Store"""
    print(f"🧠 RAG Ingestion starting for {json_path}")
    chunks = chunk_structured_json(json_path)
    if not chunks:
        print("⚠️ No content found to ingest.")
        return False
        
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"✅ Successfully ingested {len(chunks)} chunks into ChromaDB.")
    return True

def query_rag(question: str, session_id: str, top_k: int = 5):
    """Retrieve relevant chunks and answer question using Mistral"""
    vector_store = get_vector_store()
    
    # Retrieve top K matches, filtering by session_id to isolate context
    search_kwargs = {"k": top_k}
    if session_id:
        search_kwargs["filter"] = {"session_id": session_id}
        
    results = vector_store.similarity_search(question, **search_kwargs)
    
    # Build Context
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results]) if results else "No context available (no relevant documents found)."
    
    # Prompt LLM
    prompt_template = """
    You are a friendly and professional AI Document Assistant. 
    Your personality: Warm, helpful, thorough, detailed, and comprehensive.
    
    IMPORTANT RULE FOR LANGUAGE:
    If the Question is written in English, your Answer MUST be entirely in English.
    If the Question is written in Arabic, your Answer MUST be entirely in Arabic.
    You must dynamically match the language of the user's question.

    Rules:
    1. If the user greets you (like saying "Hello" or "مرحبا") or asks who you are, introduce yourself warmly as their smart Document Assistant in the exact same language they used.
    2. For all other questions, use the provided Context below to answer.
    3. If the answer is not in the context, politely inform the user that you don't have this information in the uploaded documents.
    4. Do not make up facts.
    5. Provide VERY DETAILED and COMPREHENSIVE answers. Explain the information thoroughly and elaborate on points found in the context. Avoid short or brief summaries unless explicitly asked.

    Context:
    {context}
    
    Question: {question}
    
    Answer (in the exact same language as the Question):"""
    
    llm = ChatMistralAI(
        model="mistral-medium-latest", 
        temperature=0.1,
        mistral_api_key=os.environ.get("MISTRAL_API_KEY")
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | llm
    
    response = chain.invoke({"context": context_text, "question": question})
    return response.content if hasattr(response, 'content') else str(response)
