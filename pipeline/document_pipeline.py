"""
Document Processing Pipeline.
Orchestrates: Extraction → LLM Analysis → Structured Output → RAG Ingestion
"""
import os
import re
import json

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

from extractors import extract_pdf, extract_word, extract_excel, extract_csv, extract_ppt, extract_image
from config import LLM_MODEL, LLM_TEMPERATURE

load_dotenv()

# Try importing RAG service
try:
    from services.rag_service import ingest_to_rag
except ImportError:
    ingest_to_rag = None
    print("⚠️ RAG service not available. Ingestion will be skipped.")


# =====================================================
# LLM Agent (Mistral)
# =====================================================
PARSING_PROMPT = """
Extract information from this document and return ONLY a JSON object.

Rules:
1. Detect the language from the text (english, arabic, mixed, etc.)
2. Find the author if clearly stated. Look for BOTH English and Arabic author markers:
   - English: "Prepared by:", "Author:", "Created by:", "Written by:"
   - Arabic: "إعداد:", "المؤلف:", "بقلم:", "تأليف:", "كتابة:"
3. Write a brief but informative summary (1-3 sentences).
   - If the document is in Arabic, write the summary in Arabic.
   - If the document is in English, write the summary in English.
   - If mixed, write the summary in the dominant language.

JSON format:
{{
  "language": "detected language here",
  "author": "author name or empty string",
  "summary": "informative summary describing the content (in the document's language)"
}}

Document:
{TEXT}

Return only the JSON:"""


def extract_json(text: str):
    """Extract JSON from LLM response."""
    markdown_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if markdown_match:
        return markdown_match.group(1)
    
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        raise ValueError("No JSON found in LLM response")
    return match.group(0).strip()


def run_llm_analysis(text_preview):
    """Get language, author, and summary from Mistral LLM."""
    try:
        llm = ChatMistralAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            mistral_api_key=os.environ.get("MISTRAL_API_KEY")
        )
        response = llm.invoke(PARSING_PROMPT.format(TEXT=text_preview[:3500]))
        response_text = response.content if hasattr(response, "content") else response
        
        json_str = extract_json(response_text)
        parsed = json.loads(json_str)
        
        return {
            "language": parsed.get("language", "unknown"),
            "author": parsed.get("author", ""),
            "summary": parsed.get("summary", "Document processed successfully")
        }
    except Exception as e:
        print(f"⚠️ LLM parsing failed: {e}")
        return {
            "language": "unknown",
            "author": "",
            "summary": "Document processed successfully"
        }


# =====================================================
# Structured Output Builder
# =====================================================
def build_structured_output(base_dir, source, source_id, text_content, tables_data, author=None, session_id=None):
    """Build final structured.json with all data."""
    
    print("🤖 Running Mistral LLM analysis...")
    llm_result = run_llm_analysis(text_content[:3500] if text_content else "")
    
    result = {
        "session_id": session_id,
        "source": source,
        "source_id": source_id,
        "language": llm_result["language"],
        "author": author or llm_result["author"],
        "summary": llm_result["summary"],
        "content": text_content if text_content else "",
        "tables_count": len(tables_data) if tables_data else 0
    }
    
    # Add tables for PDF/Word/PPTX
    if source in ("pdf", "word", "powerpoint") and tables_data:
        result["tables"] = tables_data
    
    # Add Excel sheets
    if source == "excel" and tables_data:
        result["excel"] = {"sheets": tables_data}
    
    # Add CSV data
    if source == "csv" and tables_data:
        result["csv"] = tables_data[0] if tables_data else {}
    
    # Save structured.json
    output_path = os.path.join(base_dir, "structured.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("✅ LLM analysis complete")
    return output_path


# =====================================================
# Main Pipeline
# =====================================================
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "word",
    ".doc": "word",
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".csv": "csv",
    ".pptx": "powerpoint",
    ".ppt": "powerpoint",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".webp": "image",
}


def pipeline(file_path, author=None, use_ocr_vlm=True, save_to_mongo=False, session_id=None):
    """
    Full document processing pipeline.
    
    Steps:
        1. Extract content based on file type (text, tables, images)
        2. Run LLM analysis (language, author, summary)
        3. Build structured.json
        4. Ingest into RAG vector database
    
    Returns:
        tuple: (base_dir, structured_json_path)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    
    file_type = SUPPORTED_EXTENSIONS[ext]
    print(f"\n📄 Processing {ext} file as '{file_type}'...")
    
    # Step 1: Extract content
    if file_type == "pdf":
        base_dir, images, doc_id, source = extract_pdf(file_path)
    elif file_type == "word":
        base_dir, images, doc_id, source = extract_word(file_path)
    elif file_type == "excel":
        base_dir, images, doc_id, source = extract_excel(file_path)
    elif file_type == "csv":
        base_dir, images, doc_id, source = extract_csv(file_path)
    elif file_type == "powerpoint":
        base_dir, images, doc_id, source = extract_ppt(file_path)
    elif file_type == "image":
        base_dir, images, doc_id, source = extract_image(file_path)
    
    # Read extracted text
    text_path = os.path.join(base_dir, "text", "content.txt")
    text_content = ""
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            text_content = f.read()
    
    # Read extracted tables
    tables_path = os.path.join(base_dir, "tables", "tables.json")
    tables_data = []
    if os.path.exists(tables_path):
        with open(tables_path, "r", encoding="utf-8") as f:
            tables_data = json.load(f)
    
    # Step 2 & 3: LLM Analysis + Build structured output
    output_path = build_structured_output(base_dir, source, doc_id, text_content, tables_data, author, session_id)
    
    # Step 4: RAG Ingestion
    if ingest_to_rag:
        try:
            ingest_to_rag(output_path)
        except Exception as e:
            print(f"⚠️ RAG Ingestion Failed: {e}")
    
    print(f"✅ Pipeline complete for: {os.path.basename(file_path)}")
    return base_dir, output_path


# =====================================================
# CLI Entry Point
# =====================================================
if __name__ == "__main__":
    print("\n📂 Enter file path:")
    path = input(">> ").strip().strip('"')
    
    if not os.path.exists(path):
        print("❌ File not found")
        exit(1)
    
    try:
        base, output = pipeline(path)
        
        print("\n✅ DONE")
        print(f"📁 Output folder: {base}")
        print(f"📊 Structured data: {output}")
        
        with open(output, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"\n📋 Summary:")
            print(f"   Source: {data['source']}")
            print(f"   Language: {data['language']}")
            print(f"   Tables: {data.get('tables_count', 0)}")
            print(f"   Summary: {data['summary'][:100]}...")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
