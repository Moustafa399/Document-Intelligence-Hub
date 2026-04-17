"""Image file extractor with PaddleOCR and VLM fallback."""
import os
import base64
from PIL import Image

from helpers.file_utils import create_document_folder, save_text, save_metadata

# Try importing PaddleOCR
try:
    from paddleocr import PaddleOCR
    # Initialize PaddleOCR globally to avoid reloading the model on every call
    # Use 'ar' (Arabic) model which handles both Arabic and Latin/English scripts
    ocr_model = PaddleOCR(use_angle_cls=True, lang='ar')
except ImportError:
    ocr_model = None
    print("⚠️ Warning: paddleocr not found. Please 'pip install paddleocr'.")

# Try importing Langchain ChatMistralAI for VLM fallback
try:
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import HumanMessage
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    ChatMistralAI = None
    print("⚠️ Warning: langchain_mistralai not found.")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_image(file_path):
    """Extract image file and save to document folder. Runs OCR and falls back to VLM if confidence is low."""
    doc_id, base, text_dir, img_dir = create_document_folder(file_path)

    img = Image.open(file_path)
    ext = os.path.splitext(file_path)[1]
    img_path = os.path.join(img_dir, f"img_1{ext}")
    
    # Save a copy to the image directory
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(img_path)

    description = ""
    source_type = "image"
    vlm_used = False
    avg_confidence = 0.0

    # 1. Try PaddleOCR first
    if ocr_model:
        print(f"🔍 Running PaddleOCR on {file_path}...")
        result = ocr_model.ocr(img_path, cls=True)
        
        texts = []
        confidences = []
        
        if result and result[0]:
            for line in result[0]:
                text, conf = line[1]
                texts.append(text)
                confidences.append(float(conf))
        
        if texts:
            description = "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences)
            print(f"📊 PaddleOCR Average Confidence: {avg_confidence:.2f}")

    # 2. VLM Fallback if confidence is low (< 0.60) or no text found
    if avg_confidence < 0.60 or len(description.strip()) < 5:
        print("📉 OCR confidence low or no text found. Falling back to VLM...")
        vlm_used = True
        
        if ChatMistralAI:
            try:
                llm = ChatMistralAI(
                    model="pixtral-12b-2409", 
                    temperature=0.1,
                    mistral_api_key=os.environ.get("MISTRAL_API_KEY")
                )
                base64_image = encode_image(img_path)
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe the contents of this document or image in detail. Extract any visible text or data."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                )
                response = llm.invoke([message])
                description = response.content if hasattr(response, "content") else str(response)
                print("✅ VLM Analysis complete.")
            except Exception as e:
                print(f"⚠️ VLM fallback failed: {e}")
                description += f"\n[VLM Analysis Failed: {str(e)}]"
        else:
            description = "No readable text found and VLM is not available."

    # Save outputs
    save_text(text_dir, description)
    save_metadata(base, {
        "source": source_type,
        "ocr_confidence": avg_confidence,
        "vlm_used": vlm_used,
        "description_length": len(description)
    })

    return base, [img_path], doc_id, source_type
