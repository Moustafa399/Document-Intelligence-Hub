"""Centralized configuration for the Document Extractor."""

# LLM Settings
LLM_MODEL = "mistral-medium-latest"
LLM_TEMPERATURE = 0.2

# OCR Settings
OCR_ENABLED = True
OCR_LANGUAGES = ["en", "ar"]  # Languages for EasyOCR
OCR_GPU = False  # Set to True if you have a CUDA-compatible GPU

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "extractor.log"

# Pipeline Settings
MAX_TEXT_LENGTH = 50  # Max chars to consider "empty" content needing OCR

# RAG Settings
ROWS_PER_CHUNK = 20        # Number of rows per chunk for Excel/CSV (change as you like)
TEXT_CHUNK_SIZE = 1000     # Characters per chunk for text content (PDF/Word/PPTX)
TEXT_CHUNK_OVERLAP = 200   # Overlap between text chunks
OCR_CONFIDENCE_THRESHOLD = 0.60  # Below this → fallback to VLM
