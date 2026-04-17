import os
import json
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration from .env
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "document_extractor")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "extractions")

def get_db_collection():
    """Connect to MongoDB and return the target collection."""
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        db = client[MONGO_DB]
        return db[MONGO_COLLECTION]
    except Exception as e:
        print(f"⚠️ Failed to connect to MongoDB: {e}")
        return None

def save_batch_to_mongodb(json_paths, session_id, author):
    """
    Reads all the extracted structured JSON files from a batch
    and saves them as a single comprehensive record in MongoDB.
    """
    collection = get_db_collection()
    if collection is None:
        print("⚠️ MongoDB connection not available. Skipping DB save.")
        return None

    documents_data = []
    
    # Read each structured.json from the batch
    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents_data.append(data)
        except Exception as e:
            print(f"⚠️ Error reading JSON from {path}: {e}")
            
    if not documents_data:
        print("⚠️ No valid data found in batch to save to MongoDB.")
        return None

    # Bundle into a single record
    batch_record = {
        "session_id": session_id,
        "author": author,
        "upload_date": datetime.utcnow(),
        "total_documents": len(documents_data),
        "documents": documents_data
    }

    try:
        result = collection.insert_one(batch_record)
        print(f"💾 Successfully saved batch to MongoDB (ID: {result.inserted_id})")
        return str(result.inserted_id)
    except Exception as e:
        print(f"❌ Failed to insert record into MongoDB: {e}")
        return None
