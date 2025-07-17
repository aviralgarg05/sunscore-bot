import pandas as pd
from pymongo import MongoClient
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

MONGODB_URI="mongodb+srv://dataxheimat:RxKjoqvpz2yUoU92@data.fq8ofzs.mongodb.net/?retryWrites=true&w=majority&appName=Data"
MONGODB_DATABASE="SunScore"
MONGODB_COLLECTION="SolarData"
OUTPUT_FILE = "solar_data.csv"

def export_optimized(batch_size=10000):
    """Optimized export for massive datasets"""
    print("Starting optimized MongoDB export for massive datasets...")
    start_time = time.time()
    
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DATABASE]
    collection = db[MONGODB_COLLECTION]
    
    # Get total count for progress tracking
    total_docs = collection.count_documents({})
    print(f"Total documents to export: {total_docs:,}")
    
    # Stream data in batches for memory efficiency
    all_batches = []
    processed = 0
    
    # Use cursor with batch processing
    cursor = collection.find().batch_size(batch_size)
    
    current_batch = []
    for doc in cursor:
        # Convert ObjectId to string efficiently
        doc['_id'] = str(doc['_id'])
        current_batch.append(doc)
        
        if len(current_batch) >= batch_size:
            all_batches.append(pd.DataFrame(current_batch))
            processed += len(current_batch)
            current_batch = []
            
            # Progress update
            progress = (processed / total_docs) * 100
            print(f"Progress: {progress:.1f}% ({processed:,}/{total_docs:,} documents)")
    
    # Handle remaining documents
    if current_batch:
        all_batches.append(pd.DataFrame(current_batch))
        processed += len(current_batch)
    
    # Combine all batches efficiently
    print("Combining batches...")
    df = pd.concat(all_batches, ignore_index=True)
    
    # Optimized CSV export with compression
    print(f"Exporting {len(df):,} documents to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, compression='infer')
    
    client.close()
    
    export_time = time.time() - start_time
    print(f"Optimized export completed in {export_time:.2f}s")
    print(f"Exported {len(df):,} documents to {OUTPUT_FILE}")
    print(f"File size optimized for fast loading by SunScore calculator")
    
    return len(df)

if __name__ == "__main__":
    export_optimized()