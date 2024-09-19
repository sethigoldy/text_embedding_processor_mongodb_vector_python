import time
import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pymongo import UpdateOne
from torch.amp import autocast


# Set environment variable for memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# MongoDB connection details
mongo_uri = "your_mongodb_uri"
database_name = "database"
collection_name = "collection"

# Async MongoDB connection
client = AsyncIOMotorClient(mongo_uri)
db = client[database_name]
collection = db[collection_name]

# Load DistilBERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', clean_up_tokenization_spaces=True)
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
model.eval()

# Define dimension for the embedding
dimensions = 256

def generate_batch_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    for key in inputs:
        inputs[key] = inputs[key].to(device, non_blocking=True)
    
    with torch.no_grad():
        with autocast(device_type="cuda"):
            outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

async def process_batch(batch):
    def process_docs_in_batch(batch):
        texts = []
        doc_ids = []
        for doc in batch:
            description = doc.get("description", {}).get("en", {}).get("value", "")
            labels = doc.get("labels", {}).get("en", {}).get("value", "")
            doc_type = doc.get("type", "")
            dynamic_string = f"{description} {labels} {doc_type}"
            texts.append(dynamic_string)
            doc_ids.append(doc["_id"])

        embeddings = generate_batch_embeddings(texts)
        return doc_ids, embeddings

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        doc_ids, embeddings = await loop.run_in_executor(executor, process_docs_in_batch, batch)

    bulk_ops = [
        UpdateOne({"_id": doc_id}, {"$set": {"embedding": embedding[:dimensions].tolist()}})
        for doc_id, embedding in zip(doc_ids, embeddings)
    ]

    if bulk_ops:
        await collection.bulk_write(bulk_ops)

    del embeddings
    torch.cuda.empty_cache()

async def process_documents(batch_size=1000):
    filter_query = {"embedding": {"$exists": False}}
    projection = {"_id": 1, "description.en.value": 1, "labels.en.value": 1, "type": 1}
    last_id = None

    processed_count = 0
    while True:
        query = filter_query.copy()
        if last_id:
            query["_id"] = {"$gt": last_id}

        cursor = collection.find(query, projection).sort("_id", 1).limit(batch_size)
        batch = await cursor.to_list(length=batch_size)

        if not batch:
            print("No more documents to process.")
            break

        last_id = batch[-1]["_id"]
        await process_batch(batch)

        processed_count += len(batch)
        print(f"Processed {processed_count} documents.")

async def main():
    start_time = time.time()
    print("Starting the processing of documents")
    await process_documents(batch_size=4000)
    print(f"Processing complete in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())