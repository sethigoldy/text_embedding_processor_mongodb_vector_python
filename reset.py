from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import time

# MongoDB connection details
mongo_uri = "mongodb+srv://admin:qwerty123@cluster0.lnvox.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"  # Update with your MongoDB URI
database_name = "wikidata"
collection_name = "text"

# Retry decorator for automatic retries on failure
def retry_on_failure(max_retries=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (ConnectionFailure, PyMongoError, Exception) as e:
                    print(f"Error: {e}, Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1
            raise Exception("Max retries reached. Could not complete the operation.")
        return wrapper
    return decorator

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[database_name]
collection = db[collection_name]

@retry_on_failure()
def remove_embeddings(batch_size=10000):
    # Filter to get documents that have the `embedding` field
    filter_query = {"embedding": {"$exists": True}}

    # Count the total number of documents with embeddings
    total_docs = collection.count_documents(filter_query)
    print(f"Total documents with embeddings: {total_docs}")

    processed_count = 0

    while processed_count < total_docs:
        # Fetch a batch of documents
        documents = list(collection.find(filter_query).limit(batch_size))
        if not documents:
            print("No more documents to process.")
            break

        # Prepare bulk operations for removing the `embedding` field
        bulk_ops = [
            {"update_one": {
                "filter": {"_id": doc["_id"]},
                "update": {"$unset": {"embedding": ""}},
                "upsert": False
            }} for doc in documents
        ]
        
        # Execute bulk write to remove embeddings
        if bulk_ops:
            collection.bulk_write(bulk_ops)
        
        processed_count += len(documents)
        print(f"Processed {processed_count} documents out of {total_docs}")

# Run the embedding removal process
start_time = time.time()
remove_embeddings(batch_size=10000)  # Adjust batch size based on performance
print(f"Embedding removal complete in {time.time() - start_time:.2f} seconds")
