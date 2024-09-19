# MongoDB Document Embedding and Search

This repository contains two Python scripts that demonstrate how to generate embeddings using the DistilBERT model and process documents in a MongoDB database. The goal is to embed text data and perform vector searches to retrieve relevant documents.

## Overview

1. **main.py**: 
   - Connects to MongoDB asynchronously.
   - Uses the DistilBERT model to generate embeddings for documents and stores them back in MongoDB.
   - Processes documents in batches and updates the embeddings field in the database.

2. **search.py**:
   - Retrieves documents from MongoDB by performing vector search based on embeddings generated from the DistilBERT model.
   - Utilizes MongoDB's `$vectorSearch` aggregation to find similar documents based on the query text.

## Dataset

We have used **Wikidata**, an open-source dataset, as the source of the documents for generating embeddings and performing vector search. You can download the dataset from the following link:

[Wikidata Download](https://academictorrents.com/details/0852ef544a4694995fcbef7132477c688ded7d9a)

Wikidata is a large dataset with rich textual descriptions, making it ideal for embedding and searching tasks. Make sure to preprocess the dataset as per your requirements before loading it into MongoDB.

## Performance

When generating embeddings using a **NVIDIA RTX 3090** GPU, this process can handle approximately **10,000 documents every 12 seconds**. This performance benchmark provides a general guideline for estimating the time needed to process larger datasets depending on your hardware capabilities.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd <repository-directory>
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running `main.py`
This script connects to MongoDB, processes documents, and generates embeddings using DistilBERT. It stores the embeddings back in the MongoDB collection.

Run the script:
```bash
python main.py
```

Make sure to replace the MongoDB connection URI, database, and collection names as needed.

### Running `search.py`
This script performs a vector search based on a query string using the generated embeddings and retrieves similar documents from the MongoDB collection.

Run the script:
```bash
python search.py
```

Again, replace the MongoDB connection URI, database, and collection names as needed.

## MongoDB Configuration

1. Ensure you have a MongoDB collection that stores your documents with the following structure:
   ```json
   {
       "_id": ObjectId,
       "description": { "en": { "value": "document description" } },
       "labels": { "en": { "value": "document labels" } },
       "type": "document type",
       "embedding": [ ... ]  // Generated embeddings will be stored here
   }
   ```

2. You should also create a vector index on the `embedding` field for efficient vector search.

## Requirements

- Python 3.8 or higher
- MongoDB
- CUDA-enabled GPU (for faster processing, but not required)

## Key Dependencies

- `torch`: PyTorch framework used for handling the DistilBERT model and GPU acceleration.
- `transformers`: Library for loading the pre-trained DistilBERT model and tokenizer.
- `motor`: Asynchronous MongoDB client for `main.py`.
- `pymongo`: MongoDB client used in `search.py`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
