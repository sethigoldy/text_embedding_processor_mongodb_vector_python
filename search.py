import time
from pymongo import MongoClient
import time
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Replace with your MongoDB connection string
client = MongoClient("your_mongodb_uri")

# Select your database and collection
db = client['your_database']
collection = db['your_collection']
dimensions = 256

class DistilBertEmbeddingGenerator:
    def __init__(self, model_name="distilbert-base-uncased"):
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained DistilBERT tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)

    def generate_embedding(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to the GPU if available
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Get the model's output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embeddings.tolist()[0][:dimensions]

generator = DistilBertEmbeddingGenerator()

# Define the aggregation pipeline
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector":  generator.generate_embedding("Elon musk"),
            "numCandidates": 10,
            "limit": 10
        }
    },
    {
        "$project": {
            "description": "$descriptions.en.value",
            "value": "$labels.en.value",
            "_id": 0
        }
    },
    {
        "$limit": 50
    }
]

# Measure the time it takes to execute the pipeline
start_time = time.time()
result = list(collection.aggregate(pipeline))
end_time = time.time()

# Print the results and the time taken
print("Aggregation Result:", result)
print("Time taken to execute:", end_time - start_time, "seconds")