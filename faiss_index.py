import faiss
import numpy as np
import json

# Load the embedded data
with open("embedded_data.json", "r") as f:
    data = json.load(f)

# Create a FAISS index
d = len(data[0]['embedding'])  # Dimensionality of embeddings
index = faiss.IndexFlatL2(d)

# Add embeddings to the index
embeddings = np.array([item['embedding'] for item in data]).astype('float32')
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, "data_index.faiss")
