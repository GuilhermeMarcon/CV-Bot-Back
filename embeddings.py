import json
from openai import OpenAI
client = OpenAI()

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

with open("data.json", "r") as f:
    data = json.load(f)

for item in data:
    print(item['title'])
    item['embedding'] = get_embedding(item['description'])

with open("embedded_data.json", "w") as f:
    json.dump(data, f)