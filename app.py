from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from openai import OpenAI
import faiss
import json
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()  # Load environment variables from .env

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

client = OpenAI(api_key=openai_api_key)

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for your front-end (you can be more specific with origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's address
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load FAISS index
index = faiss.read_index("data_index.faiss")

# Load the original data
with open("embedded_data.json", "r") as f:
    data = json.load(f)

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def retrieve_relevant_data(query, k=3):
    # Generate embedding for the query
    query_embedding = get_embedding(query)

    # Perform similarity search
    _, indices = index.search(np.array([query_embedding]).astype('float32'), k)

    # Retrieve the top-k results
    return [data[i] for i in indices[0]]

def check_relevance(query: str, context: str) -> bool:
    """
    Use OpenAI API to classify whether a question is relevant to the user's profile scope.
    """
    if "guilherme" in query.lower() or "marcon" in query.lower(): return True

    try:
        relevance_prompt = (
            "You are a living curriculum vitae trying to determine whether a question is relevant to Guilherme Marcon's professional profile. "
            "Here's the context retrieved from the database about the question: "
            "{context} "
            "Respond with only 'Relevant' if the question is within this scope, or only 'Irrelevant' if it is not.\n\n"
            f"Question: {query}\n"
            "Response:"
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": relevance_prompt}]
        )

        # Extract the model's response
        relevance_response = completion.choices[0].message.content.strip().lower()
        print(f"Response: {relevance_response}")
        return relevance_response != "irrelevant"
    except Exception as e:
        print(f"Error during relevance check: {e}")
        return False  # Default to not relevant on failure

# Request model for chatbot input
class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_bot(request: MessageRequest):
    user_message = request.message
    try:
        # Retrieve relevant data - RAG
        relevant_data = retrieve_relevant_data(user_message)
        context = "\n".join([item['description'] for item in relevant_data])

        # # Check if the question is relevant
        # if not check_relevance(user_message, context):
        #     bot_reply = "I'm sorry, but I can only answer questions related to the Guilherme's skills, projects, education and career experience."
        #     return {"reply": bot_reply}

        # Use the updated OpenAI chat-based API
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": f"You are a living curriculum vitae. Your job is to answer questions about Guilherme Marcon. Here is what you know: {context}. Refuse to answer questions out of scope."},
                {"role": "user", "content": user_message},
            ]
        )

        bot_reply = completion.choices[0].message.content.strip()
        return {"reply": bot_reply}
    except Exception as e:
        return {"error": str(e)}

# Serve other static content (e.g., profile, projects)
@app.get("/profile")
async def get_profile():
    return {
        "name": "Guilherme Marcon",
        "skills": ["Python", "Machine Learning", "NLP", "FastAPI"],
        "projects": [
            {"name": "ConSentiment", "description": "Sentiment analysis project with consensus-based models"},
            # Add other projects here
        ]
    }
