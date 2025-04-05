import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import shutil
import uvicorn

from db import create_vectorstore
from search import search_documents, get_answer

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Modal RAG API",
    description="API for querying documents with text and image support",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG system
print("Initializing RAG system...")
retriever, texts, tables = create_vectorstore()

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

class Query(BaseModel):
    question: str
    image_path: Optional[str] = None

class Response(BaseModel):
    answer: str
    has_image: bool = False

@app.post("/query", response_model=Response)
async def query_endpoint(query: Query):
    """Process a query and return an answer"""
    try:
        # Search for relevant documents
        docs = search_documents(retriever, query.question)
        
        # Get answer using the documents
        response = get_answer(query.question, docs, query.image_path)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 