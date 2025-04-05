from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from db import create_vectorstore
from search import search_documents, get_answer

# Initialize FastAPI app
app = FastAPI(
    title="Retrieve and Rerank API",
    description="API for answering questions using retrieve and rerank system",
    version="1.0.0"
)

# Initialize vector store
vectorstore = create_vectorstore()

class Query(BaseModel):
    question: str
    k: int = 10  # Number of documents to retrieve before reranking

class Response(BaseModel):
    answer: str
    context: List[str]

@app.post("/query", response_model=Response)
async def answer_query(query: Query):
    """
    Endpoint to answer questions using the retrieve and rerank system.
    The system will:
    1. Retrieve k documents using similarity search
    2. Rerank the documents using a cross-encoder model
    3. Use the top 5 reranked documents to generate an answer
    """
    try:
        # Get relevant context (includes reranking)
        context = search_documents(vectorstore, query.question, query.k)
        
        # Generate answer
        answer = get_answer(query.question, context)
        
        return Response(
            answer=answer,
            context=context
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Retrieve and Rerank API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 