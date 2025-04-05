from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from db import create_vectorstore
from search import search_documents, get_answer

# Initialize FastAPI app
app = FastAPI(
    title="RAG Query API",
    description="API for answering questions using RAG system",
    version="1.0.0"
)

# Initialize vector store
vectorstore = create_vectorstore()

class Query(BaseModel):
    question: str
    k: int = 5

class Response(BaseModel):
    answer: str
    context: List[str]

@app.post("/query", response_model=Response)
async def answer_query(query: Query):
    """
    Endpoint to answer questions using the RAG system.
    """
    try:
        # Get relevant context
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
    return {"message": "RAG Query API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 