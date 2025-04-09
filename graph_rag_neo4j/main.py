from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
from neo4j_manager import Neo4jManager
from rag_pipeline import GraphRAG

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Graph RAG API",
    description="A FastAPI implementation of RAG with Neo4j graph database",
    version="1.0.0"
)

# Initialize Neo4j manager
neo4j_manager = Neo4jManager(
    uri=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Initialize RAG pipeline
rag = GraphRAG(
    neo4j_manager=neo4j_manager,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

class QueryInput(BaseModel):
    query: str
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7

class DocumentInput(BaseModel):
    text: str
    metadata: Optional[dict] = None

@app.post("/query")
async def query_knowledge_graph(query_input: QueryInput):
    try:
        response = rag.generate_response(
            query=query_input.query,
            max_tokens=query_input.max_tokens,
            temperature=query_input.temperature
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_document(document: DocumentInput):
    try:
        rag.ingest_document(
            text=document.text,
            metadata=document.metadata
        )
        return {"status": "success", "message": "Document ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 