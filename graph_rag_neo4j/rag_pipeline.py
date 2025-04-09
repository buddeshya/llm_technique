from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List, Dict, Any, Optional
from neo4j_manager import Neo4jManager
import tiktoken

class GraphRAG:
    def __init__(self, neo4j_manager: Neo4jManager, openai_api_key: str):
        self.neo4j_manager = neo4j_manager
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            temperature=0.7
        )
        
    def ingest_document(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        # Generate embeddings for the document
        embeddings = self.embeddings.embed_query(text)
        
        # Store document and embeddings in Neo4j
        self.neo4j_manager.create_document_node(
            text=text,
            embeddings=embeddings,
            metadata=metadata
        )
        
    def generate_response(self, query: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        # Generate query embeddings
        query_embedding = self.embeddings.embed_query(query)
        
        # Retrieve similar documents
        similar_docs = self.neo4j_manager.find_similar_documents(
            query_embedding=query_embedding,
            top_k=3
        )
        
        # Construct prompt with context
        context = "\n\n".join([f"Context {i+1}:\n{doc['text']}" 
                             for i, doc in enumerate(similar_docs)])
        
        prompt = f"""Based on the following context, please answer the question. 
        If you cannot find the answer in the context, say so.

        {context}

        Question: {query}
        
        Answer:"""
        
        # Update LLM parameters
        self.llm.temperature = temperature
        
        # Generate response
        response = self.llm.predict(prompt)
        
        return response
        
    def _count_tokens(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
        
    def analyze_document_relationships(self, doc_id: str):
        # Get document and its connections
        doc = self.neo4j_manager.get_document_by_id(doc_id)
        connections = self.neo4j_manager.get_connected_documents(doc_id)
        
        return {
            "document": doc,
            "connections": connections
        } 