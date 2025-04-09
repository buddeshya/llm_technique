from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging

class Neo4jManager:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        self.driver.close()
        
    def create_document_node(self, text: str, embeddings: List[float], metadata: Dict[str, Any] = None):
        with self.driver.session() as session:
            # Create document node with text and embeddings
            query = """
            CREATE (d:Document {
                text: $text,
                embeddings: $embeddings,
                metadata: $metadata
            })
            """
            session.run(query, text=text, embeddings=embeddings, metadata=metadata)
            
    def find_similar_documents(self, query_embedding: List[float], top_k: int = 3):
        with self.driver.session() as session:
            # Find similar documents using cosine similarity
            query = """
            MATCH (d:Document)
            WITH d, gds.similarity.cosine($query_embedding, d.embeddings) AS similarity
            ORDER BY similarity DESC
            LIMIT $top_k
            RETURN d.text AS text, d.metadata AS metadata, similarity
            """
            result = session.run(query, query_embedding=query_embedding, top_k=top_k)
            return [dict(record) for record in result]
            
    def create_relationship(self, source_id: str, target_id: str, relationship_type: str):
        with self.driver.session() as session:
            query = f"""
            MATCH (source:Document {{id: $source_id}})
            MATCH (target:Document {{id: $target_id}})
            CREATE (source)-[:{relationship_type}]->(target)
            """
            session.run(query, source_id=source_id, target_id=target_id)
            
    def get_document_by_id(self, doc_id: str):
        with self.driver.session() as session:
            query = """
            MATCH (d:Document {id: $doc_id})
            RETURN d.text AS text, d.metadata AS metadata
            """
            result = session.run(query, doc_id=doc_id)
            return result.single()
            
    def get_connected_documents(self, doc_id: str, relationship_type: str = None):
        with self.driver.session() as session:
            rel_filter = f":{relationship_type}" if relationship_type else ""
            query = f"""
            MATCH (d:Document {{id: $doc_id}})-[r{rel_filter}]-(connected:Document)
            RETURN connected.text AS text, connected.metadata AS metadata, type(r) AS relationship
            """
            result = session.run(query, doc_id=doc_id)
            return [dict(record) for record in result] 