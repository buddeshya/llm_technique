import os
import uuid
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document

from extractors import process_document
from summarizers import generate_text_summaries, generate_img_summaries

def initialize_embeddings():
    """Initialize HuggingFace embeddings"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """Create retriever that indexes summaries but returns raw content"""
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, tables, and images
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever

def create_vectorstore():
    """Create or load Chroma vector store with multi-vector retriever"""
    embeddings = initialize_embeddings()
    
    # Check if Chroma DB exists
    if not os.path.exists("./chroma_db"):
        # Load documents
        fpath = "../document/"
        fname = "attention-is-all-you-need-Paper.pdf"
        
        # Process document
        texts_4k_token, tables = process_document(fpath, fname)
        
        # Create vector store
        vectorstore = Chroma(
            collection_name="mm_rag",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Generate summaries
        text_summaries, table_summaries = generate_text_summaries(
            texts_4k_token, tables, summarize_texts=True
        )
        
        # Generate image summaries
        img_base64_list, image_summaries = generate_img_summaries("figures/")
        
        # Create retriever
        retriever = create_multi_vector_retriever(
            vectorstore,
            text_summaries,
            texts_4k_token,
            table_summaries,
            tables,
            image_summaries,
            img_base64_list
        )
        
        # Persist the vector store
        vectorstore.persist()
        
        return retriever, texts_4k_token, tables
    
    # Load existing vector store
    vectorstore = Chroma(
        collection_name="mm_rag",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Create retriever with empty summaries for existing store
    retriever = create_multi_vector_retriever(
        vectorstore,
        [],  # No text summaries
        [],  # No texts
        [],  # No table summaries
        [],  # No tables
        [],  # No image summaries
        []   # No images
    )
    
    return retriever, None, None 