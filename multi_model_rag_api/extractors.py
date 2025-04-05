import os
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import CharacterTextSplitter

def extract_tables_from_pdf(path, fname):
    """Extract tables from a PDF file"""
    return partition_pdf(
        filename=os.path.join(path, fname),
        extract_images_in_pdf=True,
        infer_table_structure=True,
    )

def extract_text_from_pdf(path, fname):
    """Extract and chunk text from a PDF file"""
    return partition_pdf(
        filename=os.path.join(path, fname),
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

def categorize_elements(raw_pdf_elements):
    """Categorize extracted elements from a PDF into tables and texts"""
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables

def process_document(fpath, fname):
    """Process a PDF document and return extracted texts and tables"""
    # Extract tables and texts
    table_elements = extract_tables_from_pdf(fpath, fname)
    texts, tables = categorize_elements(table_elements)
    
    text_elements = extract_text_from_pdf(fpath, fname)
    texts, _ = categorize_elements(text_elements)
    
    # Create text splitter
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)
    
    return texts_4k_token, tables 