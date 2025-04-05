import os
import base64
import io
import re
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from langchain_core.documents import Document

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def encode_image(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    """Check if the base64 data is an image"""
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

def resize_base64_image(base64_string, size=(1300, 600)):
    """Resize an image encoded as a Base64 string"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    """Split base64-encoded images and texts"""
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc)
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def search_documents(vectorstore, query: str, k: int = 5):
    """Search for relevant documents using the vector store"""
    try:
        # Use get_relevant_documents instead of similarity_search
        docs = vectorstore.get_relevant_documents(query, k=k)
        return docs
    except Exception as e:
        print(f"Error searching documents: {str(e)}")
        raise

def get_answer(query: str, docs: List[Document], image_path: str = None) -> Dict[str, Any]:
    """Generate an answer using OpenAI API"""
    # Split documents into images and texts
    split_docs = split_image_text_types(docs)
    
    # Prepare messages for OpenAI API
    messages = []
    
    # Add system message
    messages.append({
        "role": "system",
        "content": """You are an AI assistant tasked with explaining the "Attention is All You Need" paper.
        This paper introduced the Transformer architecture, which revolutionized natural language processing.
        Your goal is to help users understand the key concepts, architecture, and innovations in this paper.
        Be precise, technical, and clear in your explanations."""
    })
    
    # Add images if present
    if split_docs["images"]:
        for image in split_docs["images"]:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze this figure from the 'Attention is All You Need' paper:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                    }
                ]
            })
    
    # Add the main query with context
    formatted_texts = "\n\n".join(split_docs["texts"])
    messages.append({
        "role": "user",
        "content": f"""Based on the following context from the 'Attention is All You Need' paper, please answer this question: {query}

Context:
{formatted_texts}

Please provide a detailed, technical explanation that helps understand the paper's concepts in relation to the question."""
    })
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1024,
        temperature=0
    )
    
    return {
        "answer": response.choices[0].message.content,
        "has_image": len(split_docs["images"]) > 0
    } 