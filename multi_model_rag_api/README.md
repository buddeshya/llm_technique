# Multi-Modal RAG API

A FastAPI-based implementation of a multi-modal RAG system for answering questions about the "Attention is All You Need" paper. The system supports both text-based queries and image analysis.

## Features

- Document-based question answering using RAG
- Image support for multi-modal queries via GPT-4 Vision
- Chroma DB for vector storage
- FastAPI with automatic API documentation
- CORS support
- Health check endpoint
- File upload handling for images

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Ensure your documents are in the `../document/` directory:
```
document/
  └── attention-is-all-you-need-Paper.pdf
```

## Running the API

Start the server with:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /query
Main endpoint for querying the RAG system.

Request body:
```json
{
    "question": "What is the multi-head attention mechanism?",
    "k": 5  // Optional: number of documents to retrieve
}
```

Optional: Upload an image file using multipart/form-data

Response:
```json
{
    "answer": "Detailed answer to the question...",
    "context": [
        {
            "content": "Relevant document content...",
            "metadata": {}
        }
    ],
    "has_image": true
}
```

### GET /health
Health check endpoint.

Response:
```json
{
    "status": "healthy"
}
```

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Notes

- The API will automatically create a new Chroma DB if one doesn't exist, or use the existing one
- Images are temporarily stored in the `uploads` directory and automatically cleaned up after processing
- The system uses GPT-4 Vision for image analysis when images are provided 